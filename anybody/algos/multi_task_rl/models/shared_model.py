import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from tensordict import TensorDict
from anybody.cfg import cfg

from skrl.models.torch import Model
from skrl.models.torch.gaussian import GaussianMixin
from skrl.models.torch.deterministic import DeterministicMixin

from .model_utils import (TransformerModel, MLP, MLP_Residual,
                          ResidualMLP_Cartpole, ResidualMLP_Cartpole_Combined,
                          MLP_LatentResidual
)

import gym
import gymnasium

torch.set_printoptions(precision=3, sci_mode=False)

def get_policy_network(net_out_dim=1):
    num_actions = cfg.BENCH.MAX_NUM_LINKS
    if "cartpole" in cfg.MULTIENV.ROBOTS:
        num_actions = 1

    n_robots = len(cfg.MULTIENV.ROBOTS)
    
    output_tanh = cfg.ACTION.ABSOLUTE
    
    if cfg.MODEL.TYPE in ["transformer", "mixed"]:
        mu_net = TransformerModel(
            net_out_dim, only_limbs=True, n_robots=n_robots, output_tanh=output_tanh
        )

    elif cfg.MODEL.TYPE == "mlp":
        mu_net = MLP(
            include_std=cfg.MODEL.OUTPUT_STD, n_robots=n_robots, output_tanh=output_tanh
        )

    elif cfg.MODEL.TYPE == "mlp-res":
        mu_net = MLP_Residual(
            include_std=cfg.MODEL.OUTPUT_STD, n_robots=n_robots, output_tanh=output_tanh
        )
        

    elif cfg.MODEL.TYPE == "mlp-res-latent":
        mu_net = MLP_LatentResidual(
            include_std=cfg.MODEL.OUTPUT_STD, n_robots=n_robots, output_tanh=output_tanh
        )

    elif cfg.MODEL.TYPE == "cartpole-mlp":
        mlp_hidden_layers = [cfg.CARTPOLE.MLP_HIDDEN_DIMS] * cfg.CARTPOLE.MLP_N_LAYERS
        mu_net = ResidualMLP_Cartpole(
            dim_list=[cfg.CARTPOLE.OBS_DIM] + mlp_hidden_layers + [1],
            final_nonlinearity=False,
            nonlinearity="relu",
        )

    if (not cfg.MODEL.OUTPUT_STD) and (cfg.AGENT_NAME != "td3"):
        log_std_parameter = nn.Parameter(
            torch.ones(num_actions) * cfg.MODEL.POLICY.INITIAL_LOG_STD
        )
    else:
        # td3 does not use log_std - it uses deterministic policy
        log_std_parameter = None

    return mu_net, log_std_parameter


def get_value_network():
    n_robots = len(cfg.MULTIENV.ROBOTS)

    # assert (
    #     not cfg.MODEL.OUTPUT_STD
    # ), "Outputting std is not supported for shared policy-value model"
    if cfg.MODEL.TYPE == "transformer":
        v_net = TransformerModel(
            1, only_limbs=False, n_robots=n_robots, output_tanh=False
        )

    elif cfg.MODEL.TYPE == "mlp":
        v_net = MLP(n_robots=n_robots, output_tanh=False)

    elif cfg.MODEL.TYPE == "mlp-res":
        v_net = MLP_Residual(n_robots=n_robots, output_tanh=False)
        
    elif cfg.MODEL.TYPE == "mlp-res-latent":
        v_net = MLP_LatentResidual(n_robots=n_robots, output_tanh=False)

    elif cfg.MODEL.TYPE == "mixed":
        v_net = MLP_Residual(n_robots=n_robots, output_tanh=False)

    elif cfg.MODEL.TYPE == "cartpole-mlp":
        mlp_hidden_layers = [cfg.CARTPOLE.MLP_HIDDEN_DIMS] * cfg.CARTPOLE.MLP_N_LAYERS
        v_net = ResidualMLP_Cartpole(
            dim_list=[cfg.CARTPOLE.OBS_DIM]
            + mlp_hidden_layers
            + [cfg.CARTPOLE.OBS_DIM],
            final_nonlinearity=False,
            nonlinearity="gelu",
        )

    # initialize the parameters of value network to be very small
    for p in v_net.parameters():
        p.data.fill_(1e-5)

    return v_net


def get_qval_network():
    n_robots = len(cfg.MULTIENV.ROBOTS)

    if cfg.MODEL.TYPE == "transformer":
        v_net = TransformerModel(
            1, only_limbs=False, n_robots=n_robots, output_tanh=False, combined=True,
            decoder_kwargs={"final_nonlinearity": False, "nonlinearity": "gelu"}
        )

    elif cfg.MODEL.TYPE == "mlp":
        v_net = MLP(n_robots=n_robots, output_tanh=False, combined=True, 
                    mlp_kwargs={"final_nonlinearity": False, "nonlinearity": "gelu"})

    elif cfg.MODEL.TYPE == "mlp-res":
        v_net = MLP_Residual(n_robots=n_robots, output_tanh=False, combined=True, 
                             mlp_kwargs={"final_nonlinearity": False, "nonlinearity": "gelu"})

    elif cfg.MODEL.TYPE == "mixed":
        v_net = MLP_Residual(n_robots=n_robots, output_tanh=False, combined=True, 
                             mlp_kwargs={"final_nonlinearity": False, "nonlinearity": "gelu"}
                             )

    elif cfg.MODEL.TYPE == "cartpole-mlp":
        mlp_hidden_layers = [cfg.CARTPOLE.MLP_HIDDEN_DIMS] * cfg.CARTPOLE.MLP_N_LAYERS
        v_net = ResidualMLP_Cartpole_Combined(
            dim_list=[cfg.CARTPOLE.OBS_DIM + 4]
            + mlp_hidden_layers
            + [cfg.CARTPOLE.OBS_DIM],
            final_nonlinearity=True,
            nonlinearity="gelu",
        )

    return v_net


class StochasticPolicy_DiscreteAction(Model):
    def __init__(
        self, observation_space, action_space, device, roles=["policy"]
    ):
        Model.__init__(self, observation_space=observation_space, action_space=action_space, device=device)
        self.roles = roles

        # use binomial distribution for discrete actions, and compute the 
        # final action using the weighted average
        # here we assume the network is transformer network
        
        assert cfg.MODEL.TYPE == "transformer", "Only transformer network is supported for discrete actions"
        assert cfg.MODEL.OUTPUT_STD, "MODEL.OUTPUT_STD must be True to avoid initializing log_std_parameter"
        
        self.logits_net, _ = get_policy_network(net_out_dim=cfg.ACTION.NUM_BINS)
        self.inv_temp = nn.Parameter(torch.tensor([1/cfg.MODEL.POLICY.INITIAL_TEMPERATURE]))
        
        self.laplace_const = 1e-5
        self.n_limbs = cfg.BENCH.MAX_NUM_LINKS  
        
        reduction = "sum"
        self._reduction = (
            torch.mean
            if reduction == "mean"
            else torch.sum if reduction == "sum" else torch.prod if reduction == "prod" else None
        )

    def compute(self, inputs, role):
        # policy computation
        obs = inputs["states"]
        if cfg.MODEL.NORMALIZE_OBS:
            processed_obs = self.normalize_obs(obs)
        else:
            processed_obs = obs

        if cfg.MODEL.USE_CACHED_OBS:
            self._shared_preprocessed_obs = processed_obs

        logits, mu_attention_maps = self.logits_net(processed_obs)
        bs = logits.shape[0]
        
        logits = logits.view(bs, self.n_limbs, cfg.ACTION.NUM_BINS)
        
        assert logits.shape == (bs, self.n_limbs, cfg.ACTION.NUM_BINS), f"Logits shape: {logits.shape}"
        return logits, {"attention_maps": mu_attention_maps}
        
        
        
    def distribution(self, role):
        return self._distribution
        
        
    def act(self, inputs, role, return_entropy=False):
        # map from states/observations to mean actions and log standard deviations

        bin_logits, outputs = self.compute(inputs, "policy")
        act_mask = inputs["states"]["act_mask"].bool()

        # print(bin_logits[0])
        # import pdb;     pdb.set_trace()

        # distribution
        bin_logits = bin_logits * (self.inv_temp.abs() + self.laplace_const)
        self._distribution = torch.distributions.Categorical(logits=bin_logits)
        # self._distribution = torch.distributions.Binomial(total_count=1, logits=bin_logits)
        actions = self._distribution.sample() # rsample for binomial doesn't exist.

        bs = actions.shape[0]
        log_prob_actions = actions
        if "taken_actions" in inputs:
            # action_shape = inputs["taken_actions"].shape
            # new_shape = action_shape[:-1] + (self.n_limbs, cfg.ACTION.NUM_BINS)
            # log_prob_actions = inputs["taken_actions"].view(new_shape)
            log_prob_actions = inputs["taken_actions"]        

        # log of the probability density function
        log_prob = self._distribution.log_prob(log_prob_actions)
        entropy = self._distribution.entropy()

        # assert actions.shape[1:] == (
        #     self.n_limbs, cfg.ACTION.NUM_BINS
        # ), f"Actions shape: {actions.shape}"


        # need to flatten actions, and 
        # for log prob and entropy need to sum over the bins
        # do the same for random act

        act_mask = act_mask[..., 0]
                
        log_prob = torch.where(act_mask, log_prob, torch.zeros_like(log_prob))
        entropy = torch.where(act_mask, entropy, torch.zeros_like(entropy))
        
        # log_prob = log_prob.sum(dim=-1)
        # entropy = entropy.sum(dim=-1)
        
        log_prob = log_prob.view(-1, self.num_actions)
        entropy = entropy.view(-1, self.num_actions)

        bs = log_prob.shape[0]
        assert log_prob.shape == (bs, self.num_actions), f"Log prob shape: {log_prob.shape}"
        assert entropy.shape == (bs, self.num_actions), f"Entropy shape: {entropy.shape}"


        actions = torch.where(act_mask, actions, torch.zeros_like(actions))

        bs = actions.shape[0]
        actions = actions.view(bs, self.num_actions)

        if self._reduction is not None:
            log_prob = self._reduction(log_prob, dim=-1)
        if log_prob.dim() != actions.dim():
            log_prob = log_prob.unsqueeze(-1)

        outputs["logits"] = bin_logits

        if return_entropy:
            return actions, log_prob, outputs, entropy.mean()
        else:
            return actions, log_prob, outputs
        
             
    def random_act(self, inputs, role, return_entropy=False):
        logits = torch.ones(inputs["states"]["act_mask"].shape[:-1] + (cfg.ACTION.NUM_BINS,)) * 0.5
        
        bs = logits.shape[0]
        # assert logits.shape == (bs, self.n_limbs, cfg.ACTION.NUM_BINS), f"Logits shape: {logits.shape}, expected: {(bs, self.num_actions, cfg.ACTION.NUM_BINS)}"
        logits = logits.to(self.device)
        
        
        self._distribution = torch.distributions.Categorical(logits=logits)
        # self._distribution = torch.distributions.Binomial(total_count=1, logits=logits)
        
        actions = self._distribution.sample()
        log_prob = self._distribution.log_prob(actions)
        entropy = self._distribution.entropy()

        # assert actions.shape[1:] == (
        #     self.n_limbs, cfg.ACTION.NUM_BINS   
        # ), f"Actions shape: {actions.shape}"
        
        assert entropy.shape == log_prob.shape, f"Entropy shape: {entropy.shape}, log_prob shape: {log_prob.shape}"
        
        act_mask = inputs["states"]["act_mask"].bool()  
        act_mask = act_mask[..., 0]
        
        log_prob = torch.where(act_mask, log_prob, torch.zeros_like(log_prob))
        entropy = torch.where(act_mask, entropy, torch.zeros_like(entropy))
        
        # bs = log_prob.shape[0]
        # log_prob = log_prob.view(bs, self.num_actions)    
        # entropy = entropy.view(bs, self.num_actions)

        actions = torch.where(act_mask, actions, torch.zeros_like(actions))
        actions = actions.view(bs, self.num_actions)
        
        if self._reduction is not None:
            log_prob = self._reduction(log_prob, dim=-1)
        if log_prob.dim() != actions.dim():
            log_prob = log_prob.unsqueeze(-1)
        
        if return_entropy:
            return actions, log_prob, {}, entropy.mean()
        else:
            return actions, log_prob, {}
        
    
    def get_entropy(self, role: str = "") -> torch.Tensor:
        if self._distribution is None:
            return torch.tensor(0.0, device=self.device)
        return self._distribution.entropy().to(self.device)
    

# self._g_clip_actions = clip_actions and isinstance(self.action_space, gymnasium.Space)

#         if self._g_clip_actions:
#             self._g_clip_actions_min = torch.tensor(self.action_space.low, device=self.device, dtype=torch.float32)
#             self._g_clip_actions_max = torch.tensor(self.action_space.high, device=self.device, dtype=torch.float32)

#         self._g_clip_log_std = clip_log_std
#         self._g_log_std_min = min_log_std
#         self._g_log_std_max = max_log_std

#         self._g_log_std = None
#         self._g_num_samples = None
#         self._g_distribution = None


class StochasticPolicy(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        roles=["policy"],
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self,
            clip_actions=cfg.MODEL.POLICY.CLIP_ACTIONS,
            clip_log_std=cfg.MODEL.POLICY.CLIP_LOG_STD,
            min_log_std=cfg.MODEL.POLICY.MIN_LOG_STD,
            max_log_std=cfg.MODEL.POLICY.MAX_LOG_STD,
            role=roles[0],
        )
        
        if (not hasattr(self, "_reduction")) and hasattr(self, "_g_reduction"):
            self._reduction = self._g_reduction
            self._clip_log_std = self._g_clip_log_std
            self._log_std_min = self._g_log_std_min
            self._log_std_max = self._g_log_std_max
            self._clip_actions = self._g_clip_actions
            if self._clip_actions:
                self._clip_actions_min = self._g_clip_actions_min
                self._clip_actions_max = self._g_clip_actions_max
                
            self._log_std = self._g_log_std
            self._num_samples = self._g_num_samples
            self._distribution = self._g_distribution
                
        # self._reduction = (
        #     torch.mean
        #     if reduction == "mean"
        #     else torch.sum if reduction == "sum" else torch.prod if reduction == "prod" else None
        # )
        self._roles = roles
        self._shared_preprocessed_obs = None

        if "cartpole" not in cfg.MULTIENV.ROBOTS:
            num_actions = cfg.BENCH.MAX_NUM_LINKS
            assert (
                num_actions == self.num_actions
            ), f"Number of actions: {num_actions}, num_actions read from action space: {self.num_actions}"

        self.mu_net, self.log_std_parameter = get_policy_network()

    def compute(self, inputs, role):
        # policy computation
        obs = inputs["states"]
        if cfg.MODEL.NORMALIZE_OBS:
            # if not cfg.EVAL:
            #     self.update_normalizers(obs)

            processed_obs = self.normalize_obs(obs)
        else:
            processed_obs = obs

        if cfg.MODEL.USE_CACHED_OBS:
            self._shared_preprocessed_obs = processed_obs

        # for k, v in obs.items():
        #     obs[k] = v.to(self.device)

        mu_log_std, mu_attention_maps = self.mu_net(
            processed_obs
        )
        
        if cfg.MODEL.OUTPUT_STD:
            reshaped_mu_log_std = mu_log_std.view(-1, self.num_actions, 2)
            mu = reshaped_mu_log_std[:, :, 0]
            log_std = reshaped_mu_log_std[:, :, 1]
        else:
            mu = mu_log_std
            log_std = self.log_std_parameter

        # clamp log standard deviations
        # clamp log standard deviations
        if self._clip_log_std:
            log_std = torch.clamp(log_std, self._log_std_min, self._log_std_max)

        # if any value in mu is nan, stop
        assert not torch.isnan(mu).any(), "Mu has nan values"
        return (
            mu,
            log_std,
            {"attention_maps": mu_attention_maps, "act_mask": obs["act_mask"]},
        )

    def gaussian_act(self, inputs, return_entropy=False):
        # map from states/observations to mean actions and log standard deviations

        mean_actions, log_std, outputs = self.compute(inputs, "policy")

        # wandb.log({"actions (max)": mean_actions.max().item(), "actions (min)": mean_actions.min().item()})

        # clamp log standard deviations
        if self._clip_log_std:
            log_std = torch.clamp(log_std, self._log_std_min, self._log_std_max)

        self._log_std = log_std
        self._num_samples = mean_actions.shape[0]

        act_mask = outputs["act_mask"].bool()

        # distribution
        self._distribution = Normal(mean_actions, log_std.exp())

        # sample using the reparameterization trick
        actions = self._distribution.sample()

        # log of the probability density function
        log_prob = self._distribution.log_prob(inputs.get("taken_actions", actions))
        entropy = self._distribution.entropy()

        assert actions.shape[1:] == (
            self.num_actions,
        ), f"Actions shape: {actions.shape}"

        log_prob = torch.where(act_mask[..., 0], log_prob, torch.zeros_like(log_prob))
        entropy = torch.where(act_mask[..., 0], entropy, torch.zeros_like(entropy))

        if self._reduction is not None:
            log_prob = self._reduction(log_prob, dim=-1)
        if log_prob.dim() != actions.dim():
            log_prob = log_prob.unsqueeze(-1)

        outputs["mean_actions"] = mean_actions

        actions = torch.where(act_mask[..., 0], actions, torch.zeros_like(actions))

        


        if return_entropy:
            return actions, log_prob, outputs, entropy.mean()
        else:
            return actions, log_prob, outputs

    def get_entropy(self, role: str = "") -> torch.Tensor:
        if self._distribution is None:
            return torch.tensor(0.0, device=self.device)
        return self._distribution.entropy().to(self.device)

    def random_act(self, inputs, role: str = "", return_entropy=False):
        """Act randomly according to the action space

        :param inputs: Model inputs. The most common keys are:

                       - ``"states"``: state of the environment used to make the decision
                       - ``"taken_actions"``: actions taken by the policy for the given states
        :type inputs: dict where the values are typically torch.Tensor
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        :raises NotImplementedError: Unsupported action space

        :return: Model output. The first component is the action to be taken by the agent
        :rtype: tuple of torch.Tensor, None, and dict
        """
        if isinstance(inputs["states"], (dict, TensorDict)):
            bs = list(inputs["states"].values())[0].shape[0]
        else:
            bs = inputs["states"].shape[0]

        if issubclass(type(self.action_space), gym.spaces.Box) or issubclass(
            type(self.action_space), gymnasium.spaces.Box
        ):
            if self._random_distribution is None:
                self._random_distribution = torch.distributions.uniform.Uniform(
                    low=torch.tensor(
                        cfg.MODEL.POLICY.ACTION_MIN,
                        device=self.device,
                        dtype=torch.float32,
                    ),
                    high=torch.tensor(
                        cfg.MODEL.POLICY.ACTION_MAX,
                        device=self.device,
                        dtype=torch.float32,
                    ),
                )

            actions = self._random_distribution.sample(
                sample_shape=(bs, self.num_actions)
            )
            act_mask = inputs["states"]["act_mask"].bool()

            # log of the probability density function
            log_prob = self._random_distribution.log_prob(
                inputs.get("taken_actions", actions)
            )

            # mask out the log_prob for the actions that are not valid
            assert actions.shape[1:] == (
                self.num_actions,
            ), f"Actions shape: {actions.shape}"

            log_prob[~act_mask[..., 0]] = 0.0

            if self._reduction is not None:
                log_prob = self._reduction(log_prob, dim=-1)
            if log_prob.dim() != actions.dim():
                log_prob = log_prob.unsqueeze(-1)

            if return_entropy:
                entropy = self._random_distribution.entropy()
                entropy[~act_mask[..., 0]] = 0.0
                return actions, log_prob, {}, entropy.mean()

            return actions, log_prob, {}
            # return self._random_distribution.sample(sample_shape=(bs, self.num_actions)), None, {}
        else:
            raise NotImplementedError(
                f"Action space type ({type(self.action_space)}) not supported"
            )

    def act(self, inputs, role, return_entropy=False):
        return self.gaussian_act(inputs, return_entropy)

    def distribution(self, role):
        return self._distribution


class DeterministicPolicy(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=cfg.MODEL.POLICY.CLIP_ACTIONS)

        if (not hasattr(self, "_clip_actions")) and hasattr(self, "_g_clip_actions"):
            self._clip_actions = self._g_clip_actions
            self._clip_actions_min = self._g_clip_actions_min
            self._clip_actions_max = self._g_clip_actions_max

        self.mu_net, log_std = get_policy_network()

        assert (
            log_std is None
        ), "Deterministic policy should not have separate log_std parameter"

    def compute(self, inputs, role):
        # policy computation
        obs = inputs["states"]
        processed_obs = obs

        mu, mu_attention_maps = self.mu_net(processed_obs)

        # if any value in mu is nan, stop
        assert not torch.isnan(mu).any(), "Mu has nan values"
        return (
            mu,
            {"attention_maps": mu_attention_maps, "act_mask": obs["act_mask"]},
        )


    def random_act(self, inputs, role: str = "", return_entropy=False):
        """Act randomly according to the action space

        :param inputs: Model inputs. The most common keys are:

                       - ``"states"``: state of the environment used to make the decision
                       - ``"taken_actions"``: actions taken by the policy for the given states
        :type inputs: dict where the values are typically torch.Tensor
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        :raises NotImplementedError: Unsupported action space

        :return: Model output. The first component is the action to be taken by the agent
        :rtype: tuple of torch.Tensor, None, and dict
        """

        if isinstance(inputs["states"], dict):
            bs = list(inputs["states"].values())[0].shape[0]
        else:
            bs = inputs["states"].shape[0]

        if issubclass(type(self.action_space), gym.spaces.Box) or issubclass(
            type(self.action_space), gymnasium.spaces.Box
        ):
            if self._random_distribution is None:
                self._random_distribution = torch.distributions.uniform.Uniform(
                    low=torch.tensor(
                        self.action_space.low[0],
                        device=self.device,
                        dtype=torch.float32,
                    ),
                    high=torch.tensor(
                        self.action_space.high[0],
                        device=self.device,
                        dtype=torch.float32,
                    ),
                )

            actions = self._random_distribution.sample(
                sample_shape=(bs, self.num_actions)
            )
            return actions, {}
            # return self._random_distribution.sample(sample_shape=(bs, self.num_actions)), None, {}
        else:
            raise NotImplementedError(
                f"Action space type ({type(self.action_space)}) not supported"
            )


class Value(DeterministicMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        roles=["value"],
    ):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(
            self, clip_actions=False, role=roles[-1]
        )

        self._roles = roles
        self._shared_preprocessed_obs = None
        self.v_net = get_value_network()

    def compute(self, inputs, role):
        processed_obs = inputs["states"]
        vals, v_attention_maps = self.v_net(processed_obs)
        assert not torch.isnan(vals).any(), "Vals has nan values"

        obs_mask = processed_obs["obs_mask"].bool()[..., 0]
        n_limbs = torch.sum(obs_mask.int(), dim=1, keepdim=True)

        if torch.any(n_limbs == 0):
            raise ValueError("All limbs are masked out")

        if cfg.MODEL.TYPE == "transformer":
            vals = torch.where(obs_mask, vals, torch.zeros_like(vals))

        vals = torch.sum(vals, dim=1, keepdim=True) / n_limbs
        return vals, {"attention_maps": v_attention_maps}

    def act(self, inputs, role, return_entropy=False):
        return self.deterministic_act(inputs)

    def deterministic_act(self, inputs):
        # map from observations/states to actions
        values, outputs = self.compute(inputs, "value")

        # wandb.log(
        #     {"values (max)": values.max().item(), "values (min)": values.min().item()}
        # )

        # do not clip values for value function, only clip actions for policy function
        # clip actions
        # if self._clip_actions:
        #     values = torch.clamp(
        #         values, min=cfg.MODEL.VALUE.VALUE_MIN, max=cfg.MODEL.VALUE.VALUE_MAX
        #     )
            # actions = torch.clamp(values, min=self._clip_actions_min, max=self._clip_actions_max)

        return values, None, outputs


class QValueFunc(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)

        self.q_net = get_qval_network()
        
    def compute(self, inputs, role):
        processed_obs = inputs["states"]
        actions = inputs["taken_actions"]
        vals, v_attention_maps = self.q_net(processed_obs, actions)
        assert not torch.isnan(vals).any(), "Vals has nan values"

        obs_mask = processed_obs["obs_mask"].bool()[..., 0]
        n_limbs = torch.sum(obs_mask.int(), dim=1, keepdim=True)

        if torch.any(n_limbs == 0):
            raise ValueError("All limbs are masked out")

        if cfg.MODEL.TYPE == "transformer":
            vals = torch.where(obs_mask, vals, torch.zeros_like(vals))

        vals = torch.sum(vals, dim=1, keepdim=True) / n_limbs
        return vals, {"attention_maps": v_attention_maps}
