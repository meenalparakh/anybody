import gymnasium as gym

import numpy as np
import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO_DEFAULT_CONFIG
from skrl.agents.torch.ppo import PPO_RNN as PPO
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
import torch.nn.init as nn_init
import math
from anybody.cfg import cfg
from anybody.algos.multi_task_rl.models.model_utils import (
    PositionalEncoding,
    TransformerEncoderLayerResidual,
    make_mlp_default,
    TransformerEncoder,
    ResidualMLP,
    ObsFeatureExtractor as FeatureExtractor,
)
from torch.distributions.normal import Normal


def init_weights(net):
    gain = nn_init.calculate_gain("relu")
    for layer in net:
        if hasattr(layer, "weight") and layer.weight is not None:
            nn_init.orthogonal_(layer.weight.data, gain=gain)
        if hasattr(layer, "bias") and layer.bias is not None:
            nn_init.constant_(layer.bias.data, 0)

    # initialize the last layer of the decoder to be normal with scale 0.01
    nn_init.normal_(net[-1].weight.data, mean=0, std=0.01)


class RNNTransformerEncDec(nn.Module):
    def __init__(self, decoder_out_dim: int, only_limbs=False):
        super().__init__()
        # Transformer Encoder

        self.only_limbs = only_limbs
        self.model_args = cfg.MODEL.TRANSFORMER
        self.d_model = cfg.MODEL.RNN.FEATURE_DIM + cfg.MODEL.RNN.HIDDEN_SIZE

        self.inp_seq_len = cfg.BENCH.MAX_NUM_LINKS + 3
        self.positional_encoding = PositionalEncoding(
            self.d_model, self.inp_seq_len, batch_first=True
        )

        encoder_layers = TransformerEncoderLayerResidual(
            # cfg.MODEL.LIMB_EMBED_SIZE,
            self.d_model,
            self.model_args.NHEAD,
            self.model_args.DIM_FEEDFORWARD,
            self.model_args.DROPOUT,
        )

        self.transformer_encoder = TransformerEncoder(
            encoder_layers,
            self.model_args.NLAYERS,
            norm=None,
        )

        # Map encoded observations to per node action mu or critic value
        decoder_input_dim = self.d_model

        self.decoder = make_mlp_default(
            [decoder_input_dim]
            + [self.model_args.DECODER_DIM] * self.model_args.N_DECODER_LAYERS
            + [decoder_out_dim],
            final_nonlinearity=False,
            nonlinearity="relu",
            dropout=self.model_args.DROPOUT,
        )

        self.init_weights()

    def init_weights(self):
        gain = nn_init.calculate_gain("relu")
        # self.decoder.init_weights()
        # Initialize decoder weights and biases
        for layer in self.decoder:
            if hasattr(layer, "weight") and layer.weight is not None:
                nn_init.orthogonal_(layer.weight.data, gain=gain)
            if hasattr(layer, "bias") and layer.bias is not None:
                nn_init.constant_(layer.bias.data, 0)

        # Initialize transformer encoder layers
        for layer in self.transformer_encoder.layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn_init.orthogonal_(param.data, gain=gain)
                else:
                    nn_init.constant_(param.data, 0)

        # initialize the last layer of the decoder to be normal with scale 0.01
        nn_init.normal_(self.decoder[-1].weight.data, mean=0, std=0.01)

        self.decoder[-1].bias.data.zero_()
        initrange = self.model_args.DECODER_INIT
        self.decoder[-1].weight.data.uniform_(-initrange, initrange)

    def forward(self, tokens, token_masks, return_attention=False):
        batch_size = tokens.shape[0]
        # token_masks = token_masks[..., 0]

        tokens = self.positional_encoding(tokens)

        # transforming the observations to (sequence, batch_size, d_model)
        observation_embed = tokens.permute(1, 0, 2)

        if return_attention:
            obs_embed_t, attention_maps = self.transformer_encoder.get_attention_maps(
                observation_embed, src_key_padding_mask=(~token_masks)
            )
        else:
            # (num_limbs, batch_size, d_model)
            obs_embed_t = self.transformer_encoder(
                observation_embed, src_key_padding_mask=(~token_masks)
            )
            attention_maps = None

        decoder_input = obs_embed_t
        # if "hfield" in cfg.ENV.KEYS_TO_KEEP and self.ext_feat_fusion == "late":
        #     decoder_input = torch.cat([decoder_input, hfield_obs], axis=2)

        # (num_limbs, batch_size, J)
        output = self.decoder(decoder_input)
        if self.only_limbs:
            output = output[
                1:-2, ...
            ]  # first is the base, and the last two are goal and obj, ignore them for actions

        # (batch_size, num_limbs, J)
        output = output.permute(1, 0, 2)
        # (batch_size, num_limbs * J)
        output = output.reshape(batch_size, -1)

        return output, attention_maps

 
class PolicyRNN(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        reduction="sum",
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self,
            cfg.MODEL.POLICY.CLIP_ACTIONS,
            cfg.MODEL.POLICY.CLIP_LOG_STD,
            cfg.MODEL.POLICY.MIN_LOG_STD,
            cfg.MODEL.POLICY.MAX_LOG_STD,
            reduction,
        )

        self.inp_seq_len = cfg.BENCH.MAX_NUM_LINKS + 3
        # likely we will be saving all environments together, concatenated together in the batch dimension
        self.num_envs = cfg.TRAIN.NUM_ENVS_PER_TASK * len(cfg.MULTIENV.ROBOTS)

        # self.num_envs = num_envs
        self.num_layers = cfg.MODEL.RNN.NUM_LAYERS
        self.hidden_size = (
            cfg.MODEL.RNN.HIDDEN_SIZE
        )  # Hcell (Hout is Hcell because proj_size = 0)
        self.sequence_length = cfg.MODEL.RNN.SEQUENCE_LENGTH

        self.project_features = nn.Linear(
            (3 + cfg.BENCH.MAX_NUM_LINKS) * cfg.MODEL.RNN.FEATURE_DIM,
            cfg.MODEL.RNN.FEATURE_DIM,
        )

        self.num_observations = cfg.MODEL.RNN.FEATURE_DIM
        

        self.lstm = nn.LSTM(
            input_size=self.num_observations,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )  # batch_first -> (batch, sequence, features)

        self.lstm.flatten_parameters()

        if "cartpole" in cfg.MULTIENV.ROBOTS:
            num_actions = 1
            self.feature_extractor = nn.Linear(cfg.CARTPOLE.OBS_DIM, cfg.MODEL.RNN.FEATURE_DIM)
            self.net = ResidualMLP(
                [cfg.MODEL.RNN.HIDDEN_SIZE]
                + [cfg.MODEL.MLP.EMBED_DIM] * cfg.MODEL.MLP.N_LAYERS
                + [num_actions],
                final_nonlinearity=False,
                nonlinearity="gelu",
                dropout=cfg.MODEL.MLP.DROPOUT,
            )

        else:   
            num_actions = cfg.BENCH.MAX_NUM_LINKS
            assert (
                num_actions == self.num_actions
            ), f"num_actions: {num_actions}, self.num_actions: {self.num_actions}"

        self.feature_extractor = FeatureExtractor(
            cfg.MODEL.RNN.FEATURE_DIM, n_robots=len(cfg.MULTIENV.ROBOTS), attention_layer=(cfg.MODEL.TYPE == "transformer")
        )
        self.feature_extractor.init_weights()

        if cfg.MODEL.TYPE == "transformer":
            self.net = RNNTransformerEncDec(1, only_limbs=True)
            # self.net.init_weights()
        elif cfg.MODEL.TYPE == "mlp-res":
            self.net = ResidualMLP(
                [cfg.MODEL.RNN.HIDDEN_SIZE]
                + [cfg.MODEL.MLP.EMBED_DIM] * cfg.MODEL.MLP.N_LAYERS
                + [self.num_actions],
                final_nonlinearity=False,
                nonlinearity="gelu",
                dropout=cfg.MODEL.MLP.DROPOUT,
            )
            # self.net.init_weights()
        elif cfg.MODEL.TYPE == "mlp":
            self.net = make_mlp_default(
                [cfg.MODEL.RNN.HIDDEN_SIZE]
                + [cfg.MODEL.MLP.EMBED_DIM] * cfg.MODEL.MLP.N_LAYERS
                + [self.num_actions],
                final_nonlinearity=False,
                nonlinearity="gelu",
                dropout=cfg.MODEL.MLP.DROPOUT,
            )
            # init_weights(self.net)
        else:
            raise ValueError(f"Invalid model type: {cfg.MODEL.TYPE}")

        # self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))
        self.log_std_parameter = nn.Parameter(
            torch.ones(self.num_actions) * cfg.MODEL.POLICY.INITIAL_LOG_STD
        )

    def get_specification(self):
        # batch size (N) is the number of envs
        return {
            "rnn": {
                "sequence_length": self.sequence_length,
                "sizes": [
                    (
                        self.num_layers,
                        self.num_envs,
                        self.hidden_size,
                    ),  # hidden states (D ∗ num_layers, N, Hout)
                    (self.num_layers, self.num_envs, self.hidden_size),
                ],
            }
        }  # cell states   (D ∗ num_layers, N, Hcell)

    def compute(self, inputs, role):
        states = inputs["states"]

        terminated = inputs.get("terminated", None)
        hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]

        feature_vector = self.feature_extractor(states)

        # training
        if self.training:

            
            _feature_vector = feature_vector.view(*feature_vector.shape[:-2], -1)
            # project the features to a lower dimension
            _feature_vector = self.project_features(_feature_vector)
            rnn_input = _feature_vector.view(
                -1, self.sequence_length, cfg.MODEL.RNN.FEATURE_DIM
            )
            # rnn_input = states.view(
            #     -1, self.sequence_length, states.shape[-1]
            # )  # (N, L, Hin): N=batch_size, L=sequence_length
            hidden_states = hidden_states.view(
                self.num_layers, -1, self.sequence_length, hidden_states.shape[-1]
            )  # (D * num_layers, N, L, Hout)
            cell_states = cell_states.view(
                self.num_layers, -1, self.sequence_length, cell_states.shape[-1]
            )  # (D * num_layers, N, L, Hcell)
            # get the hidden/cell states corresponding to the initial sequence
            hidden_states = hidden_states[
                :, :, 0, :
            ].contiguous()  # (D * num_layers, N, Hout)
            cell_states = cell_states[
                :, :, 0, :
            ].contiguous()  # (D * num_layers, N, Hcell)

            # reset the RNN state in the middle of a sequence
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = (
                    [0]
                    + (
                        terminated[:, :-1].any(dim=0).nonzero(as_tuple=True)[0] + 1
                    ).tolist()
                    + [self.sequence_length]
                )

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, (hidden_states, cell_states) = self.lstm(
                        rnn_input[:, i0:i1, :], (hidden_states, cell_states)
                    )
                    hidden_states[:, (terminated[:, i1 - 1]), :] = 0
                    cell_states[:, (terminated[:, i1 - 1]), :] = 0
                    rnn_outputs.append(rnn_output)

                rnn_states = (hidden_states, cell_states)
                rnn_output = torch.cat(rnn_outputs, dim=1)
            # no need to reset the RNN state in the sequence
            else:
                rnn_output, rnn_states = self.lstm(
                    rnn_input, (hidden_states, cell_states)
                )
        # rollout
        else:
            
            bs = feature_vector.shape[0]
            
            _feature_vector = feature_vector.view(*feature_vector.shape[:-2], -1)
            # project the features to a lower dimension
            _feature_vector = self.project_features(_feature_vector)
            rnn_input = _feature_vector.view(
                bs, 1, cfg.MODEL.RNN.FEATURE_DIM
            )
            rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

        # flatten the RNN output
        rnn_output = torch.flatten(
            rnn_output, start_dim=0, end_dim=1
        )  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)

        if cfg.MODEL.TYPE in ["mlp", "mlp-res"]:
            # Pendulum-v1 action_space is -2 to 2
            # model_output = torch.tanh(self.net(rnn_output))
            model_output = self.net(rnn_output)

        elif cfg.MODEL.TYPE == "transformer":
            rnn_output = rnn_output.unsqueeze(1).expand(-1, self.inp_seq_len, -1)
            new_features = torch.cat([feature_vector, rnn_output], dim=2)
            obs_mask = states["obs_mask"].bool()[..., 0]
            model_output, attention_vals = self.net(
                new_features, obs_mask, return_attention=False
            )
            # model_output = torch.tanh(model_output)

        else:
            raise ValueError(f"Invalid model type: {cfg.MODEL.TYPE}")

        if self._clip_log_std:
            log_std = torch.clamp(
                self.log_std_parameter, self._log_std_min, self._log_std_max
            )
        else:
            log_std = self.log_std_parameter

        return (
            model_output,
            log_std,
            {
                "rnn": [rnn_states[0], rnn_states[1]],  # "act_mask": states["act_mask"]
            },
        )

    def gaussian_act(self, inputs, return_entropy=False):
        # map from states/observations to mean actions and log standard deviations

        mean_actions, log_std, outputs = self.compute(inputs, "policy")

        # clamp log standard deviations
        if self._clip_log_std:
            log_std = torch.clamp(log_std, self._log_std_min, self._log_std_max)

        self._log_std = log_std
        self._num_samples = mean_actions.shape[0]

        act_mask = inputs["states"]["act_mask"].bool()
        inverted_act_mask = ~act_mask

        # distribution
        self._distribution = Normal(mean_actions, log_std.exp())

        # sample using the reparameterization trick
        actions = self._distribution.rsample()

        # log of the probability density function
        log_prob = self._distribution.log_prob(inputs.get("taken_actions", actions))
        entropy = self._distribution.entropy()

        assert actions.shape[1:] == (
            self.num_actions,
        ), f"Actions shape: {actions.shape}"

        log_prob[inverted_act_mask[..., 0]] = 0.0
        entropy[inverted_act_mask[..., 0]] = 0.0

        # actions is of shape (batch_size, num_limbs)  where num_limbs = num_robots * num_links

        if self._reduction is not None:
            log_prob = self._reduction(log_prob, dim=-1)
        if log_prob.dim() != actions.dim():
            log_prob = log_prob.unsqueeze(-1)

        outputs["mean_actions"] = mean_actions

        actions[inverted_act_mask[..., 0]] = 0.0

        if return_entropy:
            return actions, log_prob, outputs, entropy.mean()
        else:
            return actions, log_prob, outputs

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

        # discrete action space (Discrete)
        # if issubclass(type(self.action_space), gym.spaces.Discrete) or issubclass(type(self.action_space), gymnasium.spaces.Discrete):
        #     return torch.randint(self.action_space.n, (bs, 1), device=self.device), None, {}
        # continuous action space (Box)
        if issubclass(type(self.action_space), gym.spaces.Box) or issubclass(
            type(self.action_space), gym.spaces.Box
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
            act_mask = inputs["states"]["act_mask"].bool()

            # clip actions
            if self._clip_actions:
                actions = torch.clamp(
                    actions,
                    min=cfg.MODEL.POLICY.ACTION_MIN,
                    max=cfg.MODEL.POLICY.ACTION_MAX,
                )

            # log of the probability density function
            log_prob = self._random_distribution.log_prob(
                inputs.get("taken_actions", actions)
            )

            # mask out the log_prob for the actions that are not valid
            assert actions.shape[1:] == (
                self.num_actions,
            ), f"Actions shape: {actions.shape}"

            log_prob[~act_mask[..., 0]] = 0.0

            # actions is of shape (batch_size, num_limbs)  where num_limbs = num_robots * num_links

            if self._reduction is not None:
                log_prob = self._reduction(log_prob, dim=-1)
            if log_prob.dim() != actions.dim():
                log_prob = log_prob.unsqueeze(-1)

            if return_entropy:
                entropy = self._random_distribution.entropy()
                entropy[~act_mask[..., 0]] = 0.0
                return actions, log_prob, {}, entropy.mean()

            hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]

            seq_len = cfg.BENCH.MAX_NUM_LINKS + 3
            feature_vector = self.feature_extractor(inputs["states"])
            bs = feature_vector.shape[0]
            
            _feature_vector = feature_vector.view(bs, seq_len * cfg.MODEL.RNN.FEATURE_DIM)
            _feature_vector = self.project_features(_feature_vector)
            rnn_input = _feature_vector.view(bs, 1, cfg.MODEL.RNN.FEATURE_DIM)
            rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

            return actions, log_prob, {"rnn": [rnn_states[0], rnn_states[1]]}
            # return self._random_distribution.sample(sample_shape=(bs, self.num_actions)), None, {}
        else:
            raise NotImplementedError(
                f"Action space type ({type(self.action_space)}) not supported"
            )

    def act(self, inputs, role, return_entropy=False):
        return self.gaussian_act(inputs, return_entropy)


class CriticRNN(DeterministicMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
    ):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=cfg.MODEL.VALUE.CLIP_ACTIONS)

        # likely we will be saving all environments together, concatenated together in the batch dimension
        self.num_envs = cfg.TRAIN.NUM_ENVS_PER_TASK * len(cfg.MULTIENV.ROBOTS)

        self.inp_seq_len = cfg.BENCH.MAX_NUM_LINKS + 3
        # self.num_envs = num_envs
        self.num_layers = cfg.MODEL.RNN.NUM_LAYERS
        self.hidden_size = (
            cfg.MODEL.RNN.HIDDEN_SIZE
        )  # Hcell (Hout is Hcell because proj_size = 0)
        self.sequence_length = cfg.MODEL.RNN.SEQUENCE_LENGTH

        self.project_features = nn.Linear(
            (3 + cfg.BENCH.MAX_NUM_LINKS) * cfg.MODEL.RNN.FEATURE_DIM,
            cfg.MODEL.RNN.FEATURE_DIM,
        )

        self.num_observations = cfg.MODEL.RNN.FEATURE_DIM
        self.lstm = nn.LSTM(
            input_size=self.num_observations,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )  # batch_first -> (batch, sequence, features)

        self.lstm.flatten_parameters()

        num_actions = cfg.BENCH.MAX_NUM_LINKS

        assert (
            num_actions == self.num_actions
        ), f"num_actions: {num_actions}, self.num_actions: {self.num_actions}"

        self.feature_extractor = FeatureExtractor(
            cfg.MODEL.RNN.FEATURE_DIM, n_robots=len(cfg.MULTIENV.ROBOTS), attention_layer=(cfg.MODEL.TYPE == "transformer")
        )
        self.feature_extractor.init_weights()

        if cfg.MODEL.TYPE == "transformer":
            self.net = RNNTransformerEncDec(1, only_limbs=False)
            # self.net.init_weights()
        elif cfg.MODEL.TYPE == "mlp-res":
            self.net = ResidualMLP(
                [cfg.MODEL.RNN.HIDDEN_SIZE]
                + [cfg.MODEL.MLP.EMBED_DIM] * cfg.MODEL.MLP.N_LAYERS
                + [1],
                final_nonlinearity=False,
                nonlinearity="relu",
                dropout=cfg.MODEL.MLP.DROPOUT,
            )
        elif cfg.MODEL.TYPE == "mlp":
            self.net = make_mlp_default(
                [cfg.MODEL.RNN.HIDDEN_SIZE]
                + [cfg.MODEL.MLP.EMBED_DIM] * cfg.MODEL.MLP.N_LAYERS
                + [1],
                final_nonlinearity=False,
                nonlinearity="relu",
                dropout=cfg.MODEL.MLP.DROPOUT,
            )
            # init_weights(self.net)

        else:
            raise ValueError(f"Invalid model type: {cfg.MODEL.TYPE}")

        # initialize the weights to be very small
        for p in self.net.parameters():
            p.data.fill_(1e-5)


    def deterministic_act(self, inputs):
        # map from observations/states to actions
        values, outputs = self.compute(inputs, "value")

        # clip actions
        if self._clip_actions:
            values = torch.clamp(
                values, min=cfg.MODEL.VALUE.VALUE_MIN, max=cfg.MODEL.VALUE.VALUE_MAX
            )

        return values, None, outputs

    def compute(self, inputs, role):
        states = inputs["states"]
        terminated = inputs.get("terminated", None)
        hidden_states, cell_states = inputs["rnn"][0], inputs["rnn"][1]

        feature_vector = self.feature_extractor(states)
        
        # flatten the link information
        assert feature_vector.shape[-2:] == (self.inp_seq_len, cfg.MODEL.RNN.FEATURE_DIM), f"feature_vector shape: {feature_vector.shape}"
    
        # training
        if self.training:
            _feature_vector = feature_vector.view(*feature_vector.shape[:-2], -1)
            _feature_vector = self.project_features(_feature_vector)
        
            rnn_input = _feature_vector.view(
                -1, self.sequence_length, cfg.MODEL.RNN.FEATURE_DIM
            )
            hidden_states = hidden_states.view(
                self.num_layers, -1, self.sequence_length, hidden_states.shape[-1]
            )  # (D * num_layers, N, L, Hout)
            cell_states = cell_states.view(
                self.num_layers, -1, self.sequence_length, cell_states.shape[-1]
            )  # (D * num_layers, N, L, Hcell)
            # get the hidden/cell states corresponding to the initial sequence
            hidden_states = hidden_states[
                :, :, 0, :
            ].contiguous()  # (D * num_layers, N, Hout)
            cell_states = cell_states[
                :, :, 0, :
            ].contiguous()  # (D * num_layers, N, Hcell)

            # reset the RNN state in the middle of a sequence
            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)
                indexes = (
                    [0]
                    + (
                        terminated[:, :-1].any(dim=0).nonzero(as_tuple=True)[0] + 1
                    ).tolist()
                    + [self.sequence_length]
                )

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i + 1]
                    rnn_output, (hidden_states, cell_states) = self.lstm(
                        rnn_input[:, i0:i1, :], (hidden_states, cell_states)
                    )
                    hidden_states[:, (terminated[:, i1 - 1]), :] = 0
                    cell_states[:, (terminated[:, i1 - 1]), :] = 0
                    rnn_outputs.append(rnn_output)

                rnn_states = (hidden_states, cell_states)
                rnn_output = torch.cat(rnn_outputs, dim=1)
            # no need to reset the RNN state in the sequence
            else:
                rnn_output, rnn_states = self.lstm(
                    rnn_input, (hidden_states, cell_states)
                )
        # rollout
        else:

            _feature_vector = feature_vector.view(*feature_vector.shape[:-2], -1)
            # project the features to a lower dimension
            _feature_vector = self.project_features(_feature_vector)
            rnn_input = _feature_vector.view(-1, 1, _feature_vector.shape[-1])
            
            rnn_output, rnn_states = self.lstm(rnn_input, (hidden_states, cell_states))

        # flatten the RNN output
        rnn_output = torch.flatten(
            rnn_output, start_dim=0, end_dim=1
        )  # (N, L, D ∗ Hout) -> (N * L, D ∗ Hout)

        if cfg.MODEL.TYPE in ["mlp", "mlp-res"]:
            # Pendulum-v1 action_space is -2 to 2
            model_output = self.net(rnn_output)

        elif cfg.MODEL.TYPE == "transformer":
            rnn_output = rnn_output.unsqueeze(1).expand(-1, self.inp_seq_len, -1)
            new_features = torch.cat([feature_vector, rnn_output], dim=2)

            obs_mask = states["obs_mask"].bool()[..., 0]
            vals, attention_map = self.net(
                new_features, obs_mask, return_attention=False
            )

            n_limbs = torch.sum(obs_mask.int(), dim=1, keepdim=True)

            if torch.any(n_limbs == 0):
                raise ValueError("All limbs are masked out")


            vals[~obs_mask] = 0.0
            model_output = torch.sum(vals, dim=1, keepdim=True) / n_limbs

            if torch.isnan(vals).any():
                import pdb

                pdb.set_trace()
        else:
            raise ValueError(f"Invalid model type: {cfg.MODEL.TYPE}")

        return (
            model_output,
            {"rnn": [rnn_states[0], rnn_states[1]]},
        )

    def act(self, inputs, role):
        return self.deterministic_act(inputs)

    def get_specification(self):
        # batch size (N) is the number of envs
        return {
            "rnn": {
                "sequence_length": self.sequence_length,
                "sizes": [
                    (
                        self.num_layers,
                        self.num_envs,
                        self.hidden_size,
                    ),  # hidden states (D ∗ num_layers, N, Hout)
                    (self.num_layers, self.num_envs, self.hidden_size),
                ],
            }
        }  # cell states   (D ∗ num_layers, N, Hcell)
