import math

import torch
import torch.nn as nn
from anybody.cfg import cfg

from anybody.utils.voxel_utils import generate_batched_cuboid_point_clouds
from torch.nn import functional as F
from .transformer import TransformerEncoder
from .transformer import TransformerEncoderLayerResidual
from isaaclab.utils.math import matrix_from_quat

import torch.nn.init as nn_init

def w_init(module, gain=1):
    nn.init.orthogonal_(module.weight.data, gain=gain)
    nn.init.constant_(module.bias.data, 0)
    return module


def make_mlp(dim_list):

    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(w_init(nn.Linear(dim_in, dim_out)))
        layers.append(nn.Tanh())

    return nn.Sequential(*layers)


def make_mlp_default(dim_list, final_nonlinearity=True, nonlinearity="relu", dropout=0.0):
    layers = []
    for idx, (dim_in, dim_out) in enumerate(zip(dim_list[:-1], dim_list[1:])):
        layers.append(nn.Linear(dim_in, dim_out))
        if nonlinearity == "relu":
            layers.append(nn.ReLU())
        elif nonlinearity == "tanh":
            layers.append(nn.Tanh())
        elif nonlinearity == "gelu":
            layers.append(nn.GELU())
        if dropout > 0:
            if idx != len(dim_list) - 2:
                layers.append(nn.Dropout(dropout))

    if not final_nonlinearity:
        layers.pop()
        
    return nn.Sequential(*layers)


class ActionEncoder(nn.Module):
    def __init__(self, feature_dim, n_robots: int):
        # no need to calculate the obs size, just use the cfg values
        super().__init__()
        self.d_model = feature_dim
        self.act_embed = nn.Linear(1, self.d_model)
        
        # initialize the weights
        self.act_embed = w_init(self.act_embed)

    def forward(self, action):
        # action has shape (batch_size, act_dim)
        bs, act_dim = action.shape
        action_embed = self.act_embed(action.unsqueeze(-1))
        assert action_embed.shape == (bs, act_dim, self.d_model), f"action_embed shape: {action_embed.shape}"
        
        return action_embed


class PointCloudEncoder(nn.Module):
    def __init__(self, input_dim=3, feature_dim=128):
        super(PointCloudEncoder, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 32, 1)
        self.conv2 = nn.Conv1d(32, 32, 1)
        self.conv3 = nn.Conv1d(32, feature_dim, 1)
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(feature_dim)

    def forward(self, x):
        """
        Args:
            x: (batch_size, 1, box_dim + box_pose)
            x: Input point cloud, shape (batch_size, num_points, input_dim).
        Returns:
            Global feature vector, shape (batch_size, feature_dim).
        """
        # with no gradients
        with torch.no_grad():    
            obstacle_pose = x[:, 0, :7]
            obstacle_shape = x[:, 0, 7 + 3:]   # the first 3 are the object type
            pcd = generate_batched_cuboid_point_clouds(
                obstacle_shape, obstacle_pose, matrix_from_quat,
                num_points=cfg.OBSERVATION.PCD_POINTS, 
                noise_std=cfg.OBSERVATION.PCD_NOISE,
                device=x.device
            )
        
        x = pcd.permute(0, 2, 1)  # (batch_size, input_dim, num_points)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x)) # (batch_size, feature_dim, num_points)
        x = torch.max(x, 2, keepdim=True)[0] # (batch_size, feature_dim, 1)
        x = x.permute(0, 2, 1) # (batch_size, 1, feature_dim)
        return x
    
class ObstacleFeatureExtractor(nn.Module):
    def __init__(self, feature_dim, n_robots):
        super().__init__()
        
        self.d_model = feature_dim

        if cfg.OBSERVATION.HIGH_DIM:
            # need code that first converts the ld features to hd 
            # we have ld to save space in buffer
            self.obstacle_embed = PointCloudEncoder(input_dim=3, feature_dim=self.d_model)
            
        else:
            # a simple linear encoder to embed the obstacle features
            self.obstacle_embed = nn.Linear(cfg.OBSERVATION.OBJ_DIM, self.d_model)
            
    def forward(self, obstacle_obs):
        # obstacle_obs has shape (batch_size, 1, num_features)
        bs = obstacle_obs.shape[0]
        
        obs_embed = self.obstacle_embed(obstacle_obs)
        assert obs_embed.shape == (bs, 1, self.d_model), f"obs_embed shape: {obs_embed.shape}"
        return obs_embed
            


# J: Max num joints between two limbs. 1 for 2D envs, 2 for unimal
class ObsFeatureExtractor(nn.Module):
    def __init__(self, feature_dim: int, n_robots, attention_layer=False):
        super().__init__()

        # no need to calculate the obs size, just use the cfg values
        link_obs_size = cfg.OBSERVATION.ROBO_LINK_DIM
        robo_base_obs_size = cfg.OBSERVATION.ROBO_BASE_DIM
        robo_goal_size = cfg.OBSERVATION.ROBO_GOAL_DIM
        obj_obs_size = cfg.OBSERVATION.OBJ_DIM
        
        self.d_model = feature_dim
        self.link_embed = nn.Linear(link_obs_size, self.d_model)
        self.obj_embed = nn.Linear(obj_obs_size, self.d_model)
        self.robo_base_embed = nn.Linear(robo_base_obs_size, self.d_model)
        self.robo_goal_embed = nn.Linear(robo_goal_size, self.d_model)
        self.obstacle_embed = ObstacleFeatureExtractor(self.d_model, n_robots=n_robots)

        self.link_pos_embed = nn.Parameter(torch.randn(1, 1, self.d_model)) 
        self.robo_base_pos_embed = nn.Parameter(torch.randn(1, 1, self.d_model))
        self.robo_goal_pos_embed = nn.Parameter(torch.randn(1, 1, self.d_model))
        self.obj_pos_embed = nn.Parameter(torch.randn(1, 1, self.d_model))
        self.obstacle_pos_embed = nn.Parameter(torch.randn(1, 1, self.d_model))

        inp_embed_dim = self.d_model

        if cfg.MODEL.IMPLICIT_OBS:            
            self.implicit_link_features = nn.Parameter(
                torch.randn(n_robots, cfg.BENCH.MAX_NUM_LINKS, inp_embed_dim)
            )

            self.project_link_features = nn.Linear(2 * inp_embed_dim, inp_embed_dim)
            
            
        self.attention_layer = attention_layer
        if self.attention_layer:
            # apply attention to the obtained embedded tokens
            self.attn_layer = TransformerEncoderLayerResidual(
                # cfg.MODEL.LIMB_EMBED_SIZE,
                self.d_model,
                cfg.MODEL.TRANSFORMER.NHEAD,
                cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD,
                cfg.MODEL.TRANSFORMER.DROPOUT,
            )        
        else:
            self.attn_layer = None
            
        self.init_weights()

    def init_weights(self):
        
        # orthogonal initialization for the weights
        # constant initialization for the biases
        gain = nn_init.calculate_gain('relu')
        nn_init.orthogonal_(self.link_embed.weight.data, gain=gain)
        nn_init.constant_(self.link_embed.bias.data, 0)
        
        nn_init.orthogonal_(self.robo_goal_embed.weight.data, gain=gain)
        nn_init.constant_(self.robo_goal_embed.bias.data, 0)
        
        nn_init.orthogonal_(self.obj_embed.weight.data, gain=gain)
        nn_init.constant_(self.obj_embed.bias.data, 0)
        
        nn_init.orthogonal_(self.robo_base_embed.weight.data, gain=gain)
        nn_init.constant_(self.robo_base_embed.bias.data, 0)
        
        # if self.attention_layer:
        #     # Initialize transformer encoder layers
        #     for param in self.attn_layer.parameters():
        #         if param.dim() > 1:
        #             nn_init.orthogonal_(param.data)
        #         else:
        #             nn_init.constant_(param.data, 0)
        
        
        if cfg.MODEL.IMPLICIT_OBS:
            nn_init.orthogonal_(self.project_link_features.weight.data, gain=gain)
            nn_init.constant_(self.project_link_features.bias.data, 0)


    def forward(self, x, return_attention=False):
        # x is a dictionary         
        # "robo_base": robo_base_obs,
        # "robo_link": link_obs,
        # "obj": obj_obs,
        # "obs_mask": observation_mask,
        # "act_mask": act_mask.bool(),
        
        robo_base = x["robo_base"]
        
        bs = robo_base.shape[0]
        # link has shape (batch_size, num_links, num_features)
        link = x["robo_link"]
        
        if cfg.OBSERVATION.PREV_ACTION:
            link = link[..., :-2]    # the last two dims are for action vals and dones
        
        # robo_goal has shape (batch_size, 1, num_features)
        goal = x["robo_goal"]
        
        # obj has shape (batch_size, 1, num_features)
        obj = x["obj"]
        
        # obstacle has shape (batch_size, 1, num_features)
        obstacle = x["obstacle"]
        
        # obs_mask has shape (batch_size, num_links + 3, 1)
        obs_mask = x["obs_mask"]
        
        # flatten the inputs
        robo_base_embed = self.robo_base_embed(robo_base)
        link_embed = self.link_embed(link)
        
        if cfg.MODEL.IMPLICIT_OBS:
            robo_indices = x['robot_id'][:, 0, 0].long()
            # according to robo_indices, select the implicit link features
            # assert robo_indices.shape == (bs,), f"robo_indices shape: {robo_indices.shape}"
            implicit_features = self.implicit_link_features[robo_indices, ...]
            link_embed = torch.cat([link_embed, implicit_features], dim=2)
            link_embed = self.project_link_features(link_embed)
        
        
        # add positional embeddings
        link_embed = link_embed + self.link_pos_embed
        robo_base_embed = robo_base_embed + self.robo_base_pos_embed
        goal_embed = self.robo_goal_embed(goal) + self.robo_goal_pos_embed
        obj_embed = self.obj_embed(obj) + self.obj_pos_embed
        obstacle_embed = self.obstacle_embed(obstacle) + self.obstacle_pos_embed
        
        obs_embed = torch.cat([robo_base_embed, link_embed, goal_embed, obj_embed, obstacle_embed], dim=1)
        assert obs_embed.shape == (bs, cfg.BENCH.MAX_NUM_LINKS + cfg.OBSERVATION.ADDITIONAL_DIM, self.d_model), f"obs_embed shape: {obs_embed.shape}"
        assert obs_mask.shape == (bs, cfg.BENCH.MAX_NUM_LINKS + cfg.OBSERVATION.ADDITIONAL_DIM, 1), f"obs_mask shape: {obs_mask.shape}"
     
        result = obs_embed * obs_mask
        
        if self.attn_layer:
            _obs_mask = obs_mask[..., 0].bool()
            # permute the dimensions to (num_limbs, batch_size, d_model)
            obs_embed = obs_embed.permute(1, 0, 2)
            obs_embed_t = self.attn_layer(
                obs_embed, src_key_padding_mask=(~_obs_mask)
            )
            result = obs_embed_t.permute(1, 0, 2)
        
        
        return result
    
    
class CombinedFeatureExtractor(nn.Module):
    def __init__(self, feature_dim, n_robots):
        
        super().__init__()
        
        self.d_model = feature_dim
        self.feature_extractor = ObsFeatureExtractor(feature_dim, n_robots)
        self.action_encoder = ActionEncoder(feature_dim, n_robots)
        
        self.project_features = nn.Linear(2 * feature_dim, feature_dim)
        
        # self.feature_extractor.init_weights()
        # self.action_encoder.init_weights()
        
    def forward(self, observations, actions):
        obs_embed = self.feature_extractor(observations)
        act_embed = self.action_encoder(actions)
        
        bs, n_elems, d_model = obs_embed.shape
        assert act_embed.shape == (bs, n_elems - 3, d_model), f"act_embed shape: {act_embed.shape}"
        
        obs_link_features = obs_embed[:, 1:-2, :]
        combined_features = torch.cat([obs_link_features, act_embed], dim=2)
        combined_features = self.project_features(combined_features)
        
        obs_robo_features = obs_embed[:, :1, :]
        obs_goal_obj_features = obs_embed[:, -2:, :]
        
        combined_features = torch.cat([obs_robo_features, combined_features, obs_goal_obj_features], dim=1)
        return combined_features        
    
    
class ResidualBlock(nn.Module):
  
    def __init__(self, in_dim, out_dim):
        super(ResidualBlock, self).__init__()

        self.net = make_mlp_default([in_dim, out_dim, out_dim], final_nonlinearity=False, nonlinearity='relu')
        self.residual = nn.Linear(in_dim, out_dim)
        
        self.init_weights()
        
    def init_weights(self):
        
        gain = nn_init.calculate_gain('relu')
        
        nn_init.orthogonal_(self.residual.weight.data, gain=gain)
        nn_init.constant_(self.residual.bias.data, 0)
        
        # Initialize decoder weights and biases
        for layer in self.net:
            if hasattr(layer, 'weight') and layer.weight is not None:
                nn_init.orthogonal_(layer.weight.data, gain=gain)
            if hasattr(layer, 'bias') and layer.bias is not None:
                nn_init.constant_(layer.bias.data, 0)
                
    
    def forward(self, x):
        return self.net(x) + self.residual(x)
        

class ResidualMLP(nn.Module):
    def __init__(self, dim_list, final_nonlinearity=True, nonlinearity="gelu", dropout=0.0):
        super().__init__()
        layers = []
        for idx, (dim_in, dim_out) in enumerate(zip(dim_list[:-1], dim_list[1:])):
            layers.append(ResidualBlock(dim_in, dim_out))
            
            if (idx == (len(dim_list) - 2)) and not final_nonlinearity:
                break
            
            if nonlinearity == "relu":
                layers.append(nn.ReLU())
            elif nonlinearity == "tanh":
                layers.append(nn.Tanh())
            if dropout > 0:
                if idx != len(dim_list) - 2:
                    layers.append(nn.Dropout(dropout))
        
        # if not final_nonlinearity:
        #     layers.pop()
            
        self.net = nn.Sequential(*layers)
        
    def init_weights(self):
        for layer in self.net:
            if hasattr(layer, 'init_weights'):
                layer.init_weights()
            if hasattr(layer, 'weight') and layer.weight is not None:
                nn_init.orthogonal_(layer.weight.data)
            if hasattr(layer, 'bias') and layer.bias is not None:
                nn_init.constant_(layer.bias.data, 0)
                
        
    def forward(self, x):
        return self.net(x)


class MLPBase(nn.Module):

    def __init__(self, include_std=False, n_robots=1, output_tanh=False, combined=False, mlp_kwargs={}):
        super().__init__()
    
        inp_embed_dim = cfg.MODEL.LIMB_EMBED_SIZE
        hidden_layer_dim = cfg.MODEL.MLP.EMBED_DIM
        
        if cfg.MODEL.IMPLICIT_OBS:            
            self.implicit_link_features = nn.Parameter(
                torch.randn(n_robots, cfg.BENCH.MAX_NUM_LINKS, inp_embed_dim)
            )

            self.project_link_features = nn.Linear(2 * inp_embed_dim, inp_embed_dim)
            
        input_dim = (cfg.OBSERVATION.ADDITIONAL_DIM + cfg.BENCH.MAX_NUM_LINKS) * inp_embed_dim   
        output_dim = cfg.BENCH.MAX_NUM_LINKS
        
        if include_std:
            output_dim *= 2
        
        self.initialize_network(
            input_dim=input_dim,
            hidden_layer_dim=hidden_layer_dim,
            n_layers=cfg.MODEL.MLP.N_LAYERS,
            output_dim=output_dim,
            mlp_kwargs=mlp_kwargs
        )

        self.combined = combined
        if combined:
            self.feature_extractor = CombinedFeatureExtractor(inp_embed_dim, n_robots)
        else:
            self.feature_extractor = ObsFeatureExtractor(inp_embed_dim, n_robots)        
        self.mask_embed = nn.Linear(1, inp_embed_dim)
        self.output_tanh = output_tanh  
        
        self.embed_dim = inp_embed_dim
        self.init_weights() 
        
        
    def initialize_network(self, input_dim, hidden_layer_dim, n_layers, output_dim, mlp_kwargs):
        raise NotImplementedError
    
                
    def init_weights(self):
        # orthogonal initialization for the weights
        # constant initialization for the biases
        return
        
        gain = nn_init.calculate_gain('relu')

        nn_init.orthogonal_(self.mask_embed.weight.data, gain=gain)
        nn_init.constant_(self.mask_embed.bias.data, 0)
        
        if cfg.MODEL.IMPLICIT_OBS:
            nn_init.orthogonal_(self.project_link_features.weight.data, gain=gain)
            nn_init.constant_(self.project_link_features.bias.data, 0)
        
    def forward(self, observation, action=None):
        # x is a dictionary         
        # "robo_base": robo_base_obs,
        # "robo_link": link_obs,
        # "obj": obj_obs,
        # "obs_mask": observation_mask,
        # "act_mask": act_mask.bool(),

        # act_mask = x["act_mask"]    
        
        if self.combined:
            assert action is not None, "Action is required for combined model"
            obs_embed = self.feature_extractor(observation, action)
        else:
            assert action is None, "Action is not required for specific model"
            obs_embed = self.feature_extractor(observation)
        
        bs = observation["robo_base"].shape[0]
        obs_embed = obs_embed.reshape(bs, -1)
        output = self.net(obs_embed)
        
        if self.output_tanh:
            output = torch.tanh(output)
        else:
            output = output
        
        return output, None


class MLP(MLPBase):
    def __init__(self, include_std=False, n_robots=1, output_tanh=False, combined=False, mlp_kwargs={"final_nonlinearity": False, "nonlinearity": "gelu"}):
        super().__init__(include_std=include_std, n_robots=n_robots, output_tanh=output_tanh, combined=combined, mlp_kwargs=mlp_kwargs)
        
    def initialize_network(self, input_dim, hidden_layer_dim, n_layers, output_dim, mlp_kwargs):
        self.net = make_mlp_default([input_dim] + [hidden_layer_dim] * n_layers + [output_dim], 
                                dropout=cfg.MODEL.MLP.DROPOUT, **mlp_kwargs)
        
    def init_weights(self):
        super().init_weights()
        
        return
        
        gain = nn_init.calculate_gain('relu')
        for layer in self.net:
            if hasattr(layer, 'weight') and layer.weight is not None:
                nn_init.orthogonal_(layer.weight.data, gain=gain)
            if hasattr(layer, 'bias') and layer.bias is not None:
                nn_init.constant_(layer.bias.data, 0)
                
        # initialize the last layer of the decoder to be normal with scale 0.01
        nn_init.normal_(self.net[-1].weight.data, mean=0, std=0.01)
        
        
class MLP_Residual(MLPBase):
    def __init__(self, include_std=False, n_robots=1, output_tanh=False, combined=False, mlp_kwargs={"final_nonlinearity": False, "nonlinearity": "gelu"}):
        super().__init__(include_std, n_robots, output_tanh, combined, mlp_kwargs)
        
    def initialize_network(self, input_dim, hidden_layer_dim, n_layers, output_dim, mlp_kwargs):
        
        self.net = ResidualMLP(
            [input_dim] + [hidden_layer_dim] * n_layers + [output_dim],
            dropout=cfg.MODEL.MLP.DROPOUT,
            **mlp_kwargs
        )
        
    def init_weights(self):
        super().init_weights()
        
        return
        
        self.net.init_weights()
                
                
class MLP_LatentResidual(MLPBase):
    def __init__(self, include_std=False, n_robots=1, output_tanh=False, combined=False, mlp_kwargs={"final_nonlinearity": False, "nonlinearity": "gelu"}):
        super().__init__(include_std, n_robots, output_tanh, combined, mlp_kwargs)
        
    def initialize_network(self, input_dim, hidden_layer_dim, n_layers, output_dim, mlp_kwargs):
        
        if cfg.MODEL.MLP_LATENT == "l":
            enc_dims = [512, 128, 32, 8, 2]
        else:
            enc_dims = [128, 64, 16, 4]
        self.net = LatentResidualMLP(
            input_dim, output_dim, enc_dims=enc_dims
        )
        
    def init_weights(self):
        super().init_weights()
        
        return
        self.net.init_weights()
                
        
class LatentResidualMLP(nn.Module):
    def __init__(self, input_dim, output_dim, enc_dims):
        super().__init__()
        
        # dims = [input_dim, 1024, 256, 64, 16, 64, 256, 1024, output_dim]
        # self.layers = nn.ModuleList()
        self.layers = nn.ModuleDict()
        activation = nn.LeakyReLU()
        self.activation = activation
        
        self.enc_dims = enc_dims
        
        self.layers["input"] = nn.Linear(input_dim, enc_dims[0])
        
        for idx, (dim_in, dim_out) in enumerate(zip(enc_dims[:-1], enc_dims[1:])):
            self.layers[f"enc_layer_{idx}"] = nn.Sequential(
                nn.LeakyReLU(),
                nn.Linear(dim_in, dim_out * 2),
                nn.BatchNorm1d(dim_out * 2),
                activation,
                nn.Linear(dim_out * 2, dim_out),
                nn.BatchNorm1d(dim_out),                
            )
        
        for idx, (dim_in, dim_out) in enumerate(zip(enc_dims[::-1][:-1], enc_dims[::-1][1:])):
            self.layers[f"dec_layer_{idx}"] = nn.Sequential(
                nn.LeakyReLU(),
                nn.Linear(dim_in, dim_in * 2),
                activation,
                nn.Linear(dim_in * 2, dim_out),
                nn.BatchNorm1d(dim_out),                
            )
            
        self.layers["output"] = nn.Linear(enc_dims[0], output_dim)
        
    def init_weights(self):
        pass
    
        # for _, layer in self.layers.items():
        #     import pdb; pdb.set_trace()
        #     if hasattr(layer, 'weight') and layer.weight is not None:
        #         nn_init.orthogonal_(layer.weight.data)
        #     if hasattr(layer, 'bias') and layer.bias is not None:
        #         nn_init.constant_(layer.bias.data, 0)
                
        
    def forward(self, x):
        
        inp_features = self.layers["input"](x)
        
        book_keep = [inp_features]
        
        for idx in range(len(self.enc_dims) - 1):
            features = self.layers[f"enc_layer_{idx}"](inp_features)
            book_keep.append(features)
            inp_features = features

        

        # decoder
        book_keep = book_keep[::-1] 
        for idx in range(len(self.enc_dims) - 1):
            features = self.layers[f"dec_layer_{idx}"](inp_features)
            inp_features = features + book_keep[idx + 1]
            inp_features = self.activation(inp_features)
            
        # output
        result = self.layers['output'](inp_features)    
        return result
        
                
class ResidualMLP_Cartpole(ResidualMLP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_weights()
        
    def forward(self, inp):
        # inp is a dictionary
        # "robot"
        # "obs_mask"
        # "act_mask"
        
        robo = inp["robot"]
        return self.net(robo), None

class ResidualMLP_Cartpole_Combined(ResidualMLP_Cartpole):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.act_enc = nn.Linear(1, 4)
        self.act_enc = w_init(self.act_enc)
    
    def forward(self, obs, act):
        bs, obs_dim = obs['robot'].shape
        assert act.shape == (bs, 1), f"act shape: {act.shape} vs. expected: {(bs, 1)}"
        act_embed = self.act_enc(act)
        obs_embed = torch.cat([obs['robot'], act_embed], dim=1)
        assert obs_embed.shape == (bs, obs_dim + 4), f"obs_embed shape: {obs_embed.shape}"
        return self.net(obs_embed), None


# J: Max num joints between two limbs. 1 for 2D envs, 2 for unimal
class TransformerModel(nn.Module):
    def __init__(self, decoder_out_dim: int, only_limbs=False, n_robots=1, output_tanh=False, combined=False, 
                 decoder_kwargs={"final_nonlinearity": False, "nonlinearity": "gelu"}):
        super().__init__()
        self.only_limbs = only_limbs        # for policy learning, we only care about the limbs
                                            # for value learning, we care about all the limbs and the base and the goal
        self.model_args = cfg.MODEL.TRANSFORMER

        self.seq_len = cfg.OBSERVATION.ADDITIONAL_DIM + cfg.BENCH.MAX_NUM_LINKS
        self.d_model = cfg.MODEL.LIMB_EMBED_SIZE

        if combined:
            self.feature_extractor = CombinedFeatureExtractor(self.d_model, n_robots)
        else:
            self.feature_extractor = ObsFeatureExtractor(self.d_model, n_robots)
            
        self.ext_feat_fusion = self.model_args.EXT_MIX


        # Transformer Encoder
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
        
        decoder_input_dim = self.d_model
        self.decoder = make_mlp_default(
            [decoder_input_dim] + [self.model_args.DECODER_DIM] * self.model_args.N_DECODER_LAYERS + [decoder_out_dim],
            dropout=self.model_args.DROPOUT,
            **decoder_kwargs
        )
        
        self.output_tanh = output_tanh  
        self.combined = combined
        
        self.init_weights()

    def init_weights(self):
        
        initrange = self.model_args.EMBED_INIT

        self.feature_extractor.init_weights()
        
        # Initialize decoder weights and biases
        for layer in self.decoder:
            if hasattr(layer, 'weight') and layer.weight is not None:
                nn_init.orthogonal_(layer.weight.data)
            if hasattr(layer, 'bias') and layer.bias is not None:
                nn_init.constant_(layer.bias.data, 0)
        
        # Initialize transformer encoder layers
        # for layer in self.transformer_encoder.layers:
        #     for param in layer.parameters():
        #         if param.dim() > 1:
        #             nn_init.orthogonal_(param.data)
        #         else:
        #             nn_init.constant_(param.data, 0)

        # initialize the last layer of the decoder to be normal with scale 0.01
        nn_init.normal_(self.decoder[-1].weight.data, mean=0, std=1.0)
        nn_init.constant_(self.decoder[-1].bias.data, 0)


    def forward(self, state, action=None, return_attention=False):
        # state is a dict of the form
        # "robo_base": (bs, 1, robo_base_size)
        # "robo_link": (bs, num_links, robo_link_size)
        # "robo_goal": (bs, 1, robo_goal_size)
        # "obj": (bs, 1, obj_size)
        # "obs_mask": (bs, num_links + 3, 1)
        # "act_mask": (bs, num_links + 3, 1)

        if self.combined:
            assert action is not None, "Action is required for combined model"
            obs_embed = self.feature_extractor(state, action)
        else:
            assert action is None, "Action is not required for specific model"
            obs_embed = self.feature_extractor(state)


        # convert all observations to corresponding embeddings
        batch_size = state["robo_base"].shape[0]
        all_obs_embed = obs_embed
        
        assert all_obs_embed.shape == (
            batch_size, 
            self.seq_len,
            self.d_model
        ), f"all_obs_embed shape: {all_obs_embed.shape}"
        
        
        # all_obs_embed = self.pos_embedding(all_obs_embed)
        obs_mask = state["obs_mask"][..., 0].bool()

        all_obs_embed = all_obs_embed.permute(1, 0, 2)

        if return_attention:
            obs_embed_t, attention_maps = self.transformer_encoder.get_attention_maps(
                all_obs_embed, src_key_padding_mask=(~obs_mask)
            )
        else:
            # (num_limbs, batch_size, d_model)
            obs_embed_t = self.transformer_encoder(
                all_obs_embed, src_key_padding_mask=(~obs_mask)
            )
            attention_maps = None

        decoder_input = obs_embed_t
        # (num_limbs, batch_size, J)
        output = self.decoder(decoder_input)        
        # select only the limbs
        if self.only_limbs:
            output = output[1:-3, ...]     # first is the base, and the last two are goal and obj, ignore them for actions
        # (batch_size, num_limbs, J)
        output = output.permute(1, 0, 2)
        # (batch_size, num_limbs * J)
        output = output.reshape(batch_size, -1)
        if self.output_tanh:
            output = torch.tanh(output)
        
        return output, attention_maps

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout=0.1, batch_first=False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        if batch_first:
            self.pe = nn.Parameter(torch.randn(1, seq_len, d_model))
        else:
            self.pe = nn.Parameter(torch.randn(seq_len, 1, d_model))

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe
        return self.dropout(x)


class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, seq_len, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(seq_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe
        return self.dropout(x)

