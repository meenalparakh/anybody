from urdfpy import URDF
import torch
from dataclasses import MISSING
from pathlib import Path
from collections.abc import Sequence
import isaaclab.utils.string as string_utils

from isaaclab.assets import Articulation

from isaaclab.utils import configclass
from isaaclab.envs.mdp.actions.actions_cfg import JointActionCfg

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.envs import ManagerBasedEnv

# from omni.isaac.lab_tasks.manager_based.classic.cartpole.mdp.rewards import joint_pos_target_l2

from anybody.cfg import cfg
import anybody.envs.tasks.env_utils as eu
import anybody.envs.sim.utils as iu
from anybody.utils.utils import save_pickle, load_pickle
from anybody.cfg import cfg as global_cfg
from .funcs import get_action_bins
                

class MyJointPositionAction(ActionTerm):
    r"""Base class for joint actions.

    This action term performs pre-processing of the raw actions using affine transformations (scale and offset).
    These transformations can be configured to be applied to a subset of the articulation's joints.

    Mathematically, the action term is defined as:

    .. math::

       \text{action} = \text{offset} + \text{scaling} \times \text{input action}

    where :math:`\text{action}` is the action that is sent to the articulation's actuated joints, :math:`\text{offset}`
    is the offset applied to the input action, :math:`\text{scaling}` is the scaling applied to the input
    action, and :math:`\text{input action}` is the input action from the user.

    Based on above, this kind of action transformation ensures that the input and output actions are in the same
    units and dimensions. The child classes of this action term can then map the output action to a specific
    desired command of the articulation's joints (e.g. position, velocity, etc.).
    """

    cfg: JointActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _scale: torch.Tensor | float
    """The scaling factor applied to the input action."""
    _offset: torch.Tensor | float
    """The offset applied to the input action."""

    def __init__(self, cfg, env: ManagerBasedEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        self._action_mask = self.cfg.action_mask.to(env.scene.device)
        self._custom_action_dim = self.cfg.custom_action_dim

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names, preserve_order=True
        )
        
        
        self._num_joints = len(self._joint_ids)
        # log the resolved joint names for debugging
        print(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )

        # Avoid indexing across all joints for efficiency
        # if self._num_joints == self._asset.num_joints:
        #     self._joint_ids = slice(None)

        if global_cfg.ACTION.DISCRETE:
            # //////////////////////////////////////////////////////////////////////////
            self.action_bins = get_action_bins().to(env.scene.device)
            print("action_bins", self.action_bins)
            self._raw_actions = torch.zeros(
                self.num_envs, self.action_dim, device=self.device
            )
            # //////////////////////////////////////////////////////////////////////////
            self._processed_actions = torch.zeros(
                self.num_envs, self.action_dim, device=self.device
            )
    
        else:
            # create tensors for raw and processed actions
            self._raw_actions = torch.zeros(
                self.num_envs, self.action_dim, device=self.device
            )
            
            self._processed_actions = torch.zeros_like(self.raw_actions)

        # parse scale
        if isinstance(self.cfg.scale, (float, int)):
            self._scale = float(self.cfg.scale)
        elif isinstance(self.cfg.scale, dict):
            raise NotImplementedError("Dictionary scale is not supported yet.")
            self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(
                self.cfg.scale, self._joint_names
            )
            self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
        elif isinstance(self.cfg.scale, torch.Tensor):
            self._scale = self.cfg.scale.to(self.device)
        else:
            raise ValueError(
                f"Unsupported scale type: {type(self.cfg.scale)}. Supported types are float and dict."
            )
        # parse offset
        if isinstance(self.cfg.offset, (float, int)):
            self._offset = float(self.cfg.offset)
        elif isinstance(self.cfg.offset, dict):
            raise NotImplementedError("Dictionary scale is not supported yet.")
            self._offset = torch.zeros_like(self._raw_actions)
            # resolve the dictionary config
            index_list, _, value_list = string_utils.resolve_matching_names_values(
                self.cfg.offset, self._joint_names
            )
            self._offset[:, index_list] = torch.tensor(value_list, device=self.device)
        elif isinstance(self.cfg.offset, torch.Tensor):
            self._offset = self.cfg.offset.to(self.device)
        else:
            raise ValueError(
                f"Unsupported offset type: {type(self.cfg.offset)}. Supported types are float and dict."
            )

        if self.cfg.use_default_offset:
            assert False, "we are using delta movements so no offset needed"
            
            self._offset = self._asset.data.default_joint_pos[
                :, self._joint_ids
            ].clone()

    @property
    def action_dim(self) -> int:
        return self._custom_action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        # store the raw actions
        self._raw_actions[:] = actions
        # apply the affine transformations
        
        # clamp the actions to -1 to 1
        # self._raw_actions = torch.clamp(self._raw_actions, -1.0, 1.0)
        
        if global_cfg.ACTION.DISCRETE:
            
            # actions contain the indices of the bins - which bin to choose as the final value
            # print("actions", actions[0].detach().cpu().numpy())
            _actions = self.action_bins[self._raw_actions.long()]
                        
            # raw_actions_shape = self._raw_actions.shape
            # new_shape = raw_actions_shape[:-1] + (global_cfg.BENCH.MAX_NUM_LINKS, global_cfg.ACTION.NUM_BINS)
            # raw_actions = self._raw_actions.view(*new_shape)
            
            # actions = (raw_actions @ self.action_bins.unsqueeze(1)).squeeze(-1)
            
        else:
            _actions = self._raw_actions 
        
        
        if global_cfg.ACTION.ABSOLUTE:        
            self._processed_actions = _actions * self._scale + self._offset
        else:
            self._processed_actions = _actions


    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0

    def apply_actions(self):
        # set position targets
        
        actions = self.processed_actions[:, self._action_mask.bool()]
        # print("actions", actions.detach().cpu().numpy())
        
        if global_cfg.ACTION.ABSOLUTE:
            current_actions = actions
        else:        
            # add the delta to current joint positions, then take tanh to map to -1 to 1, followed by proper scaling and offset.
            current_joint_positions = self._asset.data.joint_pos[:, self._joint_ids]
            # print("actions", actions[0] * global_cf)
            current_actions = current_joint_positions + actions * global_cfg.ACTION.SCALE
            
            # clamp the actions to within limits
            joint_lb = self._asset.data.joint_pos_limits[:, self._joint_ids, 0]
            joint_ub = self._asset.data.joint_pos_limits[:, self._joint_ids, 1]
            
            current_actions = torch.clamp(current_actions, joint_lb, joint_ub)
            # print("current pos", current_actions[:4, 0].detach().cpu().numpy())
                        
        self._asset.set_joint_position_target(current_actions, joint_ids=self._joint_ids)
    
        # visualize the goal and current pose for the robot (for reach task), 
        # visualize the goal and current pose for the object and robot (for push task)
        
        
        



@configclass
class MyJointPositionActionCfg(JointActionCfg):
    """Configuration for the joint position action term.

    See :class:`JointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = MyJointPositionAction
    action_mask: torch.Tensor = MISSING
    custom_action_dim: int = MISSING
    true_action_dim: int = MISSING  

    use_default_offset: bool = False
    """Whether to use default joint positions configured in the articulation asset as offset.
    Defaults to True.

    If True, this flag results in overwriting the values of :attr:`offset` to the default joint positions
    from the articulation asset.
    """


@configclass
class ActionsCfg:
    """Action specifications for the environment."""

    def __init__(self, prob: eu.ProblemSpec):

        robo_id = list(prob.robot_dict.keys())[0]
        robo = prob.robot_dict[robo_id]
            
        robo_info_pickle_fname = robo.robot_urdf.replace(".urdf", "_info.pkl")
        
        
        if not Path(robo_info_pickle_fname).exists() or cfg.FORCE_RECOMPUTE_LINK_INFO:
            robo_urdfpy = URDF.load(robo.robot_urdf)
            links_info = iu.get_robo_link_info(robo_urdfpy)
            joints_info, child_dict = iu.get_robo_joint_cfg(
                robo_urdfpy, return_child_dict=True
            )
            save_pickle((links_info, joints_info, child_dict), robo_info_pickle_fname)
        else:
            links_info, joints_info, child_dict = load_pickle(robo_info_pickle_fname)

        link_idx_dict = iu.get_robo_link_vec(
            links_info=links_info,
            joints_info=joints_info,
            child_dict=child_dict,
            goal_state=prob.goal_robo_state_dict.get(robo_id, None),
            ee_link_name=robo.ee_link,
            link_idx_encode_dim=cfg.OBSERVATION.ROBO_LINK_IDX_ENCODE_DIM,
        )

        robo_act_mask = torch.zeros(cfg.BENCH.MAX_NUM_LINKS)
        robo_scale = torch.ones_like(robo_act_mask)        
        robo_offset = torch.zeros_like(robo_act_mask)

        robo_indices = list(prob.robot_dict.keys())
        robo_indices.sort()

        act_jnames = []
        robo_id = robo_indices[0]
        info = link_idx_dict

        link_indices = list(info.keys())
        link_indices.sort()

        robo_joint_names = prob.robot_dict[robo_id].act_info["joint_names"]
        robo_lb = prob.robot_dict[robo_id].act_info["joint_lb"]
        robo_ub = prob.robot_dict[robo_id].act_info["joint_ub"]

        for il, link_idx in enumerate(link_indices):
            _, _, _, _, jname = info[link_idx]
            if jname in robo_joint_names:
                act_jnames.append(jname)
                robo_act_mask[il] = True
                lb = robo_lb[robo_joint_names.index(jname)]
                ub = robo_ub[robo_joint_names.index(jname)]
                robo_scale[il] = (ub - lb) / 2.0
                robo_offset[il] = (ub + lb) / 2.0

        custom_action_dim = cfg.BENCH.MAX_NUM_LINKS
        # if cfg.ACTION.DISCRETE:
            # custom_action_dim = custom_action_dim * cfg.ACTION.NUM_BINS

        self.__setattr__(
            "robo_jpos",
            MyJointPositionActionCfg(
                asset_name=f"robot_{robo_id}",
                joint_names=act_jnames,
                # scale=new_scale * cfg.ACTION.SCALE,          # 0.1 limits the delta to 10% of the joint limits
                scale=robo_scale,
                offset=robo_offset,
                use_default_offset=False,
                action_mask=robo_act_mask,
                custom_action_dim=custom_action_dim,
                true_action_dim=cfg.BENCH.MAX_NUM_LINKS,
            ),
        )

