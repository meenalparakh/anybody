from urdfpy import URDF
import torch
from pathlib import Path

from isaaclab.utils import configclass

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg

import anybody.envs.tasks.env_utils as eu
import anybody.envs.sim.utils as iu
from anybody.utils.utils import load_pickle, save_pickle
from anybody.envs.sim.mdp import funcs as ifunc

from anybody.cfg import cfg

# robot's observations include

# link idx (4D)
# parent link idx (4D)
# link geometry (6D)
# link origin (7D)

# joint type (3D) - revolute, prismatic, fixed
# joint limits (2D)
# joint axis (3D)
# joint origin (7D)
# goal joint flag (1D)
# goal joint value (1D)
# goal pose flag (1D)
# goal joint pose (7D) - this pose is relative pose of link wrt parent link

# joint value (1D)


@configclass
class FullStateObsCfg(ObsGroup):
    """Observations for policy group."""

    # observation terms (order preserved)

    def __init__(self, prob: eu.ProblemSpec, robo_task: str, task_idx: int):
        """Important: Variable overwrites must not happen - otherwise
        during observation computation the overwritten values will be used
        instead of the original ones. for example functions referencing variables in the
        init function.
        use params to pass the values to the functions, instead of relying on
        context variables that may change
        """

        super().__init__()
        self.enable_corruption = False
        self.concatenate_terms = False


        # the additional 4 are robo_base, robo_goal, obj, obs pose    
        obs_mask = torch.zeros(cfg.BENCH.MAX_NUM_LINKS + 4, 1)
        
        assert cfg.BENCH.MAX_NUM_ROBOTS == 1, "Only one robot is supported for now"
    
        robo_id = list(prob.robot_dict.keys())[0]
        robo = prob.robot_dict[robo_id]

        # get robo_info pickled file
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

        link_vec_dict = iu.get_robo_link_vec(
            links_info=links_info,
            joints_info=joints_info,
            child_dict=child_dict,
            goal_state=prob.goal_robo_state_dict.get(robo_id, None),
            ee_link_name=robo.ee_link,
            link_idx_encode_dim=cfg.OBSERVATION.ROBO_LINK_IDX_ENCODE_DIM,
        )

        # arrange the links according to their link index numbers
        to_iterate_indices = list(link_vec_dict.keys())
        to_iterate_indices.sort()

        link_vecs = []
        joint_vecs = []
        ee_flag = torch.zeros(len(to_iterate_indices), cfg.OBSERVATION.EE_FLAG_DIM)
        lnames = []
        movable_joints = torch.zeros(len(to_iterate_indices))
        movable_jnames = []

        obs_mask[0, 0] = 1.0   # robo base
        # iterate through indices to fill the link and robo base vectors
        
        jval_lb = []
        jval_ub = []
        for idx, link_idx in enumerate(to_iterate_indices):
            link_info = link_vec_dict[link_idx]
            lvec, jvec, gvec, lname, jname = link_info
            _l_idx, _p_idx = lvec[:2]
            _l_idx = int(_l_idx)
            _p_idx = int(_p_idx)
            assert _l_idx == link_idx, f"Link index mismatch {_l_idx} != {link_idx}"
            
            if _p_idx == 0:
                p_idx = -1
            else:
                # get parent idx's position from 
                p_idx = to_iterate_indices.index(_p_idx)
            lvec[:2] = torch.tensor([p_idx - idx, p_idx-idx])
            link_vecs.append(lvec)
            joint_vecs.append(jvec)
            lnames.append(lname)
            if jname in robo.act_info["joint_names"]:
                movable_joints[idx] = 1.0
                movable_jnames.append(jname)
                robo_j_idx = robo.act_info["joint_names"].index(jname)
                jval_lb.append(robo.act_info["joint_lb"][robo_j_idx])
                jval_ub.append(robo.act_info["joint_ub"][robo_j_idx])
                # act_mask[idx + 1, 0] = 1.0

            if robo.ee_link == lname:
                ee_flag[idx, :] = 1.0

            obs_mask[idx + 1, 0] = 1.0

        #

        lvecs = torch.stack(link_vecs)
        jvecs = torch.stack(joint_vecs)
        jval_lb = torch.tensor(jval_lb)
        jval_ub = torch.tensor(jval_ub)


        remaining_lvecs = torch.zeros(
            cfg.BENCH.MAX_NUM_LINKS - lvecs.shape[0], lvecs.shape[1]
        )
        remaining_jvecs = torch.zeros(
            cfg.BENCH.MAX_NUM_LINKS - jvecs.shape[0], jvecs.shape[1]
        )
        remaining_ee_link = torch.zeros(cfg.BENCH.MAX_NUM_LINKS - lvecs.shape[0], ee_flag.shape[1])
        remaining_movable_joints = torch.zeros(cfg.BENCH.MAX_NUM_LINKS - lvecs.shape[0])

        lvecs = torch.cat([lvecs, remaining_lvecs], dim=0)
        jvecs = torch.cat([jvecs, remaining_jvecs], dim=0)
        ee_flag = torch.cat([ee_flag, remaining_ee_link], dim=0)
        movable_joints = torch.cat([movable_joints, remaining_movable_joints], dim=0)

        n_joint_val_enc_dim = cfg.OBSERVATION.JOINT_VALUE_ENCODER.DIM
        link_encoder_sinusoid_freq = ifunc.get_sinusoid_frequencies(
            num_freqs=n_joint_val_enc_dim // 2,
            min_wavelength=cfg.OBSERVATION.JOINT_VALUE_ENCODER.MIN_WAVELENGTH,
            max_wavelength=cfg.OBSERVATION.JOINT_VALUE_ENCODER.MAX_WAVELENGTH,
        )
        
        movable_joints = movable_joints.bool()

        # it is 7 length vector
        self.__setattr__(
            "robo_base",
            ObsTerm(
                func=ifunc.robo_base_vec,
                params={
                    "asset_cfg": SceneEntityCfg(f"robot_{robo_id}"),
                },
            ),
        )

        self.__setattr__(
            "robo_link",
            ObsTerm(
                func=ifunc.get_link_obs,
                params={
                    "asset_cfg": SceneEntityCfg(
                        f"robot_{robo_id}",
                        body_names=lnames,
                        joint_names=movable_jnames,
                        preserve_order=True,
                    ),
                    "lvecs": lvecs,
                    "jvecs": jvecs,
                    "ee_flag": ee_flag,
                    "movable_joints": movable_joints,
                    "joint_encoder_freqs": link_encoder_sinusoid_freq,
                    "jval_lb": jval_lb,
                    "jval_ub": jval_ub,
                    "action_bins": ifunc.get_action_bins(),
                },
            ),
        )

        # robot goal observations if it is reach task
        if robo_task == "reach":
            self.__setattr__(
                "robo_goal",
                ObsTerm(
                    func=ifunc.robo_goal_vec,
                    params={
                        "asset_name": f"robot_{robo_id}",
                    },
                ),
            )
            
        else:
            self.__setattr__(
                "robo_goal",
                ObsTerm(
                    func=ifunc.return_vec,
                    params={
                        "vec": torch.zeros(1, 7),
                    }
                )   
            )
            
            
        obstacles_ids = []
        movable_obj_ids = []
        
        for obj_id, obj in prob.obj_dict.items():
            if obj.static:
                obstacles_ids.append(obj_id)
            else:
                movable_obj_ids.append(obj_id)
                
        assert len(movable_obj_ids) <= 1, "Only one movable object is supported for now"
        
        
        # the movable and non-movable object are treated differently
        # in particular, the obstacles could be represented using depth image, 
        # but we always give the state info for movable object
        if len(movable_obj_ids) > 0:
            obj_id = movable_obj_ids[0]
            obj = prob.obj_dict[obj_id]
            obj_shape_vec = iu.get_obj_shape_vec(obj)
            self.__setattr__(
                "obj",
                ObsTerm(
                    func=ifunc.get_obj_obs,
                    params={
                        "asset_cfg": SceneEntityCfg(f"obstacle_{obj_id}"),                      
                        "shape_vec": obj_shape_vec,  
                    }
                )
            )

        else:
            obj_vec = torch.zeros(cfg.BENCH.MAX_NUM_OBJECTS, 7 + 6)
            self.__setattr__(
                "obj",
                ObsTerm(
                    func=ifunc.return_vec,
                    params={
                        "vec": obj_vec,
                    }
                )
            )
        
        obstacles_ids.sort()
        
        # for now also restrict to only one obstacle for easier batch processing
        assert len(obstacles_ids) <= 1, "Only one obstacle is supported for now"
        
        if len(obstacles_ids) > 0:
            obs_shape_vec = iu.get_obj_shape_vec(prob.obj_dict[obstacles_ids[0]])   
            self.__setattr__(
                "obstacle",
                ObsTerm(
                    func=ifunc.get_obj_obs,
                    params={
                        "asset_cfg": SceneEntityCfg(f"obstacle_{obstacles_ids[0]}"),                      
                        "shape_vec": obs_shape_vec,
                    }
                )
            )
        else:
            self.__setattr__(
                "obstacle",
                ObsTerm(
                    func=ifunc.return_vec,
                    params={
                        "vec": torch.zeros(1, 7 + 6),
                    }
                )
            )
        
        if robo_task == 'reach':
            obs_mask[-3, 0] = 1.0    # robo goal

        if len(movable_obj_ids) > 0:    # movable object present.
            obs_mask[-2, 0] = 1.0    # obj
            
        if len(obstacles_ids) > 0:    # obstacle present.
            obs_mask[-1, 0] = 1.0    # obs
            
        self.obs_mask = ObsTerm(
            func=ifunc.return_vec,
            params={
                "vec": obs_mask,
            }
        )
        self.act_mask = ObsTerm(
            func=ifunc.return_vec,
            params={
                "vec": movable_joints.view(-1, 1),
            }
        )
                
        # only if implicit observation is being used
        if cfg.OBSERVATION.ROBOT_ID:            
            self.robot_id = ObsTerm(
                func=ifunc.return_vec,
                params={
                    "vec": torch.tensor([[task_idx]], dtype=torch.long),
                }
            )
        
@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    def __init__(self, prob, robo_task, task_idx=0):
        # observation groups
        self.policy: FullStateObsCfg = FullStateObsCfg(prob, robo_task, task_idx)
