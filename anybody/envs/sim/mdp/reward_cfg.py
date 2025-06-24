import torch

from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp

from isaaclab.managers import SceneEntityCfg

# from omni.isaac.lab_tasks.manager_based.classic.cartpole.mdp.rewards import joint_pos_target_l2
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm


import anybody.envs.tasks.env_utils as eu
import anybody.envs.sim.utils as iu
import anybody.envs.sim.mdp.funcs as ifunc

from anybody.cfg import cfg

@configclass
class ReachRewardCfg:
    def __init__(self, prob: eu.ProblemSpec):

        robo_id = list(prob.robot_dict.keys())[0]
        robo = prob.robot_dict[robo_id]
        # assert robo.ee_link is not None
        
        self.robo_0_ee = RewTerm(
                func=ifunc.ee_pose_dist_with_action_penalty,
                weight=-cfg.REWARD.EE_POSE_WEIGHT,
                params={
                    "asset_cfg": SceneEntityCfg(
                        f"robot_{robo_id}", 
                        joint_names=prob.robot_dict[robo_id].act_info["joint_names"],
                        body_names=[robo.ee_link], 
                        preserve_order=True
                    ),
                    "pos_only": cfg.REWARD.EE_POSE_POS_ONLY,
                    # "energy_dist_threshold": cfg.REWARD.ENERGY_EE_THRESHOLD,
                    "success_reward": cfg.REWARD.REACH_REWARD,
                    "success_pose_threshold": cfg.TERMINATION.SUCCESS_POSE_THRESHOLD,
                    "end_reward": True,
                },
            )
        
        self.robo_0_acc = RewTerm(
            func=mdp.joint_acc_l2,
            weight=-cfg.REWARD.JOINT_ACC_WEIGHT,
            params={
                "asset_cfg": SceneEntityCfg(f"robot_{robo_id}"),
            }
        )
        
        ## if actions are diff - then we also want to penalize the policy if robot joints are out of bounds
        if not cfg.ACTION.ABSOLUTE:
            self.robo_0_jpos_limits = RewTerm(
                func=ifunc.joint_pos_limits,
                weight=-cfg.REWARD.JOINT_LIMITS_WEIGHT,
                params={
                    "asset_cfg": SceneEntityCfg(f"robot_{robo_id}"),
                }
            )



@configclass
class PushRewardCfg:
    def __init__(self, prob: eu.ProblemSpec):
        # precompute the target values for robots    
        # minimize energy rewards
        robo_id = list(prob.robot_dict.keys())[0]

        # reward for objects
        for obj_id in prob.goal_obj_state_dict:
            
            # we only check the x-coordinate, if it is > 0.3 then it is success.
            # else, we get the -distance from the x-coordinate to 0.3
            self.__setattr__(
                f"obj_{obj_id}_dist",
                RewTerm(
                    func=ifunc.object_x_dist,
                    weight=-cfg.REWARD.OBJ_DIST_WEIGHT,
                    params={
                        "asset_cfg": SceneEntityCfg(f"obstacle_{obj_id}"),
                        "success_reward": cfg.REWARD.COMPLETE_REWARD,
                        "x_dist_threshold": cfg.TERMINATION.SUCCESS_OBJ_X_DIST_THRESHOLD,
                    },
                )
            )                   
        
        for obj_id in prob.goal_obj_state_dict:
            # assuming there is only one robot
            robo_id = list(prob.robot_dict.keys())[0]
            ee_link = prob.robot_dict[robo_id].ee_link
            assert ee_link is not None
            self.__setattr__(
                f"ee_obj_{robo_id}_{obj_id}",
                RewTerm(
                    func=ifunc.ee_obj_dist,
                    weight=-cfg.REWARD.EE_OBJ_WEIGHT,
                    params={
                        "robo_asset_cfg": SceneEntityCfg(
                            f"robot_{robo_id}",
                            # joint_names=prob.robot_dict[robo_id].act_info["joint_names"],
                            body_names=ee_link,
                            preserve_order=True,
                        ),
                        "obj_asset_cfg": SceneEntityCfg(f"obstacle_{obj_id}"),
                        # "pos_threshold": cfg.REWARD.EE_OBJ_THRESHOLD,
                    },
                )
            )
                
        if not cfg.ACTION.ABSOLUTE:
            self.robo_0_jpos_limits = RewTerm(
                func=ifunc.joint_pos_limits,
                weight=-cfg.REWARD.JOINT_LIMITS_WEIGHT,
                params={
                    "asset_cfg": SceneEntityCfg(f"robot_{robo_id}"),
                }
            )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    def __init__(self, prob: eu.ProblemSpec, task):
        # (1) Time out
        self.time_out = DoneTerm(func=mdp.time_out, time_out=True)

        if task == "reach":
            early_terminate = cfg.TERMINATION.REACH_EARLY_TERMINATION
        else:
            early_terminate = cfg.TERMINATION.PUSH_EARLY_TERMINATION

        if early_terminate:
            # precompute the target values for robots
            robo_ee_pose_dict = {}
            robo_jpos_dict = {}
            robo_joint_names_dict = {}
            # robo_ee_names_dict = {}
            
            robo_joint_dividend_dict = {}
            robo_joint_range_dict = {}
            
            scene_cfgs = {}
            
            for robo_id, robo_state in prob.goal_robo_state_dict.items():
                if robo_state.ee_pose is not None:
                    # robo_ee_names_dict[robo_id] = prob.robot_dict[robo_id].ee_link
                    robo_ee_pose_dict[str(robo_id)] = iu.pose_encode(
                        torch.from_numpy(robo_state.ee_pose)
                    )
                    scene_cfgs[f"robot_{robo_id}"] = SceneEntityCfg(
                        f"robot_{robo_id}", 
                        # joint_names=robo_joint_names_dict[robo_id],
                        body_names=prob.robot_dict[robo_id].ee_link, preserve_order=True
                    )
                    
                if robo_state.joint_pos is not None:
                    jvals = []
                    for jname in prob.robot_dict[robo_id].act_info["joint_names"]:
                        jvals.append(robo_state.joint_pos[jname])

                    robo_jpos_dict[robo_id] = torch.tensor(jvals)
                    robo_joint_names_dict[robo_id] = prob.robot_dict[robo_id].act_info[
                        "joint_names"
                    ]
                    
                    joint_dividend = prob.robot_dict[robo_id].act_info.get('joint_dividend', None)
                    joint_range = prob.robot_dict[robo_id].act_info.get("joint_range", None)
                    if joint_range is None:
                        lb = prob.robot_dict[robo_id].act_info['joint_lb']
                        ub = prob.robot_dict[robo_id].act_info['joint_ub']
                        joint_range = torch.tensor(ub) - torch.tensor(lb)
                        joint_dividend = joint_range
                    else:
                        joint_range = torch.tensor(joint_range)
                        joint_dividend = torch.tensor(joint_dividend)
                    
                    robo_joint_dividend_dict[robo_id] = joint_dividend
                    robo_joint_range_dict[robo_id] = joint_range
                    
                    scene_cfgs[f"robot_{robo_id}"] = SceneEntityCfg(
                        f"robot_{robo_id}", 
                        joint_names=prob.robot_dict[robo_id].act_info["joint_names"], 
                        preserve_order=True
                    )

            # precompute the target values for objects
            obj_goal_dict = {}
            for obj_id, obj_state in prob.goal_obj_state_dict.items():
                if obj_state.pose is not None:
                    obj_goal_dict[str(obj_id)] = iu.pose_encode(torch.from_numpy(obj_state.pose))
                    scene_cfgs[f"obstacle_{obj_id}"] = SceneEntityCfg(f"obstacle_{obj_id}")

            goal_term_params = {
                    "robo_ee_pose_dict": robo_ee_pose_dict,
                    "obj_goal_dict": obj_goal_dict,
                    "success_pose_threshold": cfg.TERMINATION.SUCCESS_POSE_THRESHOLD,
                    "success_obj_pose_threshold": cfg.TERMINATION.SUCCESS_OBJ_POSE_THRESHOLD,
                    "x_dist_threshold": cfg.TERMINATION.SUCCESS_OBJ_X_DIST_THRESHOLD,
                    "success_reward": cfg.REWARD.COMPLETE_REWARD,
                    "ee_pos_only": cfg.REWARD.EE_POSE_POS_ONLY,
                    "obj_pos_only": cfg.REWARD.OBJ_POSE_POS_ONLY,
                    "obj_simple_task":cfg.REWARD.OBJ_SIMPLE_REWARD,
                    "scene_cfgs": scene_cfgs,
                }

            # (2) Goal reached
            self.goal_reached = DoneTerm(
                func=ifunc.goal_reached,
                params=goal_term_params,
                time_out=False,
            )



@configclass
class ReachCurriculumCfg:
    
    def __init__(self, prob: eu.ProblemSpec):
        
        # no curriculum for reach task
        pass

        # if cfg.CURRICULUM.ACTIVE:
            
        #     self.reach_curriculum = CurrTerm(
        #         func=ifunc.modify_reach_thd,
        #         params={
        #             "rew_term_name": "robo_0_ee",
        #             "goal_term_name": "goal_reached",   
        #         }
        #     )
        
@configclass
class PushCurriculumCfg:
    """Curriculum terms for the MDP."""

    def __init__(self, prob: eu.ProblemSpec):
        
        if cfg.CURRICULUM.ACTIVE:
        
            for obj_id in prob.goal_obj_state_dict:

                term_name = "goal_reached"
                rew_name = f"obj_{obj_id}_dist" if cfg.REWARD.OBJ_SIMPLE_REWARD else f"obj_{obj_id}_pose"

                self.__setattr__(
                    f"{obj_id}_curriculum",
                    CurrTerm(
                        func=ifunc.modify_termination_thd,
                        params={
                            "rew_term_name": rew_name,   
                            "ter_term_name": term_name,
                        }
                    )                
                )
