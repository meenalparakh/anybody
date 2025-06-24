import numpy as np
import torch

from isaaclab.utils import configclass

from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import EventTermCfg as EventTerm

import anybody.envs.tasks.env_utils as eu
from anybody.utils.utils import load_pickle
from anybody.utils.path_utils import get_problem_spec_dir
from isaaclab.utils.math import quat_from_matrix
from anybody.cfg import cfg

from isaaclab.envs import mdp


def set_random_jpos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    allowed_jpos: torch.Tensor,
    indices: torch.Tensor | None = None,
):
    """Choose random joint positions for the robot."""
    asset: Articulation = env.scene[asset_cfg.name]

    if indices is None:
        # randomly choose env_ids length number of random joint positions from allowed_jpos
        indices = torch.randint(0, allowed_jpos.shape[0], (env_ids.shape[0],))

    jpos = allowed_jpos[indices].to(env.scene.device)
    
    jvel = asset.data.default_joint_vel[env_ids].clone()
    # set jpos to the robot
    asset.write_joint_state_to_sim(jpos, jvel, env_ids=env_ids)


def set_random_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    allowed_pos_quat: torch.Tensor,
    indices: torch.Tensor | None = None,
):
    asset: RigidObject = env.scene[asset_cfg.name]

    if indices is None:
        indices = torch.randint(0, allowed_pos_quat.shape[0], (env_ids.shape[0],))

    pos_quat = allowed_pos_quat[indices].to(env.scene.device)

    pos_quat[:, :3] = pos_quat[:, :3] + env.scene.env_origins[env_ids]

    asset.write_root_pose_to_sim(pos_quat, env_ids=env_ids)


def reset_robot_and_objects(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    robo_dict,
    obj_dict,
    obstacle_dict,
    n_cfgs: int,
    asset_cfgs,
):
    indices = torch.randint(0, n_cfgs, (env_ids.shape[0],))

    for robo_id, robo_jpos in robo_dict.items():
        asset_cfg = asset_cfgs[f"robot_{robo_id}"]
        # asset_cfg = SceneEntityCfg(f"robot_{robo_id}", joint_names=robo_joint_names_dict[robo_id], preserve_order=True)
        set_random_jpos(env, env_ids, asset_cfg, robo_jpos, indices)

    for obj_id, obj_pos_quat in obj_dict.items():
        asset_cfg = asset_cfgs[f"obstacle_{obj_id}"]
        set_random_pose(env, env_ids, asset_cfg, obj_pos_quat, indices)
        
    for obs_id, obs_pos_quat in obstacle_dict.items():
        asset_cfg = asset_cfgs[f"obstacle_{obs_id}"]
        set_random_pose(env, env_ids, asset_cfg, obs_pos_quat, indices)


@configclass
class EventCfg:
    """Configuration for events."""

    def __init__(self, prob: eu.ProblemSpec):
        if (prob.additional_configs is not None) and (not cfg.FIXED_INITIAL_STATE):
            # load the additional configs
            configs = load_pickle(get_problem_spec_dir() / prob.additional_configs)

            # randomize over robots joint positions
            robo_init_cfgs = configs["init_robo_state_dict"]
            robo_joint_names_dict = {
                k: prob.robot_dict[k].act_info["joint_names"] for k in robo_init_cfgs
            }

            scene_cfgs = {}

            n_cfgs = None
            for robo_id in robo_init_cfgs:
                robo_init_cfgs[robo_id] = torch.from_numpy(
                    np.stack(robo_init_cfgs[robo_id])
                ).float()
                scene_cfgs[f"robot_{robo_id}"] = SceneEntityCfg(
                    f"robot_{robo_id}",
                    joint_names=robo_joint_names_dict[robo_id],
                    preserve_order=True,
                )

                if n_cfgs is None:
                    n_cfgs = robo_init_cfgs[robo_id].shape[0]
                else:
                    assert (
                        n_cfgs == robo_init_cfgs[robo_id].shape[0]
                    ), f"n_cfgs: {n_cfgs}, robo_init_cfgs[robo_id].shape[0]: {robo_init_cfgs[robo_id].shape[0]}"

            obj_init_cfgs = configs["init_obj_state_dict"]
            obj_init_poses = {}
            for obj_id, obj_init_cfg in obj_init_cfgs.items():
                poses = torch.from_numpy(np.stack(obj_init_cfg)).float()
                positions = poses[:, :3, 3]
                # positions[:, 2] = prob.obj_dict[obj_id].pose[2, 3]
                positions[:, 2] -= prob.ground
                rotations = poses[:, :3, :3]
                quats = quat_from_matrix(rotations)

                pos_quat = torch.cat([positions, quats], dim=1)

                assert pos_quat.shape == (len(obj_init_cfg), 7)
                obj_init_poses[obj_id] = pos_quat
                assert (
                    n_cfgs == poses.shape[0]
                ), f"n_cfgs: {n_cfgs}, poses.shape[0]: {poses.shape[0]}"
                scene_cfgs[f"obstacle_{obj_id}"] = SceneEntityCfg(f"obstacle_{obj_id}")


            # obstacle_states = configs['obstacles_state_dict']
            obstacle_states = configs.get("obstacles_state_dict", {})
            obstacle_posquat = {}
            # obstacle_poses = {}
            for obs_id, obs_X in obstacle_states.items():
                poses = torch.from_numpy(np.stack(obs_X)).float()
                positions = poses[:, :3, 3]
                rotations = poses[:, :3, :3]
                positions[:, 2] -= prob.ground
                quats = quat_from_matrix(rotations)

                pos_quat = torch.cat([positions, quats], dim=1)

                assert pos_quat.shape == (len(obs_X), 7)
                obstacle_posquat[obs_id] = pos_quat
                # obstacle_poses[obs_id] = poses
                assert (
                    n_cfgs == poses.shape[0]
                ), f"n_cfgs: {n_cfgs}, poses.shape[0]: {poses.shape[0]}"
                scene_cfgs[f"obstacle_{obs_id}"] = SceneEntityCfg(f"obstacle_{obs_id}")


            # for k, v in scene_cfgs.items():
            #     v.resolve

            # isaaclab flags error if keys are not strings while validating the configs
            _robo_init_cfgs = {str(k): v for k, v in robo_init_cfgs.items()}
            _obj_init_poses = {str(k): v for k, v in obj_init_poses.items()}
            _obstacle_posquat = {str(k): v for k, v in obstacle_posquat.items()}

            event_term_params = {
                "robo_dict": _robo_init_cfgs,
                "obj_dict": _obj_init_poses,
                "obstacle_dict": _obstacle_posquat,
                # "obstacle_X": obstacle_poses,
                "n_cfgs": n_cfgs,
                "asset_cfgs": scene_cfgs,
                # "kwargs": scene_cfgs,
            }

            # event_term_params.update(scene_cfgs)

            self.__setattr__(
                "reset_robot_and_objects",
                EventTerm(
                    func=reset_robot_and_objects,
                    mode="reset",
                    params=event_term_params,
                ),
            )