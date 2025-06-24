import torch

from isaaclab.assets import RigidObject, Articulation
import isaaclab.envs.mdp as mdp

from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs import ManagerBasedRLEnv
from collections.abc import Sequence

# only used in assert statement
from anybody.cfg import cfg as global_cfg
import wandb

def get_rigid_object_state(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    state = asset.data.root_state_w.clone()
    # the above state is in simulation world frame,
    scene_origin = env.scene.env_origins
    assert (
        scene_origin.shape[0] == state.shape[0]
    ), f"the scene origin shape {scene_origin.shape} and state shape {state.shape} do not match"

    state[:, :3] -= scene_origin

    assert state.shape == (
        env.num_envs,
        13,
    ), f"state shape {state.shape} does not match expected shape (13,)"
    # print(f"object state shape: {state.shape}")
    return state


def get_rigid_object_pose(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    return get_rigid_object_state(env, asset_cfg)[:, :7]


def get_rigid_object_pose_with_args(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, vec: torch.Tensor
) -> torch.Tensor:
    res1 = get_rigid_object_pose(env, asset_cfg)
    # print(f"avg height: {res1[:, 2].mean()}")

    res2 = return_vec(env, vec)
    gvec = get_obj_goal_command(env, asset_cfg.name)
    res3 = torch.cat([res1, res2, gvec], dim=1)
    # res3 = torch.cat([res1, res2], dim=1)
    return res3


def return_vec(env: ManagerBasedEnv, vec: torch.Tensor):
    new_vec = vec[None, ...]
    shape = vec.shape
    return new_vec.expand(env.num_envs, *shape).to(env.scene.device)
    

def robo_base_vec(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg):
    asset: Articulation = env.scene[asset_cfg.name]
    base_pose = asset.data.root_state_w[:, :7].clone()
    # the above state is in simulation world frame,
    scene_origin = env.scene.env_origins
    assert (
        scene_origin.shape[0] == base_pose.shape[0]
    ), f"the scene origin shape {scene_origin.shape} and base_pose shape {base_pose.shape} do not match"

    base_pose[:, :3] -= scene_origin
    return base_pose.unsqueeze(1)


def robo_base_vec_with_args(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    vec: torch.Tensor,
):
    res1 = robo_base_vec(env, asset_cfg)

    if global_cfg.OBSERVATION.MINIMAL:
        return res1[:, :3]

    res2 = return_vec(env, vec)
    res3 = torch.cat([res1, res2], dim=1)
    return res3


def robo_movable_link_vec_with_args(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    lvec: torch.Tensor,
    jvec: torch.Tensor,
    joint_idx: int,
    is_ee: bool,
    joint_encoder_freqs: torch.Tensor,
    jval_lb: torch.Tensor,
    jval_ub: torch.Tensor,
):
    # res1 = mdp.joint_pos(env, asset_cfg)
    res1 = encoded_joint_pos(env, asset_cfg, frequencies=joint_encoder_freqs, jval_lb=jval_lb, jval_ub=jval_ub)

    # res2 = return_vec(env, vec)
    if is_ee:
        gvec = get_robo_goal_command(env, asset_cfg.name, joint_encoder_freqs, joint_idx)
    else:
        gvec = torch.zeros(env.num_envs, global_cfg.OBSERVATION.GOAL_VEC_DIM).to(
            env.scene.device
        )
        gvec[:, 1:] = global_cfg.OBSERVATION.FILL_UP_VAL

    # if global_cfg.OBSERVATION.MASK_ROBO_MORPH:
    #     return torch.cat([res1, gvec], dim=1)  # ignore the lvec and jvec

    lvec = return_vec(env, lvec)
    jvec = return_vec(env, jvec)

    if global_cfg.OBSERVATION.LINK_POSE:
        link_pose = get_link_pose(env, asset_cfg)
        res3 = torch.cat([lvec, gvec, jvec, res1, link_pose], dim=1)
    else:
        res3 = torch.cat([lvec, gvec, jvec, res1], dim=1)

    # res3 = torch.cat([res2, res1], dim=1)
    return res3


def get_link_pose(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
):
    asset: Articulation = env.scene[asset_cfg.name]

    assert (
        len(asset_cfg.body_ids) == 1
    ), f"link pose should have 1 body, got {len(asset_cfg.body_ids)}."
    body_id = asset_cfg.body_ids[0]
    link_pose = asset.data.body_state_w[:, body_id, :7].clone()

    # the above state is in simulation world frame,
    scene_origin = env.scene.env_origins
    assert (
        scene_origin.shape[0] == link_pose.shape[0]
    ), f"the scene origin shape {scene_origin.shape} and base_pose shape {link_pose.shape} do not match"

    link_pose[:, :3] -= scene_origin
    return link_pose


def robo_fixed_link_vec(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    lvec: torch.Tensor,
    jvec: torch.Tensor,
):
    gvec = torch.zeros(env.num_envs, global_cfg.OBSERVATION.GOAL_VEC_DIM).to(
        env.scene.device
    )
    gvec[:, 1:] = global_cfg.OBSERVATION.FILL_UP_VAL

    jval = torch.zeros(env.num_envs, global_cfg.OBSERVATION.JOINT_VALUE_ENCODER.DIM).to(
        env.scene.device
    )

    # if global_cfg.OBSERVATION.MASK_ROBO_MORPH:
    #     return torch.cat([jval, gvec], dim=1)
    # else:
    lvec = return_vec(env, lvec)
    jvec = return_vec(env, jvec)

    if global_cfg.OBSERVATION.LINK_POSE:
        link_pose = get_link_pose(env, asset_cfg)
        result = torch.cat([lvec, gvec, jvec, jval, link_pose], dim=1)
    else:
        result = torch.cat([lvec, gvec, jvec, jval], dim=1)

    return result


def joint_pos_dist(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    joint_dividend: torch.Tensor,
    joint_range: torch.Tensor,
    target_jpos: torch.Tensor,
    jpos_threshold=10.0,
    success_reward=10.0,
    success_jpos_threshold=0.1,
    end_reward=False,
):
    # the true dist is when both goal and current joint pos are divided by joint_dividend
    # and the distance is measured

    # however, we also want to normalize by the joint range

    # so - first divide the target and current joint by the joint_dividend
    # take the difference
    # then divide by the true joint range (where there are no wraps around 2 * pi)

    joint_dividend = joint_dividend.to(env.scene.device)[None, :]
    joint_range = joint_range.to(env.scene.device)[None, :]

    
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]

    # target_jpos = target_jpos.to(env.scene.device)[None, :].expand(env.num_envs, -1)
    # get the target jpos from command manager
    target = env.command_manager.get_command("goal_command")[asset_cfg.name][
        "joint_pos"
    ]
    assert (
        target.shape == joint_pos.shape
    ), f"target_jpos shape {target.shape} does not match joint_pos shape {joint_pos.shape}"

    joint_pos = joint_pos % joint_dividend
    target = target % joint_dividend

    diff = torch.abs(joint_pos - target)
    normalized_diff = diff / joint_range

    dist = torch.sum(torch.square(normalized_diff), dim=1)

    # valid only when there is one robot, if there are multiple robots,
    # RL will cheat to repeatedly get the high reward

    if not end_reward:
        return dist

    done_mask = dist < success_jpos_threshold
    # give large positive reward if the distance is less than success_threshold
    penalty = torch.where(done_mask, -success_reward, dist)
    return penalty

    # return dist

    mask = dist < jpos_threshold
    return torch.where(mask, dist, torch.ones_like(dist) * jpos_threshold * 5)


def ee_obj_dist(
    env: ManagerBasedRLEnv,
    robo_asset_cfg: SceneEntityCfg,
    obj_asset_cfg: SceneEntityCfg,
    # pos_threshold=0.05,
):
    # robo_ee_pose = get_rigid_object_pose(env, robo_asset_cfg)
    asset: Articulation = env.scene[robo_asset_cfg.name]
    # get the end-effector pose in simulation frame
    ee_pose = asset.data.body_state_w[:, robo_asset_cfg.body_ids, :7].clone()
    assert ee_pose.shape[1:] == (
        1,
        7,
    ), f"ee_pose shape {ee_pose.shape} does not match expected shape (n_envs, 1, 7)"

    ee_pose = ee_pose.squeeze(1)
    # get the ee_pose in scene frame
    scene_origin = env.scene.env_origins
    ee_pose[:, :3] -= scene_origin

    ee_pos = ee_pose[:, :3]

    obj_pose = get_rigid_object_pose(env, obj_asset_cfg)
    obj_pos = obj_pose[:, :3]

    # print(f"ee_pos: {ee_pos[0, 0].item():.3f}, {obj_pos[0, 0].item():.3f}")

    dist = torch.sum(torch.square(ee_pos - obj_pos), dim=1)
    # if within certain threshold, no penalty
    return dist

    # mask = dist < pos_threshold
    # penalty = torch.where(mask, 0.0, dist)
    # return penalty


def ee_pose_dist(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    pos_only=False,
    success_reward=10.0,
    success_pose_threshold=0.1,
    end_reward=False,
):
    asset: Articulation = env.scene[asset_cfg.name]
    # get the end-effector pose in simulation frame
    ee_pose = asset.data.body_state_w[:, asset_cfg.body_ids, :7].clone()
    assert ee_pose.shape[1:] == (
        1,
        7,
    ), f"ee_pose shape {ee_pose.shape} does not match expected shape (n_envs, 1, 7)"

    ee_pose = ee_pose.squeeze(1)
    # get the ee_pose in scene frame
    scene_origin = env.scene.env_origins
    ee_pose[:, :3] -= scene_origin

    # target = target_pose.to(env.scene.device)[None, :].expand(env.num_envs, -1)
    target = env.command_manager.get_command("goal_command")[asset_cfg.name]["ee_pose"]

    assert target.shape == (
        env.num_envs,
        7,
    ), f"target shape {target.shape} does not match expected shape (n_envs, 7)"

    if pos_only:
        ee = ee_pose[:, :3]
        targ = target[:, :3]
    else:
        ee = ee_pose
        targ = target

    dist = torch.sum(torch.square(ee - targ), dim=1)

    if not end_reward:
        return dist

    # wandb.log("reach_threshold", success_pose_threshold)

    done_mask = dist < success_pose_threshold
    # give large positive reward if the distance is less than success_threshold

    if global_cfg.REWARD.SPARSE_REWARD:
        return torch.where(done_mask, -success_reward, 1.0)
    else:
        return torch.where(done_mask, -success_reward, dist)

def object_pose_dist(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_pose: torch.Tensor,
    pose_threshold=10.0,
    pos_only=False,
    success_reward=10.0,
    success_obj_pose_threshold=0.1,
    end_reward=False,
):
    obj_pose = get_rigid_object_pose(env, asset_cfg)
    assert obj_pose.shape == (
        env.num_envs,
        7,
    ), f"obj_pose shape {obj_pose.shape} does not match expected shape (n_envs, 7)"

    target = env.command_manager.get_command("goal_command")[asset_cfg.name]

    # target = target_pose.to(env.scene.device)[None, :].expand(env.num_envs, -1)
    assert target.shape == (
        env.num_envs,
        7,
    ), f"target shape {target.shape} does not match expected shape (n_envs, 7)"

    if pos_only:
        obj_pos = obj_pose[:, :3]
        target_pos = target[:, :3]
        dist = torch.sum(torch.square(obj_pos - target_pos), dim=1)
    else:
        dist = torch.sum(torch.square(obj_pose - target), dim=1)

    if not end_reward:
        return dist

    done_mask = dist < success_obj_pose_threshold
    # give large positive reward if the distance is less than success_threshold
    penalty = torch.where(done_mask, -success_reward, dist)
    return penalty


def object_x_dist(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    success_reward=10.0,
    x_dist_threshold=0.3,
):
    obj_pose = get_rigid_object_pose(env, asset_cfg)
    assert obj_pose.shape == (
        env.num_envs,
        7,
    ), f"obj_pose shape {obj_pose.shape} does not match expected shape (n_envs, 7)"

    # target = env.command_manager.get_command("goal_command")[asset_cfg.name]

    # assert (
    #     target[0, 0] > x_dist_threshold
    # ), f"target x position {target[0, 0]} is less than 0.299"
    # if obj_pose[0] > 0.3: reward +10.0
    # if obj_pose[0] < 0.3: distance to 0.3
    # changed from x to y

    penalty = torch.where(
        obj_pose[:, 1] > x_dist_threshold,
        -success_reward,
        x_dist_threshold - obj_pose[:, 1],
    )
    return penalty

    # if not end_reward:
    #     return penalty

    # done_mask = penalty < 0.0
    # # give large positive reward if the distance is less than success_threshold
    # penalty = torch.where(done_mask, -1.0, dist)
    # return penalty


def goal_reached(
    env: ManagerBasedRLEnv,
    robo_ee_pose_dict,
    obj_goal_dict,
    success_pose_threshold=0.01,
    success_obj_pose_threshold=0.01,
    x_dist_threshold=0.3,
    success_reward=10.0,
    ee_pos_only=False,
    obj_pos_only=False,
    obj_simple_task=False,
    scene_cfgs={},
):
    reached = torch.tensor([True] * env.num_envs, device=env.scene.device)

    # from anybody.utils.path_utils import get_figures_dir
    # save_dir = get_figures_dir() / "diff_ik" / "reached"
    # save_dir.mkdir(parents=True, exist_ok=True)

    # check if robots reached their goals
    for robo_id, target_pose in robo_ee_pose_dict.items():
        robo_id = int(robo_id)
        # asset_cfg = SceneEntityCfg(
        #     f"robot_{robo_id}",
        #     # joint_names=robo_joint_names_dict[robo_id],
        #     body_names=robo_ee_names_dict[robo_id], preserve_order=True
        # )
        # asset_cfg._resolve_body_names(env.scene)
        
        # print("Resolving asset cfg is inside goal_reached. need to make it an argument")

        asset_cfg = scene_cfgs[f"robot_{robo_id}"]
        dist = ee_pose_dist(
            env,
            asset_cfg=asset_cfg,
            success_pose_threshold=success_pose_threshold,
            pos_only=ee_pos_only,
            end_reward=False,
        )

        # with open(save_dir / f"reached_{robo_id}_ee.txt", "a") as f:
        #     f.write(f"{(dist < success_pose_threshold)[0].item()}\n")

        reached &= dist < success_pose_threshold

    # print(f"reached: {reached}")

    # for robo_id, target_jpos in robo_jpos_dict.items():
    #     # asset_cfg = SceneEntityCfg(
    #     #     f"robot_{robo_id}", joint_names=robo_joint_names_dict[robo_id], preserve_order=True
    #     # )
    #     # asset_cfg._resolve_joint_names(env.scene)

    #     asset_cfg = kwargs[f"robot_{robo_id}"]
    #     dist = joint_pos_dist(
    #         env,
    #         asset_cfg,
    #         joint_dividend=robo_joint_dividend_dict[robo_id],
    #         joint_range=robo_joint_range_dict[robo_id],
    #         target_jpos=target_jpos,
    #         jpos_threshold=success_jpos_threshold,
    #         end_reward=False,
    #     )

    #     # with open(save_dir / f"reached_{robo_id}_jpos.txt", "a") as f:
    #     #     f.write(f"{(dist < success_jpos_threshold)[0].item()}\n")

    #     reached &= dist < success_jpos_threshold

    # check if objects reached their goals
    for obj_id, target_pose in obj_goal_dict.items():
        obj_id = int(obj_id)
        # asset_cfg = SceneEntityCfg(f"obstacle_{obj_id}")
        asset_cfg = scene_cfgs[f"obstacle_{obj_id}"]

        if not obj_simple_task:
            dist = object_pose_dist(
                env,
                asset_cfg,
                target_pose,
                success_obj_pose_threshold,
                pos_only=obj_pos_only,
                end_reward=False,
            )

            # with open(save_dir / f"reached_{obj_id}.txt", "a") as f:
            #     f.write(f"{(dist < success_obj_pose_threshold)[0].item()}\n")

            reached &= dist < success_obj_pose_threshold

        else:
            dist = object_x_dist(
                env, asset_cfg, success_reward=success_reward, x_dist_threshold=x_dist_threshold
            )
            reached &= dist < 0.0

    assert reached.shape == (
        env.num_envs,
    ), f"reached shape {reached.shape} does not match expected shape (n_envs,)"
    return reached


def get_robo_goal_command(
    env: ManagerBasedRLEnv,
    asset_name: str,
    joint_encoder_freqs: torch.Tensor,
    # asset_cfg: SceneEntityCfg,
    joint_idx: int,
):
    joint_goal_embed_dim = joint_encoder_freqs.shape[0] * 2
    if joint_idx < 0:
        return torch.zeros(env.num_envs, 1 + joint_goal_embed_dim + 1 + 7).to(
            env.scene.device
        )

    command = env.command_manager.get_command("goal_command")
    if asset_name not in command:
        if global_cfg.OBSERVATION.MINIMAL:
            return torch.zeros(env.num_envs, 1 + 7).to(env.scene.device)
        else:        
            return torch.zeros(env.num_envs, 1 + joint_goal_embed_dim + 1 + 7).to(
                env.scene.device
            )

    # get command for the given joint index
    robo_command = command[asset_name]

    # check if joint_pos is in the command
    if "joint_pos" in robo_command:
        try:
            joint_pos = robo_command["joint_pos"][:, joint_idx].unsqueeze(1)
        except IndexError:
            import pdb

            pdb.set_trace()

        # embed the joint pos using sin and cos
        joint_pos = encode_joint_vals(joint_pos, joint_encoder_freqs)
        jpos_flag = torch.ones(env.num_envs, 1).to(env.scene.device)
    else:
        joint_pos = torch.zeros(env.num_envs, joint_goal_embed_dim).to(env.scene.device)
        jpos_flag = torch.zeros(env.num_envs, 1).to(env.scene.device)

    # check if goal pose in the command
    if "ee_pose" in robo_command:
        ee_pose = robo_command["ee_pose"]
        pose_flag = torch.ones(env.num_envs, 1).to(env.scene.device)
    else:
        ee_pose = (
            torch.ones(env.num_envs, 7).float().to(env.scene.device)
            * global_cfg.OBSERVATION.FILL_UP_VAL
        )
        # ee_pose = torch.zeros(env.num_envs, 7).to(env.scene.device)
        pose_flag = torch.zeros(env.num_envs, 1).to(env.scene.device)

    if global_cfg.OBSERVATION.MINIMAL:
        assert "joint_pos" not in robo_command, "joint_pos should not be in the command"
        result = torch.cat([pose_flag, ee_pose], dim=1)
        return result

    # need to up-weigh joint pos to be similar to ee_pose
    # joint_pos = joint_pos.expand(-1, 7)

    result = torch.cat([jpos_flag, joint_pos, pose_flag, ee_pose], dim=1)
    assert (
        result.shape[1] == global_cfg.OBSERVATION.GOAL_VEC_DIM
    ), f"result shape {result.shape} does not match expected shape {global_cfg.OBSERVATION.GOAL_VEC_DIM}"
    return result


def get_obj_goal_command(
    env: ManagerBasedRLEnv,
    asset_name: str,
):
    # get command for the given joint index
    command = env.command_manager.get_command("goal_command")
    if asset_name not in command:
        return torch.zeros(env.num_envs, 1 + 7).to(env.scene.device)

    obj_command = env.command_manager.get_command("goal_command")[asset_name]

    goal_flag = torch.ones(env.num_envs, 1).to(env.scene.device)

    return torch.cat([goal_flag, obj_command], dim=1)


def get_sinusoid_frequencies(num_freqs=6, min_wavelength=0.1, max_wavelength=12.0):
    wavelengths = torch.logspace(
        torch.log10(torch.tensor(min_wavelength)),
        torch.log10(torch.tensor(max_wavelength)),
        num_freqs,
    )
    frequencies = 1.0 / wavelengths
    return frequencies


def encode_joint_vals(joint_vals, frequencies):
    if global_cfg.OBSERVATION.JOINT_VALUE_ENCODER.TYPE == "sinusoidal":
        # joint_vals have shape: (num_envs, 1)
        # frequencies have shape: (num_freqs,)

        # the returned values have shape: (num_envs, num_freqs * 2)

        frequencies = frequencies.view(1, -1).to(
            joint_vals.device
        )  # Shape (1, num_freqs)
        prod = frequencies * joint_vals

        assert (
            prod.shape == (joint_vals.shape[0], frequencies.shape[1])
        ), f"prod shape {prod.shape} does not match expected shape (num_envs, num_freqs)"

        embedding = torch.cat([torch.sin(prod), torch.cos(prod)], dim=1)
        assert (
            embedding.shape == (joint_vals.shape[0], frequencies.shape[1] * 2)
        ), f"embedding shape {embedding.shape} does not match expected shape (num_envs, num_freqs * 2)"

        return embedding

    elif global_cfg.OBSERVATION.JOINT_VALUE_ENCODER.TYPE == "repeat":
        # we repeat the joint values for the number of frequencies * 2
        return joint_vals.expand(-1, frequencies.shape[0] * 2)


def encoded_joint_pos(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, frequencies: torch.Tensor, jval_lb, jval_ub
) -> torch.Tensor:
    """The joint positions of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    jvals = asset.data.joint_pos[:, asset_cfg.joint_ids]
    
    # normalize the joint values to -1 to 1
    # jvals has shape (num_envs, 1)
    jvals = (jvals - jval_lb) / (jval_ub - jval_lb) * 2 - 1
    
    return encode_joint_vals(jvals, frequencies)


def modify_termination_thd(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    rew_term_name: str,
    ter_term_name: str, 
    # start_threshold: float,
    # end_threshold: float,
    # n_updates: int,
):
    """Curriculum that modifies a reward weight a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        weight: The weight of the reward term.
        num_steps: The number of steps after which the change should be applied.
    """
    # assuming total steps are given by global_cfg.TRAINER.TIMESTEPS
    total_steps = global_cfg.CURRICULUM.NUM_STEPS
    update_every = total_steps // global_cfg.CURRICULUM.N_UPDATES    
    
    termination_factor = torch.linspace(
        global_cfg.CURRICULUM.START_TERMINATION_THD, 1.0, global_cfg.CURRICULUM.N_UPDATES
    )
    
    if env.common_step_counter > total_steps:
        return
        
    if env.common_step_counter % update_every == 0:
        idx = env.common_step_counter // update_every
        
        if idx >= global_cfg.CURRICULUM.N_UPDATES:
            return
        
        term_cfg = env.reward_manager.get_term_cfg(rew_term_name)
        
        if rew_term_name.endswith("dist") :
            thd = termination_factor[idx] * global_cfg.TERMINATION.SUCCESS_OBJ_X_DIST_THRESHOLD
            # update rew settings
            term_cfg.params['x_dist_threshold'] = thd
            
        else:
            # it is the pose threshold that needs modification
            thd = termination_factor[idx] * global_cfg.TERMINATION.SUCCESS_OBJ_POSE_THRESHOLD
            # update rew settings
            term_cfg.params['success_obj_pose_threshold'] = thd
            
        env.reward_manager.set_term_cfg(rew_term_name, term_cfg)
        
        # update termination settings
        term_cfg = env.termination_manager.get_term_cfg(ter_term_name)
        thd1 = termination_factor[idx] * global_cfg.TERMINATION.SUCCESS_OBJ_X_DIST_THRESHOLD
        thd2 = termination_factor[idx] * global_cfg.TERMINATION.SUCCESS_OBJ_POSE_THRESHOLD
        term_cfg.params["x_dist_threshold"] = thd1
        term_cfg.params["success_obj_pose_threshold"] = thd2
        
        env.termination_manager.set_term_cfg(ter_term_name, term_cfg)
        
        print(f"Step: {env.common_step_counter}/{total_steps}. Updating thd for {rew_term_name} and {ter_term_name} to {thd1} and {thd2}.")



def modify_reach_thd(
    env: ManagerBasedRLEnv,
    env_ids,
    rew_term_name: str,
    goal_term_name: str,
):
    total_steps = global_cfg.CURRICULUM.NUM_STEPS
    update_every = total_steps // global_cfg.CURRICULUM.N_UPDATES
    
    
    # initial threshold is 100 times the final threshold
    # thd_scale = torch.linspace(float(global_cfg.CURRICULUM.N_UPDATES), 1.0, global_cfg.CURRICULUM.N_UPDATES)
    
    # make the threshold decrease fast initially (in the first half) and then slow down
    initial_thd_val = 500.0
    mid_thd_val = 50.0
    final_thd_val = 1.0
        
    if env.common_step_counter > total_steps:
        return
        
    if env.common_step_counter % update_every == 0:
        
        idx = env.common_step_counter // update_every
        
        if idx <= global_cfg.CURRICULUM.N_UPDATES // 2:
            # the first half of updates - decrease the threshold fast
            frac = idx / (global_cfg.CURRICULUM.N_UPDATES // 2)
            thd_scale = initial_thd_val * (1 - frac) + mid_thd_val * frac
        else:
            # quadratic decrease in the second half
            frac = (idx - global_cfg.CURRICULUM.N_UPDATES // 2) / (global_cfg.CURRICULUM.N_UPDATES // 2)
            # when frac is 0.0, we want to output the mid_thd_val
            # when frac is 1.0, we want to output the final_thd_val
            # f(x) = 49.0(x-1) ** 2 + 1.0
            thd_scale = (mid_thd_val - final_thd_val) * (frac - 1) ** 2 + final_thd_val
        
        term_cfg = env.reward_manager.get_term_cfg(rew_term_name)
        term_cfg.params["success_pose_threshold"] = global_cfg.TERMINATION.SUCCESS_POSE_THRESHOLD * thd_scale
        env.reward_manager.set_term_cfg(rew_term_name, term_cfg)
        
        term_cfg = env.termination_manager.get_term_cfg(goal_term_name)
        term_cfg.params["success_pose_threshold"] = global_cfg.TERMINATION.SUCCESS_POSE_THRESHOLD * thd_scale
        env.termination_manager.set_term_cfg(goal_term_name, term_cfg)
        
        print(f"Step: {env.common_step_counter}/{total_steps}. Updating thd for {rew_term_name} to {global_cfg.TERMINATION.SUCCESS_POSE_THRESHOLD * thd_scale}.")
    


def encoded_joint_pos_batched(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, frequencies: torch.Tensor, jval_lb:torch.Tensor, jval_ub: torch.Tensor
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    jvals = asset.data.joint_pos[:, asset_cfg.joint_ids]

    # jval_lb has shape (num_joints,)
    # jval_ub has shape (num_joints,)

    jval_lb = jval_lb.to(jvals.device)
    jval_ub = jval_ub.to(jvals.device)
    # normalize the joint values to -1 to 1
    # jvals has shape (num_envs, n_joints)
    jvals = (jvals - jval_lb.unsqueeze(0)) / (jval_ub - jval_lb).unsqueeze(0) * 2 - 1
    
    # joint_vals have shape: (num_envs, num_joints)
    # we want to encode each joint value with the given frequencies
    # the frequencies have shape: (num_freqs,)
    # the returned values should have shape: (num_envs, num_joints, num_freqs * 2)
    
    
    if global_cfg.OBSERVATION.JOINT_VALUE_ENCODER.TYPE == 'sinusoidal':
        frequencies = frequencies.view(1, 1, -1).to(
            jvals.device
        )
        prod = frequencies * jvals.unsqueeze(2)
        embedding = torch.cat([torch.sin(prod), torch.cos(prod)], dim=2)
        return embedding
    
    elif global_cfg.OBSERVATION.JOINT_VALUE_ENCODER.TYPE == 'repeat':
        return jvals.unsqueeze(2).expand(-1, -1, frequencies.shape[0] * 2)
    
    else:
        raise ValueError("Invalid joint value encoder type")
        

def get_link_pose_batched(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
):
    asset: Articulation = env.scene[asset_cfg.name]
    link_pose = asset.data.body_state_w[:, asset_cfg.body_ids, :7].clone()
    scene_origin = env.scene.env_origins
    link_pose[:, :, :3] -= scene_origin.unsqueeze(1)
    return link_pose


def get_link_obs(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, 
                 lvecs: torch.Tensor, jvecs: torch.Tensor, 
                 ee_flag: torch.Tensor, movable_joints: torch.Tensor, joint_encoder_freqs: torch.Tensor,
                 jval_lb: torch.Tensor, jval_ub: torch.Tensor, action_bins=torch.Tensor):   
    
    # the observation output here is of the form
    # links stacked together, [batch_size, n_links, n_features]
    # the n_features are [link_vec, ee_flag, joint_vec, joint_val_encodeed]
    
    # get the joint values
    jvals_encoded = encoded_joint_pos_batched(env, asset_cfg, frequencies=joint_encoder_freqs, jval_lb=jval_lb, jval_ub=jval_ub)
    # jvals have shape (batch_size, n_joints, n_features)
    lvecs = lvecs.to(env.scene.device)
    jvecs = jvecs.to(env.scene.device)
    ee_flag = ee_flag.to(env.scene.device)
    # repeat ee_flag cfg.OBSERVATION.EE_FLAG_DIM times
    movable_joints = movable_joints.to(env.scene.device)

    # expand the given vectors to the batch size
    lvecs = lvecs.unsqueeze(0).repeat(env.num_envs, 1, 1)
    jvecs = jvecs.unsqueeze(0).repeat(env.num_envs, 1, 1)
    ee_flag = ee_flag.unsqueeze(0).repeat(env.num_envs, 1, 1)
    
    # put the jvals where the joints are movable
    jvals = torch.zeros(env.num_envs, lvecs.shape[1], jvals_encoded.shape[2]).to(env.scene.device)
    jvals[:, movable_joints, :] = jvals_encoded
    
    if global_cfg.OBSERVATION.MASK_ROBO_MORPH:
        lvecs = torch.zeros_like(lvecs)
        jvecs = torch.zeros_like(jvecs)
        ee_flag = torch.zeros_like(ee_flag)
        
        return torch.cat([lvecs, ee_flag, jvecs, jvals], dim=2)

    elif global_cfg.OBSERVATION.LINK_POSE:
        link_poses = get_link_pose_batched(env, asset_cfg)
        # need to fill up the link poses with remaining values
        n_envs, n_links, _ = link_poses.shape
        n_rem_links = lvecs.shape[1] - n_links
        link_poses = torch.cat([link_poses, torch.zeros(n_envs, n_rem_links, 7).to(env.scene.device)], dim=1)
        return torch.cat([lvecs, ee_flag, jvecs, jvals, link_poses], dim=2)
    elif global_cfg.OBSERVATION.PREV_ACTION:
        prev_action = env.action_manager.action
        _actions = process_actions(prev_action, action_bins).unsqueeze(-1)
        # _actions: (batch_size, n_links, 1)
        dones = env.termination_manager.dones
        # dones: (batch_size,)
        dones = dones.unsqueeze(1).expand(-1, lvecs.shape[1]).unsqueeze(-1)
        action_dones = torch.cat([_actions, dones], dim=-1)
        return torch.cat([lvecs, ee_flag, jvecs, jvals, action_dones], dim=2)
    else:
        return torch.cat([lvecs, ee_flag, jvecs, jvals], dim=2)


def robo_goal_vec(env: ManagerBasedRLEnv, asset_name: SceneEntityCfg):    
    command = env.command_manager.get_command("goal_command")
    assert asset_name in command, f"asset {asset_name} not in command, got {command.keys()}"

    ee_pose_command = command[asset_name]["ee_pose"]
    assert ee_pose_command.shape == (env.num_envs, 7), f"ee_pose_command shape {ee_pose_command.shape} does not match expected shape (n_envs, 7)"
    return ee_pose_command.unsqueeze(1)

def get_obj_obs(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg, shape_vec: torch.Tensor
):
    shape_vec = shape_vec.to(env.scene.device)
    shape_vec = shape_vec.unsqueeze(0).repeat(env.num_envs, 1, 1)
    
    asset: RigidObject = env.scene[asset_cfg.name]
    wf_pose = asset.data.root_state_w[:, :7].clone()
    # the above state is in simulation world frame,
    scene_origin = env.scene.env_origins
    assert (
        scene_origin.shape[0] == wf_pose.shape[0]
    ), f"the scene origin shape {scene_origin.shape} and state shape {wf_pose.shape} do not match"

    wf_pose[:, :3] -= scene_origin

    return torch.cat([wf_pose.unsqueeze(1), shape_vec], dim=2)


def get_obstacle_pose(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg
):    
    asset: RigidObject = env.scene[asset_cfg.name]
    wf_pose = asset.data.root_state_w[:, :7].clone()
    # the above state is in simulation world frame,
    scene_origin = env.scene.env_origins
    assert (
        scene_origin.shape[0] == wf_pose.shape[0]
    ), f"the scene origin shape {scene_origin.shape} and state shape {wf_pose.shape} do not match"

    wf_pose[:, :3] -= scene_origin
    return wf_pose.unsqueeze(1) 

    # return torch.cat([wf_pose.unsqueeze(1), shape_vec], dim=2)



def applied_torque_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize applied torques if they cross the limits.

    This is computed as a sum of the absolute value of the difference between the applied torques and the limits.

    .. caution::
        Currently, this only works for explicit actuators since we manually compute the applied torques.
        For implicit actuators, we currently cannot retrieve the applied torques from the physics engine.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    # TODO: We need to fix this to support implicit joints.
    out_of_limits = torch.abs(
        asset.data.applied_torque[:, asset_cfg.joint_ids] - asset.data.computed_torque[:, asset_cfg.joint_ids]
    )
    return torch.sum(out_of_limits, dim=1)


def action_rate_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2-kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)


def action_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using L2-kernel."""
    return torch.sum(torch.square(env.action_manager.action), dim=1)


def ee_pose_dist_with_action_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    pos_only=False,
    # energy_dist_threshold=0.04,
    success_reward=10.0,
    success_pose_threshold=0.1,
    end_reward=False,
):
    asset: Articulation = env.scene[asset_cfg.name]
    # get the end-effector pose in simulation frame
    ee_pose = asset.data.body_state_w[:, asset_cfg.body_ids, :7].clone()
    assert ee_pose.shape[1:] == (
        1,
        7,
    ), f"ee_pose shape {ee_pose.shape} does not match expected shape (n_envs, 1, 7)"

    ee_pose = ee_pose.squeeze(1)
    # get the ee_pose in scene frame
    scene_origin = env.scene.env_origins
    ee_pose[:, :3] -= scene_origin

    # target = target_pose.to(env.scene.device)[None, :].expand(env.num_envs, -1)
    target = env.command_manager.get_command("goal_command")[asset_cfg.name]["ee_pose"]

    assert target.shape == (
        env.num_envs,
        7,
    ), f"target shape {target.shape} does not match expected shape (n_envs, 7)"

    if pos_only:
        ee = ee_pose[:, :3]
        targ = target[:, :3]
    else:
        ee = ee_pose
        targ = target

    dist = torch.sum(torch.square(ee - targ), dim=1)

    if not end_reward:
        return dist

    done_mask = dist < success_pose_threshold
    # give large positive reward if the distance is less than success_threshold

    if global_cfg.REWARD.SPARSE_REWARD:
        dist_penalty = torch.where(done_mask, -success_reward, 1.0)
    else:
        dist_penalty = torch.where(done_mask, -success_reward, dist)
        
    # action penalty
    # action_penalty = torch.ones_like(dist_penalty)
    return dist_penalty
    
    # action_penalty = torch.sum(torch.square(env.action_manager.action), dim=1)
    # action_penalty[~done_mask] = 0.0
    # return dist_penalty + global_cfg.REWARD.JOINT_MAG_WEIGHT * action_penalty
    
    

def joint_pos_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = -(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    return (2 * torch.sum(out_of_limits, dim=1)).square()


def get_action_bins():

    action_min = global_cfg.MODEL.POLICY.ACTION_MIN
    action_max = global_cfg.MODEL.POLICY.ACTION_MAX
    
    # assert action_min == (-action_max), "Only symmetric action space is supported"
    
    # get symmetric exponential bins
    n_pos_bins = (global_cfg.ACTION.NUM_BINS - 1) // 2     # 1 is for sign bit
    exp_base = torch.tensor([global_cfg.ACTION.BIN_BASE])
    pos_action_bins = torch.exp(torch.arange(n_pos_bins) * torch.log(exp_base))
    pos_action_bins = pos_action_bins * action_max / pos_action_bins[-1]
    n_neg_bins = global_cfg.ACTION.NUM_BINS - n_pos_bins - 1
    neg_action_bins = -torch.exp(torch.arange(n_neg_bins) * torch.log(exp_base))
    neg_action_bins = neg_action_bins * abs(action_min) / neg_action_bins[-1].abs()
    neg_action_bins = neg_action_bins.flip(0)
    
    action_bins = torch.cat([neg_action_bins, torch.zeros(1), pos_action_bins])
    return action_bins


def process_actions(actions, action_bins) -> torch.Tensor:
    assert not global_cfg.ACTION.ABSOLUTE, "Only relative actions are supported"
    if global_cfg.ACTION.DISCRETE:
        _actions = action_bins[actions.long()]
    else:
        _actions = actions
    return _actions
