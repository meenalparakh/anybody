import numpy as np
from numpy.random import uniform as U

import anybody.envs.tasks.env_utils as eu
from anybody.utils.path_utils import get_robot_morphs_dir
from urdfpy import URDF

# from anybody.utils.vis_server import VizServer
import random

import trimesh

allowed_sizes = [2, 3, 4, 5]
allowed_lengths = np.linspace(0.5, 0.2, 4)

# the push tasks in n-link envs - can again be cube rearrangement tasks
def get_robo_params(robo_type, robo_fname):
    if robo_type == "nlink":
        robo: eu.Robot = eu.get_robot(robo_type='nlink', robo_fname=robo_fname, robo_id=0)
        # load the urdf file
        robo_urdfpy = URDF.load(robo.robot_urdf)
        # get the number of links
        nlinks = len(robo_urdfpy.links) - 2
        # get link length from geometry of box for link "link_0"
        try:
            for link in robo_urdfpy.links:
                if link.name == "link_0":
                    link_length = link.visuals[0].geometry.box.size[0]
                    break            
        except:
            import pdb; pdb.set_trace()
    
        return nlinks, link_length

    else:
        raise ValueError(f"Unknown robo_type: {robo_type}")


def get_base_env(robo_type, robo_fname, n_obs, n_blocks, seed=0, version_name="v1",  save_dirname=None, save_fname=None):
    np.random.seed(seed)
    random.seed(seed)

    # to get s and l - load robo urdf
    s, length = get_robo_params(robo_type, robo_fname)

    # s_idx = np.random.choice(len(allowed_sizes))
    # s = int(robo_fname.split("_")[0])
    # s_idx = allowed_sizes.index(s)
    # length = allowed_lengths[s_idx]

    robot = eu.get_robot(robo_type='nlink', robo_fname=robo_fname, robo_id=0)

    robo_dict = {robot.robot_id: robot}

    joint_names = robot.act_info["joint_names"]

    obst_shape_bounds = np.array([[0.04, 0.04, 0.04], [0.2, 0.2, 0.2]])
    block_shape_bounds = np.array([[0.15, 0.15, 0.15], [0.2, 0.2, 0.2]])

    max_tries = 100
    env_found = False

    for _ in range(max_tries):
        # the robot is at 0, 0, 0
        z_ground = -0.03
        r_min = length * 0.7
        r_max = length * (s - 0.5)

        obst_pos_bounds = np.array([[-1.0, -1, 0.0], [1.0, 1, 0.01]])

        obstacle_dict = {}

        # add obstacles
        for idx in range(n_obs):
            p1 = np.eye(4)
            p1[:3, 3] = U(obst_pos_bounds[0], obst_pos_bounds[1])
            cuboid1 = eu.Prim(
                obj_type="box",
                obj_id=idx + 1,
                obj_shape=U(obst_shape_bounds[0], obst_shape_bounds[1]).tolist(),
                pose=p1,
                static=True,
                friction=0.8,
            )
            obstacle_dict[cuboid1.obj_id] = cuboid1

        # initialize collision manager
        collision_manager = trimesh.collision.CollisionManager()
        for obj_id, obj in obstacle_dict.items():
            collision_manager.add_object(
                f"obstacle_{obj_id}", obj.mesh, transform=obj.pose
            )

        block_dict = {}
        # add blocks that do not collide with obstacles
        for idx in range(n_blocks):
            for _ in range(100):
                block_shape = U(block_shape_bounds[0], block_shape_bounds[1])
                block = eu.Prim(
                    obj_type="box",
                    obj_id=n_obs + 1 + idx,
                    obj_shape=block_shape.tolist(),
                    pose=np.eye(4),
                    static=False,
                )

                pz = z_ground + block_shape[2] / 2 + 0.001
                theta = U(2 * np.pi/3, 4 * np.pi/3)
                r = U(r_min, r_max)
                p = np.eye(4)

                p[:3, 3] = [r * np.sin(theta), r * np.cos(theta), pz]

                dist = collision_manager.min_distance_single(block.mesh, transform=p)

                is_collision = dist < 0.0
                if not is_collision:
                    block.pose = p
                    block_dict[block.obj_id] = block

                    collision_manager.add_object(
                        f"block_{idx}", block.mesh, transform=block.pose
                    )
                    break

        if len(block_dict) == 0:
            continue

        all_obs = {**obstacle_dict, **block_dict}

        found = False
        for _ in range(100):
            init_jpos = U(-np.pi, np.pi, s)
            is_collision = eu.check_collision(
                robot, dict(zip(joint_names, init_jpos)), all_obs, view=False
            )
            if not is_collision:
                found = True
                break
        # found 'init_jpos'

        if not found:
            continue

        # found a target block positions
        target_block_pos = {}
        collision_manager = trimesh.collision.CollisionManager()
        for obs_id, obst in obstacle_dict.items():
            collision_manager.add_object(
                f"obstacle_{obs_id}", obst.mesh, transform=obst.pose
            )

        block_goals_found = True
        for block_id, block in block_dict.items():
            found = False
            for k in range(100):
                pz = z_ground + block_shape[2] / 2 + 0.001
                theta = U(-np.pi/3, np.pi/3)
                r = U(r_min, r_max)
                p = np.eye(4)

                p[:3, 3] = [r * np.sin(theta), r * np.cos(theta), pz]

                dist = collision_manager.min_distance_single(block.mesh, transform=p)
                is_collision = dist < 0.0
                if not is_collision:
                    target_block_pos[block_id] = p
                    collision_manager.add_object(
                        f"block_{block_id}", block.mesh, transform=p
                    )
                    found = True
                    break
            if not found:
                block_goals_found = False
                break

        if not block_goals_found:
            continue

        env_found = True
        break

    if not env_found:
        return None

    init_robo_state_dict = {
        robot.robot_id: eu.RobotState(joint_pos=dict(zip(joint_names, init_jpos)))
    }
    goal_obj_state_dict = {k: eu.PrimState(pose=v) for k, v in target_block_pos.items()}
    init_obj_state_dict = {
        obj_id: eu.PrimState(pose=obj.pose) for obj_id, obj in block_dict.items()
    }

    # remove ground
    # all_obs.pop('ground')

    def obj_sampler(_r_min, _r_max, flag="init"):
        if flag == "init":
            theta = U(2 * np.pi / 3, 4 * np.pi / 3)
        else:
            theta = U(-np.pi/3, np.pi/3)
                
        _r = U(_r_min, _r_max)
        _p = np.eye(4)

        _p[:3, 3] = [_r * np.cos(theta), _r * np.sin(theta), 0.0]
        return _p

    def robo_jpos_sampler(_s):
        return U(-np.pi, np.pi, _s)
        # return [U(-0.1, 0.1), U(-np.pi, np.pi), U(-np.pi, np.pi)]

    def get_cfg_fn():
        return eu.get_randomized_push_simple_cfg(
            obstacle_dict,
            movable_obj_dict=block_dict,
            robot_dict=robo_dict,
            obj_sampler=obj_sampler,
            robot_sampler=robo_jpos_sampler,
            z_ground=z_ground,
            obj_sampler_args=[r_min, r_max],
            robo_sampler_args=[s],
        )

    cfg_name = eu.save_randomized_configs(
        robo_dict=robo_dict,
        get_cfg_fn=get_cfg_fn,
        task_type="push_simple",
        robo_type="nlink",
        robo_name=robo_fname,
        variation=version_name,
        seed=seed,
        save_dirname=save_dirname,
        save_fname=save_fname,
    )
    
    
    return eu.ProblemSpec(
        robot_dict=robo_dict,
        obj_dict=all_obs,
        init_robo_state_dict=init_robo_state_dict,
        goal_robo_state_dict={r.robot_id: eu.RobotState() for r in robo_dict.values()},
        init_obj_state_dict=init_obj_state_dict,
        goal_obj_state_dict=goal_obj_state_dict,
        ground=z_ground,
        additional_configs=cfg_name,
    )


def get_non_colliding_pos(robot, nlinks, obstacles={}, max_tries=100):
    joint_names = robot.act_info["joint_names"]

    for _ in range(max_tries):
        pos = [U(-np.pi + np.pi / 6, np.pi - np.pi / 6) for _ in range(nlinks)]
        # cum angles for each link
        is_collision = eu.check_collision(robot, dict(zip(joint_names, pos)), obstacles)
        if not is_collision:
            return pos
    return None


def get_v1_env(robo_type, robo_fname, seed=0, save_dirname=None, save_fname=None):
    return get_base_env(robo_type, robo_fname, n_obs=0, n_blocks=1, seed=seed, version_name="v1", save_dirname=save_dirname, save_fname=save_fname)


def get_v2_env(robo_type, robo_fname, seed=0, save_dirname=None, save_fname=None):
    return get_base_env(robo_type, robo_fname, n_obs=0, n_blocks=2, seed=seed, version_name="v2", save_dirname=save_dirname, save_fname=save_fname)


def get_v3_env(robo_type, robo_fname, seed=0, save_dirname=None, save_fname=None):
    return get_base_env(robo_type, robo_fname, n_obs=1, n_blocks=1, seed=seed, version_name="v3", save_dirname=save_dirname, save_fname=save_fname)


def get_v4_env(robo_type, robo_fname, seed=0, save_dirname=None, save_fname=None):
    n_obs = np.random.randint(2, 5)
    return get_base_env(robo_type, robo_fname, n_obs=n_obs, n_blocks=1, seed=seed, version_name="v4", save_dirname=save_dirname, save_fname=save_fname)


def get_v1_pose_env(robo_type, robo_fname, seed, save_dirname=None, save_fname=None):
    return eu.get_pose_env(get_v1_env, robo_type, robo_fname, seed, save_dirname=save_dirname, save_fname=save_fname)

def get_v2_pose_env(robo_type, robo_fname, seed, save_dirname=None, save_fname=None):
    return eu.get_pose_env(get_v2_env, robo_type, robo_fname, seed, save_dirname=save_dirname, save_fname=save_fname)

def get_v3_pose_env(robo_type, robo_fname, seed, save_dirname=None, save_fname=None):
    return eu.get_pose_env(get_v3_env, robo_type, robo_fname, seed, save_dirname=save_dirname, save_fname=save_fname)

def get_v4_pose_env(robo_type, robo_fname, seed, save_dirname=None, save_fname=None):
    return eu.get_pose_env(get_v4_env, robo_type, robo_fname, seed, save_dirname=save_dirname, save_fname=save_fname)




nlink_joint_envs = {}
nlink_pose_envs = {}

nlink_dir = get_robot_morphs_dir() / "nlink"
nlink_fnames = [f.name for f in nlink_dir.iterdir() if f.is_dir()]

for nlink_fname in nlink_fnames:
    potential_nlink_urdf = nlink_dir / nlink_fname / f"{nlink_fname}.urdf"
    if not potential_nlink_urdf.exists():
        continue

    # matches the description in base.py and __init__.py for general robots

    nlink_joint_envs.update(
        {
            f"{nlink_fname}_v1": (get_v1_env, "nlink", nlink_fname),
            f"{nlink_fname}_v2": (get_v2_env, "nlink", nlink_fname),
            f"{nlink_fname}_v3": (get_v3_env, "nlink", nlink_fname),
            f"{nlink_fname}_v4": (get_v4_env, "nlink", nlink_fname),
        }
    )
    
    nlink_pose_envs.update(
        {
            f"{nlink_fname}_v1": (get_v1_pose_env, "nlink", nlink_fname),
            f"{nlink_fname}_v2": (get_v2_pose_env, "nlink", nlink_fname),
            f"{nlink_fname}_v3": (get_v3_pose_env, "nlink", nlink_fname),
            f"{nlink_fname}_v4": (get_v4_pose_env, "nlink", nlink_fname),
        }
    )
    

# nlink_pose_envs = {}

# target type
envs = {"joint": nlink_joint_envs, "pose": nlink_pose_envs}
