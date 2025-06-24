import numpy as np
from numpy.random import uniform as U
from scipy.spatial.transform import Rotation as R
from urdfpy import URDF

import anybody.envs.tasks.env_utils as eu

# from anybody.utils.vis_server import VizServer
from anybody.utils.path_utils import get_robot_morphs_dir
import random
from copy import deepcopy
from scipy.optimize import minimize

import trimesh

urdf_names = [f"bot_{i}" for i in range(4)]


def get_robot(idx=2, robo_id=0):
    # initialize the stick model
    bot_urdf = (
        get_robot_morphs_dir()
        / "simple_bot"
        / f"{urdf_names[idx]}"
        / f"{urdf_names[idx]}.urdf"
    )

    joint_names = [
        "joint_base_rev",
        "joint_1_rev",
        "joint_2_rev",
        "joint_3_rev",
        "joint_ee_3_rev",
        "joint_ee_left1_rev",
        "joint_ee_left2_rev",
        "joint_ee_right1_rev",
        "joint_ee_right2_rev",
    ]
    default_jvals = [np.pi / 2, -1.0, 0.7, 1.0, 0.0, 0.5, -0.5, -0.5, 0.5]

    urdfpy_robot = URDF.load(bot_urdf)
    jlimits = {j.name: j.limit for j in urdfpy_robot.joints}
    joint_lb = [jlimits[jname].lower for jname in joint_names]
    joint_ub = [jlimits[jname].upper for jname in joint_names]

    robot = eu.Robot(
        control_type="joint",
        act_info={
            "joint_names": joint_names,
            "joint_lb": joint_lb,
            "joint_ub": joint_ub,
        },
        robot_id=robo_id,
        robot_urdf=str(bot_urdf),
        ee_link="ee_link_base",
        pose=np.eye(4),
        joint_pos=dict(zip(joint_names, default_jvals)),
        joint_vel=dict(zip(joint_names, np.zeros(len(joint_names)))),
    )
    return robot


# to create grasp scenarios with simple robot and planar sheets
# we have blocks that need to be picked up and placed on a lifted plane


def get_base_env(n_obs, n_blocks, seed=0):
    np.random.seed(seed)
    random.seed(seed)

    s_idx = np.random.choice(len(urdf_names))

    robot = get_robot(idx=s_idx, robo_id=0)
    robo_dict = {robot.robot_id: robot}

    joint_names = robot.act_info["joint_names"]
    joint_lb = robot.act_info["joint_lb"]
    joint_ub = robot.act_info["joint_ub"]

    obst_shape_bounds = np.array([[0.04, 0.04, 0.04], [0.2, 0.2, 0.2]])

    block_shape_bounds = np.array([[0.02, 0.02, 0.04], [0.07, 0.07, 0.07]])

    max_tries = 100
    env_found = False

    for _ in range(max_tries):
        # the robot is at 0, 0, 0
        z_ground = U(-0.1, -0.00)

        obst_pos_bounds = np.array([[-0.5, -0.0, -0.2], [0.5, 0.5, 0.5]])

        block_pos_bounds = np.array([[-0.3, 0.2, 0.0], [0.3, 0.4, 1.0]])

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

        # add the lifted platform - close to z_ground
        platform_size = [U(0.05, 0.2), 0.05]
        platform_pose = np.eye(4)
        platform_pose[:2, 3] = U(block_pos_bounds[0], block_pos_bounds[1])[:2]
        platform_pose[2, 3] = z_ground + 0.05 / 2
        platform = eu.Prim(
            obj_type="cylinder",
            obj_id=0,
            obj_shape=platform_size,
            pose=platform_pose,
            static=True,
            friction=0.8,
        )

        obstacle_dict[platform.obj_id] = platform

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

                p = np.eye(4)
                p[:3, 3] = U(block_pos_bounds[0], block_pos_bounds[1])
                # p[:3, :3] = trimesh.transformations.random_rotation_matrix()
                p[2, 3] = z_ground + block_shape[2] / 2 + 0.001

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
        all_obs["ground"] = eu.add_ground_prim(z_ground, obj_id=n_obs + n_blocks + 2)

        for _ in range(100):
            init_jpos = U(joint_lb, joint_ub)
            init_jpos[-2:] = -init_jpos[-4:-2]

            is_collision = eu.check_collision(
                robot, dict(zip(joint_names, init_jpos)), all_obs, view=False
            )
            if not is_collision:
                found = True
                break
        # found 'init_jpos'

        if not found:
            continue

        # now we want to find target positions for block that are above the platform

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
                r = U(0, platform_size[0])
                platform_x, platform_y = platform_pose[:2, 3]
                t = U(0, 2 * np.pi)
                x = platform_x + r * np.cos(t)
                y = platform_y + r * np.sin(t)
                x = np.clip(x, block_pos_bounds[0][0], block_pos_bounds[1][0])
                y = np.clip(y, block_pos_bounds[0][1], block_pos_bounds[1][1])

                p = np.eye(4)
                p[:3, 3] = [
                    x,
                    y,
                    z_ground + block_shape[2] / 2 + 0.001 + platform_size[1],
                ]
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
    all_obs.pop("ground")

    return eu.ProblemSpec(
        robot_dict=robo_dict,
        obj_dict=all_obs,
        init_robo_state_dict=init_robo_state_dict,
        goal_robo_state_dict={r.robot_id: eu.RobotState() for r in robo_dict.values()},
        init_obj_state_dict=init_obj_state_dict,
        goal_obj_state_dict=goal_obj_state_dict,
        ground=z_ground,
    )


def get_v1_env(seed=0):
    return get_base_env(0, 1, seed)


def get_v2_env(seed=0):
    return get_base_env(0, 2, seed)


def get_v3_env(seed=0):
    n_obs = np.random.randint(1, 4)
    return get_base_env(n_obs, 1, seed)


bot_joint_envs = {"v1": get_v1_env, "v2": get_v2_env, "v3": get_v3_env}

envs = {"joint": bot_joint_envs}
