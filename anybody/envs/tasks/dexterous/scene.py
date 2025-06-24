import numpy as np
import random
from urdfpy import URDF
import trimesh

from anybody.utils.path_utils import get_robot_morphs_dir
from anybody.utils.utils import transform_pcd, transform_normal
import anybody.envs.tasks.env_utils as eu


def get_robot(robo_id, pose):
    # returns the kinova robot
    bot_urdf = get_robot_morphs_dir() / "dexterous" / "ur5_kinova.urdf"

    joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
        "j2n6s300_joint_finger_1",
        "j2n6s300_joint_finger_tip_1",
        "j2n6s300_joint_finger_2",
        "j2n6s300_joint_finger_tip_2",
        "j2n6s300_joint_finger_3",
        "j2n6s300_joint_finger_tip_3",
    ]

    robo_urdfpy = URDF.load(bot_urdf)

    joint_name_mappings = {j.name: j for j in robo_urdfpy.joints}

    joint_lb = [joint_name_mappings[j].limit.lower for j in joint_names]
    joint_ub = [joint_name_mappings[j].limit.upper for j in joint_names]

    jvals = [0.0, -1.712, 1.712, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]

    jvals[:6] = np.array([-1.0, -0.4, 0.4, -0.4, -0.4, 0]) * np.pi
    jvals[6:] = joint_lb[6:]

    robot = eu.Robot(
        control_type="joint",
        act_info={
            "joint_names": joint_names,
            "joint_lb": joint_lb,
            "joint_ub": joint_ub,
        },
        robot_id=robo_id,
        robot_urdf=str(bot_urdf),
        ee_link="j2n6s300_end_effector",
        pose=pose,
        joint_pos=dict(zip(joint_names, jvals)),
        joint_vel=dict(zip(joint_names, np.zeros(12))),
    )
    return robot


def get_cube_env(seed=0):
    np.random.seed(seed)
    random.seed(seed)

    robot = get_robot(0, np.eye(4))
    robo_dict = {robot.robot_id: robot}

    obs_shape_bounds = np.array([[0.03, 0.07, 0.07], [0.04, 0.08, 0.08]])

    p1 = np.eye(4)
    obj_shape = np.random.uniform(obs_shape_bounds[0], obs_shape_bounds[1])
    p1[:3, 3] = [0.0, 0.7, obj_shape[2] / 2 + 0.001]
    obj = eu.Prim(
        obj_type="box",
        obj_id=1,
        obj_shape=obj_shape.tolist(),
        pose=p1,
        static=False,
        friction=0.9,
    )

    obj_dict = {obj.obj_id: obj}
    obj_state_dict = {obj.obj_id: eu.PrimState(pose=obj.pose)}

    init_robo_state_dict = {robot.robot_id: eu.RobotState(joint_pos=robot.joint_pos)}

    return eu.ProblemSpec(
        robot_dict=robo_dict,
        obj_dict=obj_dict,
        init_robo_state_dict=init_robo_state_dict,
        goal_robo_state_dict={r.robot_id: eu.RobotState() for r in robo_dict.values()},
        init_obj_state_dict=obj_state_dict,
        goal_obj_state_dict={},
        ground=0.0,
    )


# # for cuboid
# def get_vec_bounds():
#     return np.array([
#         [0, -np.pi/2],
#         [2*np.pi, np.pi/2]
#     ])


# def vec_to_pt_cuboid(v, cuboid_size, cube_pose):
#     x = np.sin(v[1])
#     y = np.cos(v[0]) * np.cos(v[1])
#     z = np.sin(v[0]) * np.cos(v[1])

#     m = np.max(np.abs([x, y, z]))
#     m_idx = np.argmax(np.abs([x, y, z]))
#     normal = np.zeros(3)
#     normal[m_idx] = np.sign([x, y, z])[m_idx]

#     pt = np.array([x, y, z]) / m * (np.array(cuboid_size) / 2.0)

#     transformed_pt = transform_pcd(pt.reshape((1, 3)), cube_pose)
#     transformed_normal = transform_normal(normal.reshape((1, 3)), cube_pose)

#     return transformed_pt[0], transformed_normal[0]


# def vec_to_pt(obj_shape, v, cube_pose):
#     return vec_to_pt_cuboid(v, obj_shape, cube_pose)


# def check_collision(robot_urdfpy: URDF, base_pose, jdict, object_mesh_dict, object_pose_dict):
#     collision_manager = trimesh.collision.CollisionManager()

#     for obj_id in object_mesh_dict:
#         mesh = object_mesh_dict[obj_id]
#         pose = object_pose_dict[obj_id]
#         collision_manager.add_object(f"object_{obj_id}", mesh, transform=pose)

#     link_fk = robot_urdfpy.collision_trimesh_fk(jdict)
#     for idx, (mesh, pose) in enumerate(link_fk.items()):
#         dist = collision_manager.min_distance_single(
#             mesh, transform= base_pose @ pose
#         )
#         is_collision = (dist < 0.01)
#         if is_collision:
#             return True
#         else:
#             collision_manager.add_object(
#                 f"robot_{idx}", mesh, transform=base_pose @ pose
#             )
#     return False


envs = {"joint": {"v1": get_cube_env}}
