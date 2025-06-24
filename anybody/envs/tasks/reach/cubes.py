import numpy as np
from numpy.random import uniform as U

from urdfpy import URDF

from anybody.cfg import cfg
import anybody.envs.tasks.env_utils as eu
from anybody.utils.path_utils import get_robot_morphs_dir
from anybody.utils.utils import transform_pcd, save_pickle
import random
from copy import deepcopy

from scipy.spatial.transform import Rotation as R

# allowed_sizes = [2, 3, 4, 5]
# allowed_lengths = np.linspace(0.5, 0.2, 4)

# copied from cube_urdf.py
# n_cubes = [1, 2, 3, 4, 5]
# fnames = [f"{n}_{i}" for i in range(2) for n in n_cubes]

fnames = [f"cube_{i}" for i in range(5)]

def get_robot_height(robot):
    robo_urdfpy = URDF.load(robot.robot_urdf)
    link_fk = robo_urdfpy.collision_trimesh_fk(cfg=robot.joint_pos)
    min_z = []
    for m, p in link_fk.items():
        vs = m.vertices
        vs = transform_pcd(vs, p)
        min_z.append(vs[:, 2].min())

    min_z = np.array(min_z).min()
    ht = -min_z
    return ht


def get_all_joint_pos(robo_dict, obj_dict, return_dict=True):
    joint_pos = {}
    global_obj_dict = deepcopy(obj_dict)

    max_tries = 100

    # obtain the joint positions for each robot
    for robo_id, robo in robo_dict.items():
        jpos = eu.get_non_colliding_jpos_general(
            robo, global_obj_dict, art_jdict=joint_pos, max_tries=max_tries, return_dict=return_dict
        )

        if jpos is None:
            return None

        else:
            global_obj_dict[robo_id] = robo
            joint_pos[robo_id] = jpos

    return joint_pos


def get_v1_env(robo_type, robo_name, seed=0):
    np.random.seed(seed)
    random.seed(seed)

    # ncubes = np.random.choice([1, 2, 3, 4])

    ncubes = 1
    robo_dict = {}
    joint_names = {}

    for idx in range(ncubes):
        # s = random.choice(fnames)
        robot = eu.get_robot(robo_type='cubes', robo_fname=robo_name, robo_id=idx)
        ht = get_robot_height(robot)
        robot.pose[2, 3] = ht
        robot.robot_id = idx
        robo_dict[idx] = robot
        joint_names[idx] = robot.act_info["joint_names"]

    # add obstacles
    obj_dict = {}
    n_obs = np.random.choice(10)
    n_obs = 1
    
    for idx in range(n_obs):
        p = np.eye(4)
        p[:2, 3] = U(-0.5, 0.5, 2)
        obj_shape = U(0.1, 0.2, 3).tolist()
        p[2, 3] = obj_shape[2] / 2
        p[:3, :3] = R.from_euler("xyz", [0, 0, U(-np.pi, np.pi)]).as_matrix()
        cuboid = eu.Prim(
            obj_type="box",
            obj_id=idx + ncubes,
            obj_shape=obj_shape,
            pose=p,
            static=True,
        )
        obj_dict[cuboid.obj_id] = cuboid

    # finding collision free joint pos
    init_joints = get_all_joint_pos(robo_dict, obj_dict)
    goal_joints = get_all_joint_pos(robo_dict, obj_dict)

    if (init_joints is None) or (goal_joints is None):
        raise ValueError("Could not find non-colliding poses")

    # define the initial state
    init_robo_state_dict = {
        k: eu.RobotState(joint_pos=v) for k, v in init_joints.items()
    }

    # define the goal state
    goal_robo_state_dict = {
        k: eu.RobotState(joint_pos=v) for k, v in goal_joints.items()
    }
    
    def get_cfg_fn():
        
        _init_joints = get_all_joint_pos(robo_dict, obj_dict, return_dict=False)
        if _init_joints is None:
            return None
        
        return _init_joints, _init_joints, {}, {}
        
    cfg_name = eu.save_randomized_configs(
        robo_dict=robo_dict,
        get_cfg_fn=get_cfg_fn,
        task_type="reach",
        robo_type='cubes',
        robo_name=robo_name,
        variation="v1",
        seed=seed,
    )   

    return eu.ProblemSpec(
        robot_dict=robo_dict,
        obj_dict=obj_dict,
        init_robo_state_dict=init_robo_state_dict,
        goal_robo_state_dict=goal_robo_state_dict,
        init_obj_state_dict={},
        goal_obj_state_dict={},
        ground=-0.005,
        additional_configs=cfg_name,
    )


def get_v1_pose_env(robo_type, robo_name, seed):
    return eu.get_pose_env(robo_type, robo_name, seed, get_v1_env)


cube_joint_envs = {}
cube_pose_envs = {}

cube_dir = get_robot_morphs_dir() / "cubes"
cube_fnames = [f.name for f in cube_dir.iterdir() if f.is_dir()]

for cube_fname in cube_fnames:
    potential_cube_urdf = cube_dir / cube_fname / f"{cube_fname}.urdf"
    if not potential_cube_urdf.exists():
        continue

    cube_joint_envs[f'{cube_fname}_v1'] = (get_v1_env, 'cubes', cube_fname)
    cube_pose_envs[f'{cube_fname}_v1'] = (get_v1_pose_env, 'cubes', cube_fname)

# cube_pose_envs = {
#     "v1": lambda seed: eu.get_pose_env("v1", cube_joint_envs, seed),
# }

# cube_pose_envs = {}

# target type
envs = {
    "joint": cube_joint_envs,
    "pose": cube_pose_envs,
}
