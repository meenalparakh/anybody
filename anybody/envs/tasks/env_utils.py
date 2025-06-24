import os
import re
from tqdm import tqdm
import numpy as np
from numpy.random import uniform as U
from urdfpy import URDF
from dataclasses import dataclass
import typing as T
from typing import Optional as Opt
from copy import deepcopy
import trimesh
import torch

from anybody.utils import utils
from anybody.utils.vis_server import VizServer
from anybody.utils.path_utils import get_robot_morphs_dir, get_problem_spec_dir

from isaaclab.utils import math as math_utils
from anybody.cfg import cfg
from tqdm import tqdm

from anybody.morphs.robots import get_default_robo_cfg


@dataclass
class Prim:
    obj_type: str
    obj_id: int
    obj_shape: T.List[float]
    # shape is [x, y, z] for cuboid, [r, h] for cylinder, [r] for sphere

    pose: np.ndarray
    velocity: np.ndarray = np.zeros(3)
    ang_velocity: np.ndarray = np.zeros(3)

    friction: float = 0.8
    static: bool = True
    mass: float = 0.0  # relevant only when static is False

    def __post_init__(self):
        # define the object mesh
        if self.obj_type in ["box", "cylinder", "sphere"]:
            self.mesh = utils.load_mesh(self.obj_type, self.obj_shape)
        else:
            raise ValueError(f"Unknown object type {self.obj_type}")


@dataclass
class Robot:
    control_type: str
    act_info: (
        T.Dict
    )  # for joint controls, it contains keys: joint_names, joint_lb, joint_ub
    robot_id: int
    _robot_urdf: str  # path to the robot urdf

    # assuming fixed base robot (the base motion even if it exists,
    # is in the joints (such as prismatic motion for x-axis))
    pose: np.ndarray

    joint_pos: T.Dict  # joint_name: joint_pos
    joint_vel: T.Dict  # joint_name: joint_vel

    ee_link: T.Optional[str] = None  # when control type is ee_pose

    @property
    def robot_urdf(self):
        return str(get_robot_morphs_dir() / self._robot_urdf)

@dataclass
class Articulation:
    """
    articulation class is for assets other than the robot
    that can move in a controlled way.
    """

    act_info: T.Dict
    articulation_id: int
    urdf: str

    pose: np.ndarray

    joint_pos: T.Dict
    joint_vel: T.Dict

    cfg: T.Callable  # is a function that takes in color optionally and returns the cfg

    def __post_init__(self):
        # to make it compatible with the Robot class
        self.robot_urdf = self.urdf
        self.robot_id = self.articulation_id


@dataclass
class PrimState:
    pose: np.ndarray  # 4x4 pose matrix
    velocity: np.ndarray = np.zeros(3)  # not used for static objects
    ang_velocity: np.ndarray = np.zeros(3)  # not used for static objects


@dataclass
class RobotState:
    joint_pos: T.Optional[T.Dict] = None  # dict of joint_name: joint_pos
    joint_vel: T.Optional[T.Dict] = None  # dict of joint_name: joint_vel
    ee_pose: T.Optional[np.ndarray] = None  # 4x4 pose matrix when provided


@dataclass
class ArticulationState:
    joint_pos: T.Optional[T.Dict] = None  # dict of joint_name: joint_pos
    joint_vel: T.Optional[T.Dict] = None  # dict of joint_name: joint_vel


@dataclass
class ProblemSpec:
    robot_dict: T.Dict[int, Robot]
    obj_dict: T.Dict[int, Prim]

    init_robo_state_dict: T.Dict[int, RobotState]

    init_obj_state_dict: T.Dict[int, PrimState]
    goal_obj_state_dict: T.Dict[int, PrimState]

    goal_robo_state_dict: T.Dict[int, RobotState]
    # goal_robo_state_dict: Opt[T.Dict[int, RobotState]] = None

    articulation_dict: Opt[T.Dict[int, Articulation]] = None
    init_articulation_state_dict: Opt[T.Dict[int, ArticulationState]] = None
    goal_articulation_state_dict: Opt[T.Dict[int, ArticulationState]] = None

    additional_configs: Opt[str] = (
        None  # path to additional parameteric variations of the problem
    )
    ground: float = -0.1  # inf corresponds to no ground
    name: str = "problem"
    index: int = 0
    # metadata: Opt[T.Dict] = None
    # goal_robo_pose_dict: Opt[T.Dict[int, np.ndarray]] = None # the pose of ee-link

    def __post_init__(self):
        self.metadata = {}
        if self.articulation_dict is None:
            self.articulation_dict = {}
        if self.init_articulation_state_dict is None:
            self.init_articulation_state_dict = {}
        if self.goal_articulation_state_dict is None:
            self.goal_articulation_state_dict = {}


def get_empty_problemspec_dict():
    return {
        "init_robo_state_dict": {},
        "goal_robo_state_dict": {"joint_pos": {}, "ee_pose": {}},
        "init_obj_state_dict": {},
        "goal_obj_state_dict": {},
        "obstacles_state_dict": {},
    }


# def initialize_empty_problemspec_dict(robot_dict, obj_dict):
#     pdict = get_empty_problemspec_dict()
#     pdict['goal_robo_state_dict'] = {'joint_pos': {}, 'ee_pose': {}}
#     for robo_id in robot_dict:
#         pdict["init_robo_state_dict"][robo_id] = []
#         pdict["goal_robo_state_dict"]['joint_pos'][robo_id] = []
#         pdict["goal_robo_state_dict"]['ee_pose'][robo_id] = []

#     for obj_id in obj_dict:
#         pdict["init_obj_state_dict"][obj_id] = []
#         pdict["goal_obj_state_dict"][obj_id] = []

#     return pdict


def view_full_state(
    vis: VizServer,
    robo_dict: T.Dict[int, Robot],
    obj_dict: T.Dict[int, Prim],
    jpos_dict: T.Dict[int, T.Dict[str, float]],
    pose_dict: T.Dict[int, np.ndarray],
    obj_state: T.Dict[int, PrimState],
    ground_ht: float,
    art_dict={},
    art_state_dict={},
    prefix="scene",
    obj_color=0x1234FF,
    robot_color=0x0000FF,
):
    # view robots
    for robo_id, robo in robo_dict.items():
        jpos = jpos_dict[robo_id]
        robot_urdfpy = URDF.load(robo.robot_urdf)
        robo_pose = robo.pose.copy()
        robo_pose[2, 3] -= ground_ht
        vis.view_robot(
            robot_urdfpy,
            None,
            None,
            f"{prefix}/robot/{robo_id}",
            color=robot_color,
            joint_dict=jpos,
            base_pose=robo_pose,
            # alpha=1.0
        )

        if robo_id in pose_dict:
            print(f"viewing robot {robo_id} ee pose")
            ee_pose = pose_dict[robo_id]
            vis.view_frames(
                robo_pose @ ee_pose,
                name=f"{prefix}/robot/{robo_id}/ee",
            )

    # view objects
    for obj_id, obj in obj_dict.items():
        if (not obj.static) and (obj_id in obj_state):
            pose = obj_state[obj_id].pose
            pose[2, 3] -= ground_ht
            # obj.velocity = obj_state[obj_id].velocity
            # obj.ang_velocity = obj_state[obj_id].ang_velocity
            vis.view_frames(pose, name=f"{prefix}/obj/{obj_id}/frame")
        else:
            pose = obj.pose

        _color = 0x5C4033 if obj.static else obj_color
        vis.view_trimesh(f"{prefix}/obj/{obj_id}", obj.mesh, pose, color=_color)

    # view articulated objects
    if art_dict is not None:
        for art_id, art in art_dict.items():
            # get pose and jpos
            art.pose[2, 3] -= ground_ht
            base_pose = art.pose
            if art_id in art_state_dict:
                jpos = art_state_dict[art_id].joint_pos
            else:
                jpos = art.joint_pos
            art_urdfpy = URDF.load(art.urdf)
            vis.view_robot(
                art_urdfpy,
                None,
                None,
                f"{prefix}/art/{art_id}",
                color=obj_color,
                joint_dict=jpos,
                base_pose=base_pose,
            )


def get_jpos_and_pose_dict(robo_dict, robo_state_dict, metadata={}):
    jpos_dict = {}
    pose_dict = {}

    for robo_idx, robo in robo_dict.items():
        # check if jpos is available in goal_robo_state_dict
        if robo_state_dict[robo_idx].joint_pos is not None:
            jpos_dict[robo_idx] = robo_state_dict[robo_idx].joint_pos
        else:
            # check if metadata has the target jpos
            try:
                jpos_dict[robo_idx] = metadata["target_jpos"][robo_idx]

            except KeyError:
                print(
                    f"no target jpos found for robot {robo_idx}, using the default jpos"
                )
                jpos_dict[robo_idx] = robo.joint_pos

        # check if ee_pose is available in goal_robo_state_dict
        if robo_state_dict[robo_idx].ee_pose is not None:
            pose_dict[robo_idx] = robo_state_dict[robo_idx].ee_pose

    return jpos_dict, pose_dict


def view_problem_spec(prob: ProblemSpec, prob_prefix="problem", vis_server=None):
    robo_dict = prob.robot_dict
    obj_dict = prob.obj_dict

    init_robo_state_dict = prob.init_robo_state_dict
    goal_robo_state_dict = prob.goal_robo_state_dict

    init_obj_state_dict = prob.init_obj_state_dict
    goal_obj_state_dict = prob.goal_obj_state_dict

    if vis_server is None:
        vis = VizServer()
    else:
        vis = vis_server

    # view the initial and goal states
    init_jpos_dict, init_pose_dict = get_jpos_and_pose_dict(
        robo_dict, init_robo_state_dict
    )
    ground_ht = prob.ground if prob.ground > (-2) else -2.0

    view_full_state(
        vis,
        robo_dict,
        obj_dict,
        ground_ht=ground_ht,
        jpos_dict=init_jpos_dict,
        pose_dict=init_pose_dict,
        obj_state=init_obj_state_dict,
        art_dict=prob.articulation_dict,
        art_state_dict=prob.init_articulation_state_dict,
        robot_color=0x00FF00,
        obj_color=0x1234FF,
        prefix=prob_prefix + "_init",
    )

    goal_jpos_dict, goal_pose_dict = get_jpos_and_pose_dict(
        robo_dict, goal_robo_state_dict, prob.metadata
    )

    view_full_state(
        vis,
        robo_dict,
        obj_dict,
        ground_ht=ground_ht,
        jpos_dict=goal_jpos_dict,
        pose_dict=goal_pose_dict,
        obj_state=goal_obj_state_dict,
        art_dict=prob.articulation_dict,
        art_state_dict=prob.goal_articulation_state_dict,
        robot_color=0xFF0000,
        obj_color=0xFF1234,
        prefix=prob_prefix + "_goal",
    )

    # view ground
    # vis.view_ground(name="ground", zval=0.0)
    vis.view_ground(name="ground", zval=prob.ground)
    # if prob.ground > -100:
    #     vis.view_ground(name="ground", zval=ground_ht)


def get_ee_pose(robot, jpos_dict):
    robo_urdfpy = URDF.load(robot.robot_urdf)
    link_fk = robo_urdfpy.link_fk(cfg=jpos_dict)

    ee_pose = None
    for _l in link_fk:
        if _l.name == robot.ee_link:
            ee_pose = link_fk[_l]

    assert ee_pose is not None, "No end-effector found"

    return ee_pose


# def get_pose_env(v, es, seed=0):
#     prob: ProblemSpec = es[v](seed)

#     prob.metadata['target_jpos'] = {}
#     

#     for robo_idx in prob.robot_dict:
#         robot = prob.robot_dict[robo_idx]

#         # get ee pose for init state:
#         # TODO (Meenal): is it even needed?
#         init_joint_dict = prob.init_robo_state_dict[robo_idx].joint_pos
#         init_ee_pose = get_ee_pose(robot, init_joint_dict)

#         prob.init_robo_state_dict[robo_idx].ee_pose = init_ee_pose

#         # get ee pose for goal state and move jpos info to metadata
#         assert (prob.goal_robo_state_dict is not None), "No goal state found"
#         goal_joint_dict = prob.goal_robo_state_dict[robo_idx].joint_pos

#         goal_ee_pose = get_ee_pose(robot, goal_joint_dict)
#         prob.goal_robo_state_dict[robo_idx].ee_pose = goal_ee_pose

#         prob.metadata['target_jpos'][robo_idx] = deepcopy(goal_joint_dict)
#         prob.goal_robo_state_dict[robo_idx].joint_pos = None

#     return prob


def get_pose_env(env_fn, robo_type, robo_fname, seed, *args):
    prob: ProblemSpec = env_fn(robo_type, robo_fname, seed, *args)

    prob.metadata["target_jpos"] = {}

    for robo_idx, robo_state in prob.goal_robo_state_dict.items():
        goal_joint_dict = robo_state.joint_pos
        if goal_joint_dict is None:
            continue
        ee_pose = get_ee_pose(prob.robot_dict[robo_idx], goal_joint_dict)
        robo_state.ee_pose = ee_pose

        prob.metadata["target_jpos"][robo_idx] = deepcopy(goal_joint_dict)
        robo_state.joint_pos = None

        # also update the additional_configs if it exists
        if prob.additional_configs is not None:
            new_configs = convert_jpos_to_pose(
                prob.robot_dict[robo_idx], prob.additional_configs, *args
            )
            prob.additional_configs = new_configs

    return prob


def convert_jpos_to_pose(robot, additional_configs, save_dirname=None, save_fname=None):
    robo_id = robot.robot_id

    ad_cfgs_path = get_problem_spec_dir() / additional_configs
    # load the configs
    prob_configs = utils.load_pickle(ad_cfgs_path)

    
    # need to fill configs_dict['goal_robo_state_dict']['ee_pose'][robo_id]
    # need to empty configs_dict['goal_robo_state_dict']['joint_pos'][robo_id]

    prob_configs["goal_robo_state_dict"]["ee_pose"][robo_id] = []
    
    # print("Converting joint positions to end-effector poses")
    
    for jpos in tqdm(prob_configs["goal_robo_state_dict"]["joint_pos"][robo_id]):
        jdict = {k: v for k, v in zip(robot.act_info["joint_names"], jpos)}
        ee_pos = get_ee_pose(robot, jdict)
        prob_configs["goal_robo_state_dict"]["ee_pose"][robo_id].append(ee_pos)

    
    prob_configs["goal_robo_state_dict"]["joint_pos"].pop(robo_id)

    all_poses = prob_configs["goal_robo_state_dict"]["poses"][robo_id]

    if cfg.COMMAND.FULL_VISUALIZATION:
        new_link_pose_dict = all_poses
    else:
        # also remove the other links poses from the configs to save space
        new_link_pose_dict = {robot.ee_link: all_poses[robot.ee_link]}

    prob_configs["goal_robo_state_dict"]["poses"][robo_id] = new_link_pose_dict

    # save the new configs
    new_config_fname = additional_configs.replace(".pkl", "_pose.pkl")
    utils.save_pickle(prob_configs, get_problem_spec_dir() / new_config_fname)

    return new_config_fname


def check_collision(robot, joint_dict, obstacles, view=False):
    robo_urdfpy = URDF.load(robot.robot_urdf)
    return check_is_collision(
        robo_urdfpy, joint_dict, obstacles, view=view, base_pose=robot.pose
    )

    # link_fk = robo_urdfpy.link_fk(cfg=joint_dict)
    # link_fk = robo_urdfpy.collision_trimesh_fk(cfg=joint_dict)

    # collision_manager = trimesh.collision.CollisionManager()
    # if view:
    #     vis = VizServer()

    # for obs in obstacles.values():
    #     collision_manager.add_object(
    #         f"{obs.obj_id}", obs.mesh, transform=obs.pose
    #     )

    # meshes = []
    # poses = []

    # for idx, (mesh, pose) in enumerate(link_fk.items()):

    #     meshes.append(mesh)
    #     poses.append(pose)

    #     dist = collision_manager.min_distance_single(
    #         mesh, transform=pose
    #     )
    #     is_collision = (dist < 0.0)
    #     # is_collision = collision_manager.in_collision_single(
    #     #     link_mesh, transform=pose,
    #     # )
    #     if is_collision:
    #         # print(f"collision for link {l.name}, dist: {dist}, name: {name}")
    #         if view:
    #             for obs in obstacles.values():
    #                 vis.view_trimesh(f"obs_{obs.obj_id}", obs.mesh, obs.pose, 0x00FF00, opacity=0.5)

    #             for idx, (mesh, pose) in enumerate(zip(meshes, poses)):
    #                 vis.view_trimesh(f"link_{idx}", mesh, pose, 0xFFFFFF, opacity=0.5)

    #             input("Press enter to continue")

    #         return True
    #     else:
    #         collision_manager.add_object(
    #             f"{idx}", mesh, transform=pose
    #         )
    # return False


def add_robo_to_collision_manager(
    collision_manager, robot, jdict, prefix="robot", view=False, vis=None
):
    robo_urdfpy = URDF.load(robot.robot_urdf)
    link_fk = robo_urdfpy.collision_trimesh_fk(cfg=jdict)

    for idx, (m, p) in enumerate(link_fk.items()):
        collision_manager.add_object(
            f"{prefix}_{robot.robot_id}_{idx}", m, transform=robot.pose @ p
        )
        if view:
            assert vis is not None, "vis is None"
            vis.view_trimesh(
                f"{prefix}_{robot.robot_id}_{idx}",
                m,
                robot.pose @ p,
                0x0000FF,
                opacity=0.5,
            )

    return collision_manager


def add_to_collision_manager(
    collision_manager, obj_dict, view=False, vis=None, art_jdict={}
):
    if view:
        assert vis is not None

    for obj_id, obj in obj_dict.items():
        if isinstance(obj, Prim):
            collision_manager.add_object(obj_id, obj.mesh, transform=obj.pose)
            if view:
                assert vis is not None, "vis is None"
                vis.view_trimesh(
                    f"obj_{obj.obj_id}", obj.mesh, obj.pose, 0x0000FF, opacity=0.5
                )
        elif isinstance(obj, (Articulation, Robot)):
            if obj_id in art_jdict:
                obj_jdict = art_jdict[obj_id]
            else:
                obj_jdict = obj.joint_pos
            collision_manager = add_robo_to_collision_manager(
                collision_manager, obj, obj_jdict, prefix="art", view=view, vis=vis
            )

    return collision_manager


def get_parent_child_link_names(robo_urdfpy):
    parent_child_links = set()
    for j in robo_urdfpy.joints:
        # if j.joint_type == "fixed":
        parent_child_links.add((j.parent, j.child))

    return parent_child_links


def check_is_collision(
    robo_urdfpy: URDF,
    jdict,
    obj_dict,
    art_jdict={},
    view=False,
    base_pose=np.eye(4),
    collision_manager=None,
    collision_threshold=-0.02,
):
    if view:
        vis = VizServer()
    else:
        vis = None

    if collision_manager is None:
        collision_manager = trimesh.collision.CollisionManager()
        collision_manager = add_to_collision_manager(
            collision_manager, obj_dict, view=view, vis=vis, art_jdict=art_jdict
        )

    parent_child_links = get_parent_child_link_names(robo_urdfpy)
    link_fk = robo_urdfpy.link_fk(cfg=jdict)
    # link_fk = robo_urdfpy.collision_trimesh_fk(cfg=jdict)

    # get mesh for linkname 'link_0'

    min_distances = []
    names = []
    meshes = []
    poses = []

    # for fixed links, we don't check their collision with their parent link

    if "ground" in obj_dict:
        g_ht = obj_dict["ground"].pose[2, 3]
    else:
        g_ht = -10.0

    for idx, (link, pose) in enumerate(link_fk.items()):
        mesh = link.collision_mesh

        if mesh is None:
            continue

        meshes.append(mesh)
        poses.append(pose)

        dist, name = collision_manager.min_distance_single(
            mesh, transform=base_pose @ pose, return_name=True
        )

        m_ht = (base_pose @ pose)[2, 3]
        if m_ht < g_ht:
            is_collision = True

        else:
            # print(
            #     'WARNING, WARNING, WARNING:COLLISION THRESHOLD TOO HIGH'
            # )

            # print("Collision distance: ", dist)
            is_collision = dist < collision_threshold
            min_distances.append(dist)
            names.append(name)

        if is_collision:
            if (link.name, name) in parent_child_links:
                is_collision = False
            if (name, link.name) in parent_child_links:
                is_collision = False

        if name == "ground":
            if dist < 0.0:
                is_collision = True

        # is_collision = collision_manager.in_collision_single(
        #     link_mesh, transform=pose,
        # )

        if is_collision:
            # print(f"collision for link {idx}, dist: {dist}, name: {name}")

            # print(f"collision for link {l.name}, dist: {dist}, name: {name}")
            if view:
                assert vis is not None, "vis is None"
                for obs in obj_dict.values():
                    vis.view_trimesh(
                        f"obs_{obs.obj_id}", obs.mesh, obs.pose, 0x00FF00, opacity=0.5
                    )

                for idx, (mesh, pose) in enumerate(zip(meshes, poses)):
                    vis.view_trimesh(
                        f"{link.name}", mesh, base_pose @ pose, 0xFFFFFF, opacity=0.5
                    )

                input("Press enter to continue")
            return True
        else:
            collision_manager.add_object(link.name, mesh, transform=base_pose @ pose)

    # print(f"collision for link {l.name}")

    # vis.view_robot(
    #     robo_urdfpy, None, None, "obj_robot", 0xFF0000, mesh_type="collision", joint_dict=jdict
    # )
    # input("the above configuration is not in collision.")
    # print("min_distances: ", )
    # print("#" * 50)
    # print("Collision distances (minimmum): ", np.min(min_distances))
    # print("#" * 50)

    # print("names: ", names)
    return False


def get_non_colliding_jpos(
    robot,
    obj_dict,
    partial_dict=None,
    view=False,
    max_tries=100,
    return_dict=True,
):
    # joint_pos = {}

    joint_names = robot.act_info["joint_names"]
    joint_lb = robot.act_info["joint_lb"]
    joint_ub = robot.act_info["joint_ub"]

    robo_urdfpy = URDF.load(robot.robot_urdf)

    for _ in range(max_tries):
        # get robo mesh
        jpos = U(joint_lb, joint_ub)
        jpos[-2:] = -jpos[-4:-2]  # symmetric end effector joints
        jdict = dict(zip(joint_names, jpos))

        if partial_dict is not None:
            for k in partial_dict:
                jdict[k] = U(partial_dict[k][0], partial_dict[k][1])

        is_collision = check_is_collision(
            robo_urdfpy, jdict, obj_dict, view=view, base_pose=robot.pose
        )
        if not is_collision:
            if return_dict:
                return jdict
            else:
                return jpos

    return None


def get_non_colliding_jpos_general(
    robot,
    obj_dict,
    art_jdict={},
    partial_dict=None,
    view=False,
    max_tries=200,
    return_dict=True,
    robo_type="generated",
):
    # joint_pos = {}
    # view=True

    joint_names = robot.act_info["joint_names"]
    joint_lb = robot.act_info["joint_lb"]
    joint_ub = robot.act_info["joint_ub"]

    # new_joint_lb = (5 * np.array(joint_lb) + np.array(joint_ub)) / 6.0
    # new_joint_ub = (np.array(joint_lb) + 5 * np.array(joint_ub)) / 6.0
    new_joint_lb = np.array(joint_lb)
    new_joint_ub = np.array(joint_ub)

    robo_urdfpy = URDF.load(robot.robot_urdf)

    for _ in range(max_tries):
        # get robo mesh
        jpos = U(new_joint_lb, new_joint_ub)
        jdict = dict(zip(joint_names, jpos))

        if partial_dict is not None:
            for k in partial_dict:
                jdict[k] = U(partial_dict[k][0], partial_dict[k][1])

        is_collision = check_is_collision(
            robo_urdfpy,
            jdict,
            obj_dict,
            art_jdict=art_jdict,
            view=view,
            base_pose=robot.pose,
            collision_threshold=(-0.01) if robo_type == "real" else (-0.01),
        )
        if not is_collision:
            if return_dict:
                return jdict
            else:
                return jpos

    return None


def add_ground_prim(z_ground, obj_id):
    p_ground = np.eye(4)
    p_ground[2, 3] = z_ground - 0.005
    return Prim(
        obj_type="box",
        obj_id=obj_id,
        obj_shape=[2, 2, 0.01],
        pose=p_ground,
        static=True,
    )


def get_robot(robo_type, robo_fname, robo_id):
    if robo_type in ["real", "panda_variations"]:
        urdf_rel_path = os.path.join(
            robo_type, robo_fname, f"{robo_fname}_new.urdf"
        )
    else:
        urdf_rel_path = os.path.join(
            robo_type, robo_fname, f"{robo_fname}.urdf"
        )

    bot_urdf = get_robot_morphs_dir() / urdf_rel_path
    urdfpy_robot = URDF.load(bot_urdf)

    joint_names = [j.name for j in urdfpy_robot.joints if j.joint_type != "fixed"]
    jlimits = {j.name: j.limit for j in urdfpy_robot.joints}
    joint_lb = [jlimits[jname].lower for jname in joint_names]
    joint_ub = [jlimits[jname].upper for jname in joint_names]
    joint_types = {j.name: j.joint_type for j in urdfpy_robot.joints}
    joint_types = [joint_types[jname] for jname in joint_names]

    # override the joint limits for real robots
    if (robo_type == "real" and (robo_fname != "ur5_stick")) or (
        robo_type != "real"
    ):
        # get the robo config

        if robo_type == "real":
            margin = 0.3
        elif robo_type in ["arm_ed", "arm_ur5", "simple_bot"]:
            margin = 0.4
        elif robo_type == "cubes":
            margin = 0.4
        else:
            margin = 0.4

        robo_cfg = get_default_robo_cfg(robo_type, robo_fname, robo_id)
        # get the init state
        init_state = robo_cfg.init_state.joint_pos  # a dict with jnames: jvals
        # make the joint_lb and joint_ub close to the init state but within the limits
        joint_lb = []
        joint_ub = []
        for jname in joint_names:
            # do regex matching for jname in init_state
            # _jval = [v for k, v in init_state.items() if re.match(k, jname)]
            matching_keys = [k for k in init_state.keys() if re.match(k, jname)]
            if len(matching_keys) == 0:
                import pdb; pdb.set_trace()
                raise ValueError(f"No match found for {jname}. Available keys: {init_state.keys()}")
            elif len(matching_keys) > 1:
                # see if an exact match is found
                exact_match = [k for k in matching_keys if k == jname]
                if len(exact_match) == 1:
                    _jval = [v for k, v in init_state.items() if k == jname]
                else:
                    raise ValueError(f"Multiple matches found for {jname}. Matches: {matching_keys}")
                
            else:
                # only one match found, choose the value for that key
                _jval = [v for k, v in init_state.items() if k == matching_keys[0]]
            
            # _jval = [v for k, v in init_state.items() if k == jname]
            assert (
                len(_jval) == 1
            ), f"No unique match found for {jname}. Matches: {_jval}. Available keys: {init_state.keys()}"
            jval = _jval[0]

            # jval = init_state[jname]
            jlim = jlimits[jname]
            
            if jlim.upper is None:
                jlim_upper = 2 * np.pi
                jlim_lower = -2 * np.pi
            else:
                jlim_upper = jlim.upper
                jlim_lower = jlim.lower
                
            jlb = max(jval - margin * (jlim_upper - jlim_lower), jlim_lower)
            jub = min(jval + margin * (jlim_upper - jlim_lower), jlim_upper)
            joint_lb.append(jlb)
            joint_ub.append(jub)

    # print("Number of joints: ", len(joint_names))
    # print(f"Loading robot from {bot_urdf}. File: envs/reach/base.py")
    

    for idx, jtype in enumerate(joint_types):
        if jtype == "continuous":
            joint_lb[idx] = -2 * np.pi
            joint_ub[idx] = 2 * np.pi
            # joint_lb[idx] = -np.pi - np.pi / 2
            # joint_ub[idx] = np.pi + np.pi / 2

    default_jvals = (np.array(joint_lb) + np.array(joint_ub)) / 2.0

    robot = Robot(
        control_type="joint",
        act_info={
            "joint_names": joint_names,
            "joint_lb": joint_lb,
            "joint_ub": joint_ub,
        },
        robot_id=robo_id,
        _robot_urdf=str(urdf_rel_path),
        # ee_link="ee_link_base",
        pose=np.eye(4),
        joint_pos=dict(zip(joint_names, default_jvals)),
        joint_vel=dict(zip(joint_names, np.zeros(len(joint_names)))),
    )

    if robo_type == "stick":
        robot.ee_link = "link_ball"
    elif robo_type == "nlink":
        robot.ee_link = "link_" + str(len(joint_names))
    elif robo_type == "cubes":
        robot.ee_link = "link_0"
    elif robo_type == "prims":
        robot.ee_link = "link_0_end"
    elif robo_type == "simple_bot":
        robot.ee_link = "ee_link_base"
    elif robo_type == "arm_ed":
        robot.ee_link = "wrist"
    elif robo_type == "arm_ur5":
        robot.ee_link = "ee_link"
    elif robo_type == "chain":
        lnk_names = [lnk.name for lnk in urdfpy_robot.links]
        lnk_indices = [
            int(lnk_name.split("_")[1])
            for lnk_name in lnk_names
            if (lnk_name.startswith("link_") and lnk_name.endswith("_sphere"))
        ]
        robot.ee_link = f"link_{max(lnk_indices)}_sphere"
    elif robo_type == "mf_v1":
        robot.ee_link = "spherical_baselink"
    elif robo_type == "mf_v2":
        robot.ee_link = "cylindrical_baselink"
    elif robo_type == "planar_arm":
        if ("claw" in robo_fname) or ("c" in robo_fname):
            # n = int(robo_fname.split('_')[0])
            # robot.ee_link = f"link_{n-1}"
            robot.ee_link = "ee_link_right1"
        elif ("parallel" in robo_fname) or ("p" in robo_fname):
            robot.ee_link = "link_parallel"
        else:
            raise ValueError(f"Unknown robot type: {robo_type}, {robo_fname}")
    elif robo_type == "tongs_v1":
        robot.ee_link = "cylinder_baselink"
    elif robo_type == "tongs_v2":
        robot.ee_link = "cylinder_baselink"
    elif robo_type == "mf_arm":
        robot.ee_link = "cylindrical_baselink"
    elif robo_type == "real":
        robot.ee_link = get_ee_linkname(robo_fname)
    elif robo_type == "panda_variations":
        robot.ee_link = get_ee_linkname("panda")
    else:
        raise ValueError(f"Unknown robot type: {robo_type}, {robo_fname}")

    return robot


def get_ee_linkname(robo_fname):
    if robo_fname == "fetch":
        return "gripper_link"
    elif robo_fname == "jaco2":
        return "j2s7s300_end_effector"
    elif robo_fname == "xarm7":
        return "gripper_base_link"
    elif robo_fname == "yumi":
        return "gripper_r_base"
    elif robo_fname == "kinova_gen3":
        return "gripper_base_link"
    elif robo_fname == "lwr":
        return "panda_hand"
    elif robo_fname == "widowx":
        return "gripper_rail_link"
    elif robo_fname == "panda":
        return "panda_hand"
    elif robo_fname in ["ur5_sawyer", "ur5_planar", "ur5_stick", "ur5_ez"]:
        return "end_link"
    elif robo_fname.startswith("ur5"):
        assert False, "Ur5 besides sawyer, ez, planar and stick not supported"
        return "wrist_3_link"
    else:
        raise ValueError(f"Unknown robot type: {robo_fname}")


def get_fk_pos_and_ori(robot: Robot, joint_pos_list):
    # assiming joint_pos has same order as that of robot.act_info["joint_names"]

    positions = {}
    orientations = {}

    urdf_inst = URDF.load(robot.robot_urdf)
    # positions = []
    # orientations = []

    joint_names = robot.act_info["joint_names"]

    for jpos in joint_pos_list:
        jdict = dict(zip(joint_names, jpos))
        fk = urdf_inst.collision_trimesh_fk(cfg=jdict)

        for idx, (m, _p) in enumerate(fk.items()):
            p = robot.pose @ _p

            # robot base changes by ground_ht when loading into simulation
            # p = _p

            if f"link_{idx}" not in positions:
                positions[f"link_{idx}"] = []
                orientations[f"link_{idx}"] = []

            positions[f"link_{idx}"].append(p[:3, 3])
            orientations[f"link_{idx}"].append(p[:3, :3])

    # convert orientations to quat

    for l_name, rot in orientations.items():
        mat = torch.from_numpy(np.stack(rot))

        assert mat.shape == (len(joint_pos_list), 3, 3)

        orientations[l_name] = math_utils.quat_from_matrix(mat)
        assert orientations[l_name].shape == (len(joint_pos_list), 4)

    for l_name, pos in positions.items():
        positions[l_name] = torch.from_numpy(np.stack(pos))
        assert positions[l_name].shape == (len(joint_pos_list), 3)

    return positions, orientations


def get_fk_poses(robot: Robot, joint_pos_list):
    # assiming joint_pos has same order as that of robot.act_info["joint_names"]

    poses = {}
    urdf_inst = URDF.load(robot.robot_urdf)
    # positions = []
    # orientations = []

    joint_names = robot.act_info["joint_names"]

    for jpos in joint_pos_list:
        jdict = dict(zip(joint_names, jpos))
        fk = urdf_inst.link_fk(cfg=jdict)
        # fk = urdf_inst.collision_trimesh_fk(cfg=jdict)

        for idx, (l, _p) in enumerate(fk.items()):
            if l.collision_mesh is None and (l.name != robot.ee_link):
                continue

            # p = robot.pose @ _p

            # robot base changes by ground_ht when loading into simulation
            p = _p

            if l.name not in poses:
                poses[l.name] = []

            poses[l.name].append(p)

    # convert orientations to quat

    for l_name, pose in poses.items():
        poses[l_name] = torch.from_numpy(np.stack(pose))
        assert poses[l_name].shape == (len(joint_pos_list), 4, 4)

    return poses


def get_ee_poses(robot: Robot, ee_pose_list):
    poses = {robot.ee_link: torch.from_numpy(np.stack(ee_pose_list))}
    assert poses[robot.ee_link].shape == (len(ee_pose_list), 4, 4)
    return poses


def save_randomized_configs(
    robo_dict,
    get_cfg_fn,
    task_type,
    robo_type,
    robo_name,
    variation,
    seed,
    save_dirname=None,
    save_fname=None,
    n_cfgs=None,
) -> str:
    """
    if goal_robo_fn is None, then init_robo_fn is used,
    similarly for goal_obj_fn
    """

    if not n_cfgs:
        n_cfgs = cfg.NUM_GOAL_RANDOMIZATIONS

    config_dirname = save_dirname if save_dirname else f"{task_type}/{robo_type}/"
    configs_dir = get_problem_spec_dir() / config_dirname
    configs_dir.mkdir(parents=True, exist_ok=True)

    config_name = save_fname if save_fname else f"{robo_name}_{variation}_s{seed}.pkl"

    # if (
    #     not (configs_dir / config_name).exists()
    #     or cfg.RECOMPUTE_GOAL_RANDOMIZATION_CFGS
    # ):
    if True:
        configs_dict = get_empty_problemspec_dict()
        # configs_dict = initialize_empty_problemspec_dict(robo_dict, obj_dict)
        # configs_dict['init_robo_state_dict'] = {robo_id: [] for robo_id in robo_dict}

        # currently the initial and goal configurations are sampled independently
        # so they do not need to be aligned

        n_valid = 0
        # print(f"Generating {n_cfgs} configurations")
        for _ in tqdm(range(n_cfgs)):
            result = get_cfg_fn()

            if result is None:
                break

            n_valid += 1
            robo_init, robo_goal, obj_init, obj_goal, obstacles = result

            for robo_id in robo_init:
                if robo_id not in configs_dict["init_robo_state_dict"]:
                    configs_dict["init_robo_state_dict"][robo_id] = []

                configs_dict["init_robo_state_dict"][robo_id].append(robo_init[robo_id])

            # check if 'joint_pos'  and 'ee_pose' is present in robo_goal

            if "joint_pos" not in robo_goal:
                for robo_id in robo_goal:
                    if robo_id not in configs_dict["goal_robo_state_dict"]["joint_pos"]:
                        configs_dict["goal_robo_state_dict"]["joint_pos"][robo_id] = []

                    configs_dict["goal_robo_state_dict"]["joint_pos"][robo_id].append(
                        robo_goal[robo_id]
                    )

            else:
                # it means the config provides 'joint_pos' and 'ee_pose' separately
                for robo_id in robo_goal["joint_pos"]:
                    if robo_id not in configs_dict["goal_robo_state_dict"]["joint_pos"]:
                        configs_dict["goal_robo_state_dict"]["joint_pos"][robo_id] = []

                    configs_dict["goal_robo_state_dict"]["joint_pos"][robo_id].append(
                        robo_goal["joint_pos"][robo_id]
                    )

                for robo_id in robo_goal["ee_pose"]:
                    if robo_id not in configs_dict["goal_robo_state_dict"]["ee_pose"]:
                        configs_dict["goal_robo_state_dict"]["ee_pose"][robo_id] = []

                    configs_dict["goal_robo_state_dict"]["ee_pose"][robo_id].append(
                        robo_goal["ee_pose"][robo_id]
                    )

            for obj_id in obj_init:
                if obj_id not in configs_dict["init_obj_state_dict"]:
                    configs_dict["init_obj_state_dict"][obj_id] = []

                configs_dict["init_obj_state_dict"][obj_id].append(obj_init[obj_id])

            for obj_id in obj_goal:
                if obj_id not in configs_dict["goal_obj_state_dict"]:
                    configs_dict["goal_obj_state_dict"][obj_id] = []

                configs_dict["goal_obj_state_dict"][obj_id].append(obj_goal[obj_id])

            for obj_id in obstacles:
                if obj_id not in configs_dict["obstacles_state_dict"]:
                    configs_dict["obstacles_state_dict"][obj_id] = []

                configs_dict["obstacles_state_dict"][obj_id].append(obstacles[obj_id])


        if n_valid == 0:
            raise ValueError("No valid configurations found")

        # # check if any has zero length
        # if any([len(v) == 0 for v in configs_dict['init_robo_state_dict'].values()]):
        #     raise ValueError("No valid configurations found")
        # if any([len(v) == 0 for v in configs_dict['init_obj_state_dict'].values()]):
        #     raise ValueError("No valid configurations found")
        # if any([len(v) == 0 for v in configs_dict['goal_robo_state_dict']['joint_pos'].values()]):
        #     raise ValueError("No valid configurations found")
        # if any([len(v) == 0 for v in configs_dict['goal_obj_state_dict'].values()]):
        #     raise ValueError("No valid configurations found")

        # poses are used for creating goal markers for visualization purposes
        configs_dict["goal_robo_state_dict"]["poses"] = {}
        
        for robo_id, jpos in configs_dict["goal_robo_state_dict"]["joint_pos"].items():
            poses = get_fk_poses(robo_dict[robo_id], jpos)

            configs_dict["goal_robo_state_dict"]["poses"][robo_id] = poses

        for robo_id, ee_pose in configs_dict["goal_robo_state_dict"]["ee_pose"].items():
            # assert that poses[robo_id] is not already present
            assert robo_id not in configs_dict["goal_robo_state_dict"]["poses"]
            poses = get_ee_poses(robo_dict[robo_id], ee_pose)
            configs_dict["goal_robo_state_dict"]["poses"][robo_id] = poses

        utils.save_pickle(configs_dict, configs_dir / config_name)

    # return os.path.join(config_dirname, config_name)
    return str(config_dirname) + "/" + config_name
    # return (config_dirname) + "/" + config_name


def save_default_init_goal_configs(
    prob: ProblemSpec, save_dirname=None, save_fname=None
):
    def get_cfg_fn():
        robo_init_jpos = {}
        for robo_id, robo_state in prob.init_robo_state_dict.items():
            assert robo_state.joint_pos is not None
            jnames = prob.robot_dict[robo_id].act_info["joint_names"]
            jpos = []
            for jn in jnames:
                jval = robo_state.joint_pos[jn]
                jpos.append(jval)
            robo_init_jpos[robo_id] = jpos

        robo_goal_jpos = {"joint_pos": {}, "ee_pose": {}}
        for robo_id, robo_state in prob.goal_robo_state_dict.items():
            if robo_state.joint_pos is not None:
                jnames = prob.robot_dict[robo_id].act_info["joint_names"]
                jpos = []
                for jn in jnames:
                    jval = robo_state.joint_pos[jn]
                    jpos.append(jval)
                robo_goal_jpos["joint_pos"][robo_id] = jpos
            if robo_state.ee_pose is not None:
                robo_goal_jpos["ee_pose"][robo_id] = robo_state.ee_pose

        init_obj_state_dict = {}
        for obj_id, obj_state in prob.init_obj_state_dict.items():
            init_obj_state_dict[obj_id] = obj_state.pose

        goal_obj_state_dict = {}
        for obj_id, obj_state in prob.goal_obj_state_dict.items():
            goal_obj_state_dict[obj_id] = obj_state.pose
            
            
        obstacle_dict = {}
        for obj_id, obj in prob.obj_dict.items():
            if obj_id not in prob.goal_obj_state_dict:
                obstacle_dict[obj_id] = obj.pose

        # return robo_init_jpos, robo_init_jpos, prob.init_obj_state_dict, prob.init_obj_state_dict
        return robo_init_jpos, robo_goal_jpos, init_obj_state_dict, goal_obj_state_dict, obstacle_dict

    save_randomized_configs(
        prob.robot_dict,
        get_cfg_fn,
        task_type="None",
        robo_type="None",
        robo_name="None",
        variation="None",
        seed=0,
        save_dirname=save_dirname if save_dirname else "default",
        save_fname=save_fname if save_fname else "default.pkl",
        n_cfgs=1,
    )


def get_randomized_push_cfg(
    obstacle_dict,
    movable_obj_dict,
    robot_dict,
    obj_sampler,
    robot_sampler,
    z_ground,
    obj_sampler_args=[],
    robo_sampler_args=[],
):
    # obstacle dict will not change,
    # the shapes of objects will be fixed, but we will sample their poses
    collision_manager = trimesh.collision.CollisionManager()
    for obj_id, obj in obstacle_dict.items():
        collision_manager.add_object(f"obstacle_{obj_id}", obj.mesh, transform=obj.pose)

    block_dict = {}
    # add blocks that do not collide with obstacles
    for block_id, block in movable_obj_dict.items():
        for _ in range(100):
            p = obj_sampler(*obj_sampler_args)
            p[2, 3] = z_ground + block.obj_shape[2] / 2 + 0.001

            dist = collision_manager.min_distance_single(block.mesh, transform=p)

            is_collision = dist < 0.0
            if not is_collision:
                block_dict[block.obj_id] = p
                # block_dict[block.obj_id] = block

                collision_manager.add_object(
                    f"block_{block_id}", block.mesh, transform=p
                )
                break

    if len(block_dict) != len(movable_obj_dict):
        return None

    all_obs = {**obstacle_dict, **block_dict}

    found = False
    all_obs["ground"] = add_ground_prim(
        z_ground, obj_id=len(obstacle_dict) + len(block_dict) + 2
    )

    robo_jpos_dict = {}
    for robo_id, robot in robot_dict.items():
        joint_names = robot.act_info["joint_names"]
        for _ in range(100):
            init_jpos = robot_sampler(*robo_sampler_args)
            is_collision = check_collision(
                robot, dict(zip(joint_names, init_jpos)), all_obs, view=False
            )
            if not is_collision:
                found = True
                break

        if not found:
            return None

        robo_jpos_dict[robo_id] = init_jpos

    # robo init, robo goal, obj init, obj goal
    # for push task, we don't need goal states for robot
    return robo_jpos_dict, {}, block_dict, block_dict


def get_randomized_push_simple_cfg(
    obstacle_dict: T.Dict[int, Prim],
    movable_obj_dict: T.Dict[int, Prim],
    robot_dict: T.Dict[int, Robot],
    obj_sampler,
    robot_sampler,
    z_ground,
    obj_sampler_args=[],
    robo_sampler_args=[],
):
    # obstacle dict will not change,
    # the shapes of objects will be fixed, but we will sample their poses
    collision_manager = trimesh.collision.CollisionManager()
    for obj_id, obj in obstacle_dict.items():
        collision_manager.add_object(f"obstacle_{obj_id}", obj.mesh, transform=obj.pose)


    new_obstacle_dict = {}
    block_dict = {}
    
    tmp_block_dict = {f"block_{block_id}": block for block_id, block in movable_obj_dict.items()}
    tmp_block_dict.update({f"obstacle_{obj_id}": obj for obj_id, obj in obstacle_dict.items()})
    
    # add blocks that do not collide with obstacles
    # for block_id, block in movable_obj_dict.items():
    
    for block_name, block in tmp_block_dict.items():
        for _ in range(100):
            
            if "obstacle" in block_name:
                p = obj_sampler(*obj_sampler_args, flag="obs")                
            else:
                p = obj_sampler(*obj_sampler_args, flag="init")
                p[2, 3] = z_ground + block.obj_shape[2] / 2 + 0.001

            dist = collision_manager.min_distance_single(block.mesh, transform=p)

            is_collision = dist < 0.0
            if not is_collision:
                # block_dict[block.obj_id] = p
                
                if block_name.startswith("block"):
                    block_dict[block.obj_id] = p
                else:
                    new_obstacle_dict[block.obj_id] = p
                
                # block_dict[block.obj_id] = block

                collision_manager.add_object(
                    block_name, block.mesh, transform=p
                )
                break

    if len(block_dict) != len(movable_obj_dict):
        return None
    
    if len(new_obstacle_dict) != len(obstacle_dict):
        return None

    # all_obs = {**new_obstacle_dict, **block_dict}
    all_obs = {}
    for obj_id in new_obstacle_dict:
        obj = deepcopy(obstacle_dict[obj_id])
        obj.pose = new_obstacle_dict[obj_id]
        all_obs[obj_id] = obj
    for obj_id in block_dict:
        obj = deepcopy(movable_obj_dict[obj_id])
        obj.pose = block_dict[obj_id]
        all_obs[obj_id] = obj


    found = False
    all_obs["ground"] = add_ground_prim(
        z_ground, obj_id=len(obstacle_dict) + len(block_dict) + 2
    )

    robo_jpos_dict = {}
    for robo_id, robot in robot_dict.items():
        joint_names = robot.act_info["joint_names"]
        for _ in range(100):
            init_jpos = robot_sampler(*robo_sampler_args)
            is_collision = check_collision(
                robot, dict(zip(joint_names, init_jpos)), all_obs, view=False
            )
            if not is_collision:
                found = True
                break

        if not found:
            return None

        robo_jpos_dict[robo_id] = init_jpos

    # compute goal poses for the blocks
    collision_manager = trimesh.collision.CollisionManager()
    for obj_id, obj_pose in new_obstacle_dict.items():
        collision_manager.add_object(f"obstacle_{obj_id}", obstacle_dict[obj_id].mesh, transform=obj_pose)

    block_goal_dict = {}
    # add blocks that do not collide with obstacles
    for block_id, block in movable_obj_dict.items():
        for _ in range(100):
            p = obj_sampler(*obj_sampler_args, flag="goal")
            p[2, 3] = z_ground + block.obj_shape[2] / 2 + 0.001

            dist = collision_manager.min_distance_single(block.mesh, transform=p)

            is_collision = dist < 0.0
            if not is_collision:
                block_goal_dict[block.obj_id] = p
                # block_dict[block.obj_id] = block

                collision_manager.add_object(
                    f"block_{block_id}", block.mesh, transform=p
                )
                break

    if len(block_goal_dict) != len(movable_obj_dict):
        return None

    # robo init, robo goal, obj init, obj goal
    # for push task, we don't need goal states for robot
    return robo_jpos_dict, {}, block_dict, block_goal_dict, new_obstacle_dict


def get_randomized_reach_cfg(
    obstacle_dict,
    movable_obj_dict,
    robot_dict,
    obj_sampler,
    robot_sampler,
    z_ground,
    obj_sampler_args=[],
    robo_sampler_args=[],
):
    # obstacle dict will not change,
    # the shapes of objects will be fixed, but we will sample their poses
    collision_manager = trimesh.collision.CollisionManager()
    for obj_id, obj in obstacle_dict.items():
        collision_manager.add_object(f"obstacle_{obj_id}", obj.mesh, transform=obj.pose)

    block_dict = {}
    new_obstacle_dict = {}
    # add blocks that do not collide with obstacles
    
    tmp_all_obj_dict = {f"block_{block_id}": block for block_id, block in movable_obj_dict.items()}
    tmp_all_obj_dict.update({f"obstacle_{obj_id}": obj for obj_id, obj in obstacle_dict.items()})
    # tmp_all_obj_dict = {**obstacle_dict, **movable_obj_dict}
    
    # for block_id, block in movable_obj_dict.items():
    for block_name, block in tmp_all_obj_dict.items():
        for _ in range(100):
            p = obj_sampler(*obj_sampler_args, flag="init")
            # p[2, 3] = z_ground + block.obj_shape[2] / 2 + 0.001

            dist = collision_manager.min_distance_single(block.mesh, transform=p)

            is_collision = dist < 0.0
            if not is_collision:
                
                if block_name.startswith("block"):            
                    block_dict[block.obj_id] = p
                else:
                    new_obstacle_dict[block.obj_id] = p
                # block_dict[block.obj_id] = block

                collision_manager.add_object(
                    f"{block_name}", block.mesh, transform=p
                )
                break

    if len(block_dict) != len(movable_obj_dict):
        return None
    
    if len(new_obstacle_dict) != len(obstacle_dict):
        return None

    # all_obs = {**new_obstacle_dict, **block_dict}
    all_obs = {}
    for obj_id in new_obstacle_dict:
        obj = deepcopy(obstacle_dict[obj_id])
        obj.pose = new_obstacle_dict[obj_id]
        all_obs[obj_id] = obj
    for obj_id in block_dict:
        obj = deepcopy(movable_obj_dict[obj_id])
        obj.pose = block_dict[obj_id]
        all_obs[obj_id] = obj
    

    found = False
    all_obs["ground"] = add_ground_prim(
        z_ground, obj_id=len(obstacle_dict) + len(block_dict) + 2
    )

    robo_jpos_dict = {}
    for robo_id, robot in robot_dict.items():
        joint_names = robot.act_info["joint_names"]
        for _ in range(100):
            init_jpos = robot_sampler(*robo_sampler_args)
            is_collision = check_collision(
                robot, dict(zip(joint_names, init_jpos)), all_obs, view=False
            )
            if not is_collision:
                found = True
                break

        if not found:
            return None

        robo_jpos_dict[robo_id] = init_jpos

    # robo init, robo goal, obj init, obj goal
    # for push task, we don't need goal states for robot
    return robo_jpos_dict, robo_jpos_dict, block_dict, {}, new_obstacle_dict

def get_door_fnames():
    return [f"door_{i}" for i in range(10)]


def get_drawer_fnames():
    return [f"drawer_{i}" for i in range(10)]


def get_turner_fnames():
    return [f"turner_{i}" for i in range(5)]


def create_occupancy_grid(meshes, poses, workspace_bounds, voxel_size):
    """
    Creates a 3D occupancy grid from meshes and poses.

    Args:
        meshes: List of trimesh.Trimesh objects.
        poses: List of 4x4 transformation matrices (NumPy arrays).
        workspace_bounds: Tuple of (min_x, max_x, min_y, max_y, min_z, max_z).
        voxel_size: Size of each voxel in meters.

    Returns:
        A 3D NumPy array representing the occupancy grid.
    """

    min_x, max_x, min_y, max_y, min_z, max_z = workspace_bounds
    grid_dims = (
        int((max_x - min_x) / voxel_size),
        int((max_y - min_y) / voxel_size),
        int((max_z - min_z) / voxel_size),
    )

    occupancy_grid = np.zeros(grid_dims, dtype=np.uint8)

    for mesh, pose in zip(meshes, poses):
        transformed_mesh = mesh.copy()
        transformed_mesh.apply_transform(pose)

        # Trimesh voxelization is fast and accurate.
        voxels = transformed_mesh.voxelized(pitch=voxel_size)
        
        # Get the occupied voxel coordinates.
        occupied_indices = voxels.sparse_indices

        # Convert to grid coordinates.
        grid_x = ((occupied_indices[:, 0] * voxel_size) + transformed_mesh.bounds[0, 0] - min_x) / voxel_size
        grid_y = ((occupied_indices[:, 1] * voxel_size) + transformed_mesh.bounds[0, 1] - min_y) / voxel_size
        grid_z = ((occupied_indices[:, 2] * voxel_size) + transformed_mesh.bounds[0, 2] - min_z) / voxel_size

        grid_x = np.round(grid_x).astype(int)
        grid_y = np.round(grid_y).astype(int)
        grid_z = np.round(grid_z).astype(int)

        # Filter out of bounds voxels.
        valid_indices = np.where(
            (grid_x >= 0) & (grid_x < grid_dims[0]) &
            (grid_y >= 0) & (grid_y < grid_dims[1]) &
            (grid_z >= 0) & (grid_z < grid_dims[2])
        )[0]
        
        grid_x = grid_x[valid_indices]
        grid_y = grid_y[valid_indices]
        grid_z = grid_z[valid_indices]

        occupancy_grid[grid_x, grid_y, grid_z] = 1

    return occupancy_grid

    # # Example usage
    # meshes = [trimesh.load_mesh("mesh1.obj"), trimesh.load_mesh("mesh2.obj")]
    # poses = [
    #     matrix_from_euler([0, 0, 0], [0, 0, 0]),
    #     matrix_from_euler([0, np.pi / 2, 0], [1, 0, 0]),
    # ]  # Example poses
    # workspace_bounds = (-2, 2, -2, 2, -2, 2)
    # voxel_size = 0.1

    # occupancy_grid = create_occupancy_grid(meshes, poses, workspace_bounds, voxel_size)

    # #Print shape of grid.
    # print(occupancy_grid.shape)