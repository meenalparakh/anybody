import os
from collections import OrderedDict
from scipy.spatial.transform import Rotation as R

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, ArticulationCfg
from isaaclab.assets import RigidObject, Articulation
from isaaclab.utils.math import convert_quat, quat_from_matrix, wrap_to_pi
from anybody.utils.collision_utils import get_best_fit

# from isaaclab.scene import InteractiveSceneCfg
import anybody.envs.tasks.env_utils as eu
import urdfpy as ud
import numpy as np
import typing as T
from urdfpy import URDF
from pathlib import Path    

from anybody.cfg import cfg as global_cfg
from anybody.utils.utils import load_pickle, save_pickle

def get_init_state(obj: eu.Prim, prob: eu.ProblemSpec, ground_ht=0.0):
    obj_id = obj.obj_id
    obj_state = prob.init_obj_state_dict
    if (not obj.static) and (obj_id in obj_state):
        init_pose = obj_state[obj_id].pose
    else:
        init_pose = obj.pose

    init_pose[2, 3] -= ground_ht
    obj.pose = init_pose

    init_state_cfg = RigidObjectCfg.InitialStateCfg(
        pos=init_pose[:3, 3],
        rot=convert_quat(R.from_matrix(init_pose[:3, :3]).as_quat(), "wxyz"),
    )
    return init_state_cfg


def get_rigid_props(obj: eu.Prim):
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        max_depenetration_velocity=1.0, kinematic_enabled=obj.static
    )
    return rigid_props


def get_mass_props(obj: eu.Prim):
    m = obj.mass if obj.static else 0.01
    mass_props = sim_utils.MassPropertiesCfg(mass=m)
    return mass_props


def get_physics_material(obj):
    physics_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=2.0,
        dynamic_friction=2.0,
        restitution=0.0,
    )
    return physics_material


def get_color(obj):
    if isinstance(obj, eu.Prim):
        if obj.static:
            color = (0.37, 0.3, 0.255)
        else:
            color = (0.5, 0.0, 0.0)
    elif isinstance(obj, eu.Articulation):
        color = (0.1, 0.1, 0.7)
    else:
        raise NotImplementedError(f"Object type {type(obj)} not implemented")
    return color


def get_visual_material(obj):
    color = get_color(obj)
    visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=color)
    return visual_material


def get_collision_props(obj):
    collision_props = sim_utils.CollisionPropertiesCfg(
        collision_enabled=True,
        contact_offset=0.001,
        rest_offset=0.0,
    )
    return collision_props


def load_cylinder(obj: eu.Prim, prob: eu.ProblemSpec, ground_ht=0.0):
    obj_id = obj.obj_id
    assert obj.obj_type == "cylinder"

    obj_cfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cylinder_" + str(obj_id),
        spawn=sim_utils.CylinderCfg(
            radius=obj.obj_shape[0],
            height=obj.obj_shape[1],
            axis="Z",
            rigid_props=get_rigid_props(obj),
            mass_props=get_mass_props(obj),
            physics_material=get_physics_material(obj),
            visual_material=get_visual_material(obj),
            collision_props=get_collision_props(obj),
            activate_contact_sensors=True,
        ),
        init_state=get_init_state(obj, prob, ground_ht=ground_ht),
    )
    return obj_cfg


def load_box(obj: eu.Prim, prob: eu.ProblemSpec, ground_ht=0.0):
    obj_id = obj.obj_id
    assert obj.obj_type == "box"

    obj_cfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/box_" + str(obj_id),
        spawn=sim_utils.CuboidCfg(
            size=obj.obj_shape,
            rigid_props=get_rigid_props(obj),
            mass_props=get_mass_props(obj),
            physics_material=get_physics_material(obj),
            visual_material=get_visual_material(obj),
            collision_props=get_collision_props(obj),
            activate_contact_sensors=True,
        ),
        init_state=get_init_state(obj, prob, ground_ht=ground_ht),
    )
    return obj_cfg


def get_robo_cfg_dict(prob: eu.ProblemSpec, get_cfg_fn, ground_ht=0.0):
    # get robot configs
    robot_configs_dict = OrderedDict()

    jpos_dict, _ = eu.get_jpos_and_pose_dict(prob.robot_dict, prob.init_robo_state_dict)
    for robo_id, robo in prob.robot_dict.items():
        # get isaac sim config from robot
        urdf_path = robo.robot_urdf
        robo_name = os.path.dirname(urdf_path).split("/")[-1]
        robo_cfg: ArticulationCfg = get_cfg_fn(robo_name, robo_id=robo_id)
        # robo_cfg.spawn.activate_contact_sensors = True

        # load init joint state
        pose = robo.pose
        pos = pose[:3, 3]
        pos[2] -= ground_ht
        rot = convert_quat(R.from_matrix(pose[:3, :3]).as_quat(), "wxyz")
        # rot = R.from_matrix(pose[:3, :3]).as_quat(scalar_first=True)
        robo_cfg.init_state.pos = tuple(pos)
        robo_cfg.init_state.rot = tuple(rot)

        _jp = jpos_dict[robo_id]
        _jp = {k: torch.tensor(v, dtype=torch.float32) for k, v in _jp.items()}
        # print("initial joint poses", _jp)
        robo_cfg.init_state.joint_pos = _jp
        robot_configs_dict[f"robot_{robo_id}"] = robo_cfg

    return robot_configs_dict


def get_obst_cfg_dict(prob: eu.ProblemSpec, ground_ht=0.0):
    # load obstacles
    obstacle_configs_dict = OrderedDict()
    for obj_id, obj in prob.obj_dict.items():
        if obj.obj_type == "box":
            obj_cfg = load_box(obj, prob, ground_ht=ground_ht)
        elif obj.obj_type == "cylinder":
            obj_cfg = load_cylinder(obj, prob, ground_ht=ground_ht)

        else:
            raise NotImplementedError(f"Object type {obj.obj_type} not implemented")

        obstacle_configs_dict[f"obstacle_{obj_id}"] = obj_cfg

    return obstacle_configs_dict


def get_articulations_cfg_dict(prob: eu.ProblemSpec, ground_ht=0.0):
    # load articulations
    articulation_configs_dict = OrderedDict()
    assert prob.articulation_dict is not None

    art_jdict = prob.init_articulation_state_dict

    for art_id, art in prob.articulation_dict.items():
        # get isaac sim config from articulation
        assert isinstance(art, eu.Articulation)
        color = get_color(art)

        cfg: ArticulationCfg = art.cfg(color)

        # get initial pose
        pose = art.pose
        pos = pose[:3, 3]
        pos[2] -= ground_ht
        rot = convert_quat(R.from_matrix(pose[:3, :3]).as_quat(), "wxyz")
        cfg.init_state.pos = tuple(pos)
        cfg.init_state.rot = tuple(rot)

        # get initial joint state
        if art_id in art_jdict:
            _jp = art_jdict[art_id].joint_pos
        else:
            _jp = art.joint_pos

        _jp = {k: torch.tensor(v, dtype=torch.float32) for k, v in _jp.items()}
        cfg.init_state.joint_pos = _jp
        # cfg

        articulation_configs_dict[f"articulation_{art_id}"] = cfg

    return articulation_configs_dict


def get_obj_geometry_vector(obj_type: str, shape_params: dict):
    # for box, shape params is 'dims': [x, y, z]
    # for cylinder, shape params is 'radius': r, 'height': h
    # for sphere, shape params is 'radius': r

    # representation vectors
    # sphere: [1, 0, 0], [r, r, r]
    # cylinder: [0, 1, 0], [r, r, r]
    # box: [0, 0, 1], [x, y, z]

    if obj_type == "box":
        vec = torch.tensor([0, 0, 1, *shape_params["dims"]])
    elif obj_type == "sphere":
        r = shape_params["radius"]
        vec = torch.tensor([1, 0, 0, r, r, r])
    elif obj_type == "cylinder":
        r, h = shape_params["radius"], shape_params["height"]
        vec = torch.tensor([0, 1, 0, r, r, h])
    elif obj_type == "null":
        vec = torch.tensor([0, 0, 0, 0, 0, 0])
    else:
        raise ValueError(f"Unknown object type: {obj_type}")

    return vec


def get_mesh_info(meshes_list, origin_list, link_idx):
    # find combined mesh
    combined_mesh = meshes_list[0].apply_transform(origin_list[0])
    for idx, m in enumerate(meshes_list[1:], start=1):
        combined_mesh = combined_mesh + m.apply_transform(origin_list[idx])

    shape, shape_pose, shape_dims = get_best_fit(combined_mesh)
    if shape == "box":
        info = ("box", {"dims": shape_dims}, shape_pose, link_idx)
    else:
        assert shape == "cylinder", "Only box and cylinder shapes are supported"
        info = (
            "cylinder",
            {"radius": shape_dims[0], "height": shape_dims[1]},
            shape_pose,
            link_idx,
        )

    return info
    

def get_robo_link_info(robo_urdfpy: ud.URDF):
    # info is a tuple of (obj_type, shape_params, origin)
    link_info = {}
    for idx, link in enumerate(robo_urdfpy.links):
        
        # if idx == 1:
            
            
        link: ud.Link = link
        if link.visuals is None or (len(link.visuals) == 0):
            # return the default
            info = ("null", {}, np.eye(4), idx)

        elif len(link.visuals) > 1:
            # combine the meshes into 1 trimesh object
            
            all_meshes = [link.collision_mesh]
            origin_list = [np.eye(4)]
            
            info = get_mesh_info(all_meshes, origin_list, idx)
            
            # all_meshes = []
            # origin_list = []
            # for v in link.visuals:
            #     assert (
            #         v.geometry.mesh is not None
            #     ), "Only mesh geometry is supported for multi-visual elements"
            #     all_meshes.extend(v.geometry.mesh.meshes)
            #     origin_list.extend([torch.tensor(v.origin)] * len(v.geometry.mesh.meshes))

            # info = get_mesh_info(all_meshes, origin_list, idx)

        elif len(link.visuals) == 1:
            link_geom = link.visuals[0].geometry
            origin = link.visuals[0].origin
            origin = torch.tensor(origin)

            if link_geom.box is not None:
                info = ("box", {"dims": link_geom.box.size}, origin, idx)
            elif link_geom.sphere is not None:
                info = ("sphere", {"radius": link_geom.sphere.radius}, origin, idx)
            elif link_geom.cylinder is not None:
                info = (
                    "cylinder",
                    {
                        "radius": link_geom.cylinder.radius,
                        "height": link_geom.cylinder.length,
                    },
                    origin,
                    idx,
                )

            elif link_geom.mesh is not None:
                
                info = get_mesh_info([link.collision_mesh], [np.eye(4)], idx)
                
                # # find the best fit box, or cylinder
                # meshes = link_geom.mesh.meshes
                # origin_list = [origin] * len(meshes)
                # # assert len(meshes) == 1, "Only one mesh is supported"
                # info = get_mesh_info(meshes, origin_list, idx)

            else:
                raise ValueError(f"{link.name}: Unknown geometry type")
                print(
                    "WARNING WARNING WARNING: More than one visual element for link:",
                    link.name,
                )
                # setting some default values
                info = (
                    "cylinder",
                    {
                        "radius": 0.01,
                        "height": 0.01,
                    },
                    origin,
                    idx,
                )

        else:
            import pdb; pdb.set_trace()
            raise ValueError("Unknown geometry type")

        link_info[link.name] = dict(
            zip(["obj_type", "shape_params", "origin", "idx"], info)
        )

    return link_info


def get_robo_joint_cfg(robo_urdfpy: ud.URDF, is_torch=True, return_child_dict=False):
    joint_info = {}
    joint_child_map = {}
    for joint in robo_urdfpy.joints:
        if joint.joint_type == "fixed":
            joint_limits = np.array([0, 0])
        elif joint.joint_type == "continuous":
            joint_limits = np.array([-2 * np.pi, 2 * np.pi])
        else:
            joint_limits = np.array([joint.limit.lower, joint.limit.upper])

        joint: ud.Joint = joint
        joint_key = joint.name
        joint_info[joint_key] = {
            "child": joint.child,
            "parent": joint.parent,
            "joint_type": joint.joint_type,
            "axis": joint.axis,
            "origin": joint.origin,
            "limits": joint_limits,
        }
        joint_child_map[joint.child] = joint.name
        if is_torch:
            joint_info[joint_key]["axis"] = torch.tensor(joint.axis)
            joint_info[joint_key]["origin"] = torch.tensor(joint.origin)
            joint_info[joint_key]["limits"] = torch.tensor(
                joint_info[joint_key]["limits"]
            )

    if return_child_dict:
        return joint_info, joint_child_map

    else:
        return joint_info


def binary(x, n):
    """
    Source: https://stackoverflow.com/questions/55918468/convert-integer-to-pytorch-tensor-of-binary-bits
    """
    mask = 2 ** torch.arange(n).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()


def pose_encode(pose, to="pos_quat"):
    if isinstance(pose, np.ndarray):
        pose = torch.from_numpy(pose)

    if to == "pos_quat":
        pos = pose[:3, 3]
        quat = quat_from_matrix(pose[:3, :3])
        return torch.cat([pos, quat])
    else:
        raise ValueError(f"Unknown pose encoding: {to}")


def get_robo_link_vec(
    links_info,
    joints_info,
    child_dict,
    goal_state: T.Optional[eu.RobotState] = None,
    ee_link_name=None,
    link_idx_encode_dim=4,
) -> T.Tuple[dict, dict]:
    link_vec_dict = {}

    

    for l_name, l_info in links_info.items():
        if l_name not in child_dict:
            assert l_info['idx'] == 0, "the zeroth link should not have a parent, expected to be the base link"
            # print(f"Link {l_name} is not a child of any joint")
            continue

        j_info = joints_info[child_dict[l_name]]
        l_idx = l_info["idx"]
        p_idx = links_info[j_info["parent"]]["idx"]
        # idx_vec = binary(torch.tensor([l_idx, p_idx]), link_idx_encode_dim).reshape(-1)

        # idx_vec = torch.tensor([l_idx, p_idx])
        idx_vec = torch.tensor([l_idx, p_idx])

        print(f"Link {l_name}: {l_idx}, Parent {j_info['parent']}: {p_idx}")

        geom_vec = get_obj_geometry_vector(l_info["obj_type"], l_info["shape_params"])
        l_origin_vec = pose_encode(l_info["origin"])

        j_tvec = torch.zeros(3)
        j_type = j_info["joint_type"]
        if j_type == "revolute":
            j_tvec[0] = 1
        elif j_type == "prismatic":
            j_tvec[1] = 1
        elif j_type == "fixed":
            j_tvec[2] = 1

        # j_limits = j_info["limits"]  # shape (2,)
        j_axis = j_info["axis"]  # shape (3,)
        j_origin = pose_encode(j_info["origin"])  # shape (7,)

        link_vec = torch.cat([idx_vec, geom_vec, l_origin_vec])
        joint_vec = torch.cat([j_tvec, j_axis, j_origin])

        goal_flag = torch.zeros(2)
        if (goal_state is None) or (
            goal_state.joint_pos is None and goal_state.ee_pose is None
        ):
            goal_jval = torch.zeros(1)
            goal_jpose = torch.zeros(7)

        else:
            if goal_state.joint_pos is not None:
                if child_dict[l_name] in goal_state.joint_pos:
                    # means the joint is a movable joint
                    goal_jval = torch.tensor([goal_state.joint_pos[child_dict[l_name]]])
                    goal_flag[0] = 1
                else:
                    goal_jval = torch.zeros(1)
            else:
                goal_jval = torch.zeros(1)
                    
            if (ee_link_name == l_name) and (goal_state.ee_pose is not None):
                goal_jpose = pose_encode(goal_state.ee_pose)
                goal_flag[1] = 1
            else:
                goal_jpose = torch.zeros(7)

        # need to upweigh the jpos
        # goal_jval = goal_jval.expand(7)
        # encode the goal vector

        # freqs = ifunc.get_sinusoid_frequencies(
        #     num_freqs = n_joint_val_enc_dim // 2,
        #     min_wavelength=cfg.OBSERVATION.JOINT_VALUE_ENCODER.MIN_WAVELENGTH,
        #     max_wavelength=cfg.OBSERVATION.JOINT_VALUE_ENCODER.MAX_WAVELENGTH,
        # )

        if global_cfg.OBSERVATION.MINIMAL:
            assert goal_state.joint_pos is None, "Minimal observation does not support joint goals"
            goal_vec = torch.cat([goal_flag[:1], goal_jpose])

        else:
            goal_vec = torch.cat([goal_flag[:1], goal_jval, goal_flag[1:], goal_jpose])
        
        link_vec_dict[l_idx] = (link_vec, joint_vec, goal_vec, l_name, child_dict[l_name])
        # joint_link_idx_dict[child_dict[l_name]] = l_idx    # maps joint name to link idx
        # joint_link_idx_dict[l_idx] = child_dict[l_name]  # maps link idx to joint name

    return link_vec_dict

    # assert geom_vec.shape == (6,)
    # assert idx_vec.shape == (2 * link_idx_encode_dim,)
    # assert l_origin_vec.shape == (7,)


# def get_joint_vector(joint_info):
#     pos = joint_info["origin"][:3, 3]
#     quat = quat_from_matrix(joint_info["origin"][:3, :3])
#     return torch.cat([pos, quat, joint_info["limits"]])

def get_obj_shape_vec(obj: eu.Prim):
    # the vector includes: the shape information, and the goal information
    if obj.obj_type == "box":
        shape_vec = torch.tensor([0, 0, 1, *obj.obj_shape])

    elif obj.obj_type == "cylinder":
        r, h = obj.obj_shape
        shape_vec = torch.tensor([0, 1, 0, r, r, h])

    elif obj.obj_type == "sphere":
        r = obj.obj_shape[0]
        shape_vec = torch.tensor([1, 0, 0, r, r, r])

    else:
        raise ValueError(f"Unknown object type: {obj.obj_type}")

    return shape_vec

def get_obj_vector(obj: eu.Prim, obj_goal: eu.PrimState | None):
    # the vector includes: the shape information, and the goal information
    if obj.obj_type == "box":
        shape_vec = torch.tensor([0, 0, 1, *obj.obj_shape])

    elif obj.obj_type == "cylinder":
        r, h = obj.obj_shape
        shape_vec = torch.tensor([0, 1, 0, r, r, h])

    elif obj.obj_type == "sphere":
        r = obj.obj_shape[0]
        shape_vec = torch.tensor([1, 0, 0, r, r, r])

    else:
        raise ValueError(f"Unknown object type: {obj.obj_type}")

    if obj_goal is not None:
        pose = torch.from_numpy(obj_goal.pose)
        goal_vec = torch.cat([torch.ones(1), pose_encode(pose)])
    else:
        goal_vec = torch.zeros(8)
        
    # shape: (6,)
    # goal: (8,)
    # pose: (7,)
    # flag: (2,)
    # total: 6 + 8 + 7 + 2 = 23

    return (shape_vec, goal_vec)



def precompute_robo_info(prob: eu.ProblemSpec):
    robo_joint_link_idx_dict = {}

    for robo_id, robo in prob.robot_dict.items():
        robo_urdfpy = URDF.load(robo.robot_urdf)

        robo_info_pickle_fname = robo.robot_urdf.replace(".urdf", "_info.pkl")
        
        if not Path(robo_info_pickle_fname).exists():
            robo_urdfpy = URDF.load(robo.robot_urdf)
            links_info = get_robo_link_info(robo_urdfpy)
            joints_info, child_dict = get_robo_joint_cfg(
                robo_urdfpy, return_child_dict=True
            )
            save_pickle((links_info, joints_info, child_dict), robo_info_pickle_fname)
        else:
            links_info, joints_info, child_dict = load_pickle(robo_info_pickle_fname)


        joint_link_idx_dict = get_robo_link_vec(
            links_info=links_info,
            joints_info=joints_info,
            child_dict=child_dict,
            goal_state=prob.goal_robo_state_dict.get(robo_id, None),
            ee_link_name=robo.ee_link,
            link_idx_encode_dim=global_cfg.OBSERVATION.ROBO_LINK_IDX_ENCODE_DIM,
        )

        robo_joint_link_idx_dict[robo_id] = joint_link_idx_dict

    return robo_joint_link_idx_dict


def map_trajectory_to_actions(prob: eu.ProblemSpec, robo_info_dict, traj, step):
    robo_indices = list(prob.robot_dict.keys())
    robo_indices.sort()

    action_values = torch.zeros(global_cfg.BENCH.MAX_NUM_ROBOTS, global_cfg.BENCH.MAX_NUM_LINKS + 1)
    obj_action_values = torch.zeros(global_cfg.BENCH.MAX_NUM_OBJECTS)

    for r_idx, robo_id in enumerate(robo_indices):
        info = robo_info_dict[robo_id]
        link_indices = list(info.keys())
        link_indices.sort()

        robo_joint_names = prob.robot_dict[robo_id].act_info["joint_names"]
        robo_lb = prob.robot_dict[robo_id].act_info["joint_lb"]
        robo_ub = prob.robot_dict[robo_id].act_info["joint_ub"]

        for il, link_idx in enumerate(link_indices):
            jname = info[link_idx]
            if jname in robo_joint_names:
                lb = robo_lb[robo_joint_names.index(jname)]
                ub = robo_ub[robo_joint_names.index(jname)]
                # get the joint value from the trajectory
                jval = traj[robo_id][jname][step]
                # normalize the joint value to -1, 1
                jval = (jval - lb) / (ub - lb) * 2.0 - 1.0
                action_values[r_idx, il + 1] = jval

    return torch.cat([action_values.reshape(-1), obj_action_values])


def map_trajectory_to_actions_batched(prob: eu.ProblemSpec, robo_info_dict, traj_batch, step, n_envs, obs=None):
    robo_indices = list(prob.robot_dict.keys())
    robo_indices.sort()

    # action_values = torch.zeros(n_envs, global_cfg.BENCH.MAX_NUM_ROBOTS, global_cfg.BENCH.MAX_NUM_LINKS + 1)
    # obj_action_values = torch.zeros(n_envs, global_cfg.BENCH.MAX_NUM_OBJECTS)

    action_values = torch.zeros(n_envs, global_cfg.BENCH.MAX_NUM_LINKS)
    action_mask = torch.zeros(n_envs, global_cfg.BENCH.MAX_NUM_LINKS)

    for r_idx, robo_id in enumerate(robo_indices):
        info = robo_info_dict[robo_id]
        link_indices = list(info.keys())
        link_indices.sort()

        robo_joint_names = prob.robot_dict[robo_id].act_info["joint_names"]
        robo_lb = prob.robot_dict[robo_id].act_info["joint_lb"]
        robo_ub = prob.robot_dict[robo_id].act_info["joint_ub"]

        for il, link_idx in enumerate(link_indices):
            jname = info[link_idx][-1]
            if jname in robo_joint_names:
                lb = robo_lb[robo_joint_names.index(jname)]
                ub = robo_ub[robo_joint_names.index(jname)]
                # get the joint value from the trajectory
                jval = traj_batch[robo_id][jname][:, step]
                # jval has n_envs elements
                # normalize the joint value to -1, 1
                
                if global_cfg.ACTION.ABSOLUTE:
                    jval = (jval - lb) / (ub - lb) * 2.0 - 1.0
                
                action_values[:, il] = jval
                action_mask[:, il] = 1

    # if obs is provided, make sure the action mask matches
    if obs is not None:
        given_act_mask = obs['act_mask'].bool()
        action_mask = action_mask.bool().to(given_act_mask.device)
        assert given_act_mask[:, :, 0].shape == action_mask.shape, f"Action mask shape mismatch {given_act_mask[:, :, 0].shape} != {action_mask.shape}"
        assert torch.all(given_act_mask[:, :, 0] == action_mask), f"Action mask mismatch {given_act_mask[:, :, 0]} != {action_mask}"

    # the action values are absolute here. Need to make sure that the cfg has absolute actions set
    # return torch.cat([action_values.reshape(n_envs, -1), obj_action_values], dim=1)
    return action_values

    