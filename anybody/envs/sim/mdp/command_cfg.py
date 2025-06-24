from typing import Sequence
import torch
import numpy as np
import random
import string
import trimesh
from pathlib import Path
from isaaclab.utils import configclass
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.managers import CommandTerm
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers

from isaaclab.envs import ManagerBasedRLEnv
import anybody.envs.tasks.env_utils as eu
from anybody.utils.utils import load_pickle, save_pickle, get_posquat
from anybody.utils.path_utils import get_problem_spec_dir, get_robot_morphs_dir
from anybody.cfg import cfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from anybody.utils.to_usd import mesh_to_usd_args, MeshArgsCli
from isaaclab.utils.math import quat_from_matrix, matrix_from_quat

from anybody.utils.vis_server import VizServer

from urdfpy import URDF

from typing import TYPE_CHECKING
from scipy.spatial.transform import Rotation as R


class GoalCommand(CommandTerm):
    def __init__(self, command_cfg: CommandTermCfg, env: ManagerBasedRLEnv):
        super().__init__(command_cfg, env)

        print(f"Debug visualization: {command_cfg.debug_vis} at command_cfg.py, goalCommand")

        # create buffers to store the commands

        # load the goal configs
        self.goal_configs = load_pickle(
            get_problem_spec_dir() / command_cfg.allowed_goal_configs_path
        )

        self.robo_buffer = {}
        self.obj_buffer = {}

        self.n_goals = 0

        # goal robo state dict
        for robo_id, robo_state in self.goal_configs["goal_robo_state_dict"][
            "joint_pos"
        ].items():
            self.robo_buffer[robo_id] = {
                "joint_pos": torch.zeros(
                    self.num_envs, len(robo_state[0]), device=self.device
                )
            }
            self.n_goals = len(robo_state)

        # goal robo state pose dict
        for robo_id in self.goal_configs["goal_robo_state_dict"]["ee_pose"]:
            if robo_id in self.robo_buffer:
                self.robo_buffer[robo_id]["ee_pose"] = torch.zeros(
                    self.num_envs, 7, device=self.device
                )
            else:
                self.robo_buffer[robo_id] = {
                    "ee_pose": torch.zeros(self.num_envs, 7, device=self.device)
                }
            self.n_goals = len(
                self.goal_configs["goal_robo_state_dict"]["ee_pose"][robo_id]
            )

        # goal obj state dict
        for obj_id, obj_state in self.goal_configs["goal_obj_state_dict"].items():
            self.obj_buffer[obj_id] = torch.zeros(self.num_envs, 7, device=self.device)
            self.n_goals = len(obj_state)

        print(f"Number of goal configurations: {self.n_goals}")
        self.goal_indices = np.zeros(
            self.num_envs, dtype=np.int64
        )

        # the robot and object assets
        self.robot = {}
        self.obj = {}

        for robo_id in self.robo_buffer:
            self.robot[robo_id] = env.scene[f"robot_{robo_id}"]

        for obj_id in self.obj_buffer:
            self.obj[obj_id] = env.scene[f"obstacle_{obj_id}"]


        # converting obj goal configs from 4x4 to pos+quat
        for obj_id in self.obj_buffer:
            poses = self.goal_configs["goal_obj_state_dict"][obj_id]
            
            
            poses = torch.from_numpy(np.stack(poses)).float()
            positions = poses[:, :3, 3]
            
            # positions[:, 2] = self.cfg.obj_base_ht_dict[obj_id]
            # adjust the target goal positions, similar to the initial positions
            # which were adjusted so that the ground is at z=0 plane.
            
            positions[:, 2] -= self.cfg.ground_ht
            rotations = poses[:, :3, :3]
            quats = quat_from_matrix(rotations)
            pos_quats = torch.cat([positions, quats], dim=1)
            
            self.goal_configs["goal_obj_state_dict"][obj_id] = pos_quats #.detach().cpu().numpy()

        # converting robo goal configs from 4x4 to pos+quat
        for robo_id in self.robo_buffer:
            if 'ee_pose' in self.robo_buffer[robo_id]:
                robo_base_pose = torch.from_numpy(self.cfg.robo_base_pose_dict[str(robo_id)]).float()
                
                poses = self.goal_configs["goal_robo_state_dict"]["ee_pose"][robo_id]
                poses = torch.from_numpy(np.stack(poses)).float()
                
                 
                poses = robo_base_pose @ poses
                
                positions = poses[:, :3, 3]
                
                rotations = poses[:, :3, :3]
                quats = quat_from_matrix(rotations)
                pos_quats = torch.cat([positions, quats], dim=1)
                self.goal_configs["goal_robo_state_dict"]["ee_pose"][robo_id] = pos_quats #.detach().cpu().numpy()


        if cfg.COMMAND.DEBUG_VIS:
            
            self.goal_configs["goal_robo_state_dict"]["positions"] = {}
            self.goal_configs["goal_robo_state_dict"]["orientations"] = {}


            for robo_id in self.robo_buffer:
                
                if robo_id not in self.goal_configs["goal_robo_state_dict"]["poses"]:
                    continue
                
                all_link_poses = self.goal_configs["goal_robo_state_dict"]["poses"][robo_id]
                robo_base_pose = torch.from_numpy(self.cfg.robo_base_pose_dict[str(robo_id)])

                self.goal_configs["goal_robo_state_dict"]["positions"][robo_id] = {}
                self.goal_configs["goal_robo_state_dict"]["orientations"][robo_id] = {}

                for lname, poses in all_link_poses.items():
                    
                    fixed_transform = np.eye(4)
                    fixed_transform[:3, :3] = R.from_euler("xyz", [-np.pi/2, 0, 0]).as_matrix()
                    fixed_transform = torch.from_numpy(fixed_transform)
                    
                    
                    poses = robo_base_pose @ poses @ fixed_transform
                    assert poses.shape == (self.n_goals, 4, 4)

                    self.goal_configs["goal_robo_state_dict"]["positions"][robo_id][
                        lname
                    ] = poses[:, :3, 3].detach().cpu().numpy()
                    self.goal_configs["goal_robo_state_dict"]["orientations"][robo_id][
                        lname
                    ] = quat_from_matrix(poses[:, :3, :3]).detach().cpu().numpy()

            # for obj_id in self.obj_buffer:
            del self.goal_configs["goal_robo_state_dict"]["poses"]



        # print("using VizServer, start meshcat-server in another terminal if not already running")
        # self.vis = VizServer()


    def __str__(self) -> str:
        # return all the terms in the buffer for which the command is generated
        msg = "GoalCommand:\n"
        for robo_id in self.goal_configs["goal_robo_state_dict"]["joint_pos"]:
            msg += f"\tRobot {robo_id} joint positions"
        for robo_id in self.goal_configs["goal_robo_state_dict"]["ee_pose"]:
            msg += f"\tRobot {robo_id} end effector pose"
        for obj_id in self.goal_configs["goal_obj_state_dict"]:
            msg += f"\tObject {obj_id} pose"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> dict:
        # return the commands as a dictionary for each robot and object

        command = {}

        for robo_id in self.robo_buffer:
            command["robot_" + str(robo_id)] = self.robo_buffer[robo_id]

        for obj_id in self.obj_buffer:
            command["obstacle_" + str(obj_id)] = self.obj_buffer[obj_id]

        return command

        # return self.vel_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # time for which the command was executed
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        # sample velocity commands

        # print(f"Resampling goal commands for envs {env_ids}")

        # sample the goal indices from goal configs
        goal_indices = torch.randint(0, self.n_goals, (len(env_ids),))

        if isinstance(env_ids, torch.Tensor):   
            env_ids = env_ids.detach().cpu().numpy()
            

        self.goal_indices[env_ids] = goal_indices.detach().cpu().numpy()

        # sample the goal robo state dict
        for robo_id in self.robo_buffer:
            # check if jpos exists
            # if "joint_pos" in self.robo_buffer[robo_id]:
            #     all_jpos = self.goal_configs["goal_robo_state_dict"]["joint_pos"][
            #         robo_id
            #     ]
            #     all_jpos = torch.from_numpy(np.stack(all_jpos)).float()
            #     self.robo_buffer[robo_id]["joint_pos"][env_ids] = all_jpos[
            #         goal_indices
            #     ].to(self.device)

            
            # check if ee pose exists
            if "ee_pose" in self.robo_buffer[robo_id]:
                all_ee_pose = self.goal_configs["goal_robo_state_dict"]["ee_pose"][
                    robo_id
                ]
                all_ee_pose = torch.from_numpy(np.stack(all_ee_pose)).float()
                self.robo_buffer[robo_id]["ee_pose"][env_ids] = all_ee_pose[
                    goal_indices
                ].to(self.device)

        # sample the goal obj state dict
        for obj_id in self.obj_buffer:
            # all_obj_pose = self.goal_configs["goal_obj_state_dict"][obj_id]
            # all_obj_pose = torch.from_numpy(np.stack(all_obj_pose)).float()
            all_obj_pose = self.goal_configs["goal_obj_state_dict"][obj_id]
            self.obj_buffer[obj_id][env_ids] = all_obj_pose[goal_indices].to(
                self.device
            )

    def _update_command(self):
        """Post-processes the velocity command.

        This function sets velocity command to zero for standing environments and computes angular
        velocity from heading direction if the heading_command flag is set.
        """
        pass

    def _set_debug_vis_impl(self, debug_vis: TYPE_CHECKING):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_marker_visualizer"):
                self.goal_marker_visualizer = VisualizationMarkers(
                    self.cfg.visualizer_cfg
                )
            # set visibility
            self.goal_marker_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_marker_visualizer"):
                self.goal_marker_visualizer.set_visibility(False)


    def _debug_vis_callback(self, event):
        # get the goal poses for the robot

        all_positions = []
        all_quats = []
        all_indices = []

        if self.cfg.task == "reach":
            # goal pose for robot exists

            # get the goal poses for the robots
            # command = self._env.command_manager.get_command("goal_command")
            robo_id = list(self.robo_buffer.keys())[0]
            asset_name = f"robot_{robo_id}"
            ee_pose_command = self.command[asset_name]["ee_pose"]
            
            # just reading some info
            
            possible_keys = list(self.cfg.marker_indices.keys())
            possible_keys = [k for k in possible_keys if f"robot_{robo_id}" in k and (not k.endswith("_cur"))]

            idx = self.cfg.marker_indices[possible_keys[0]] 
            
            # need to add env origins to the positions
            positions = ee_pose_command[:, :3] + self._env.scene.env_origins
                
            all_positions.append(positions.detach().cpu().numpy())
            all_quats.append(ee_pose_command[:, 3:].detach().cpu().numpy())
            all_indices.extend([idx] * len(ee_pose_command))
            
        if self.cfg.task == "push_simple":  
            
            obj_id = list(self.obj_buffer.keys())[0]
            asset_name = f"obstacle_{obj_id}"
            # for push task, the goal should be read from reward config term
            threshold = self._env.reward_manager.get_term_cfg(f"obj_{obj_id}_dist").params["x_dist_threshold"]
            # threshold is the x-coordinate of the object
            # y should be zero, z could be read from command cfg
            z = self.command["obstacle_" + str(obj_id)][:, 2].detach().cpu().numpy()
            quat = self.command["obstacle_" + str(obj_id)][:, 3:].detach().cpu().numpy()
            
            x = threshold * np.ones_like(z)
            positions = np.stack([x, np.zeros_like(x), z], axis=1) + self._env.scene.env_origins.detach().cpu().numpy()
            idx = self.cfg.marker_indices[f"obstacle_{obj_id}"]
            all_positions.append(positions)
            all_quats.append(quat)
            all_indices.extend([idx] * len(positions))
            
        
        if self.cfg.task == "push":
            obj_id = list(self.obj_buffer.keys())[0]
            asset_name = f"obstacle_{obj_id}"
            
            goal_poses = self.command["obstacle_" + str(obj_id)]
            positions = goal_poses[:, :3] + self._env.scene.env_origins
            quat = goal_poses[:, 3:]
        
            idx = self.cfg.marker_indices[f"obstacle_{obj_id}"]
            all_positions.append(positions.detach().cpu().numpy())
            all_quats.append(quat.detach().cpu().numpy())
            all_indices.extend([idx] * len(positions))
        
        # /////////////////////////////////////////////////////////////////////////////////
        # now mark the current poses for robot, and if they exist, the objects
        # robo_id = list(self.robo_buffer.keys())[0]
        
        robo_id = 0
        asset_name = f"robot_{robo_id}"
        body_name = self.cfg.robo_ee_names[str(robo_id)]
        
        scene_asset = self._env.scene[asset_name]
        body_id = scene_asset.body_names.index(body_name)
        body_pose = scene_asset.data.body_state_w[:, body_id, :7].detach().cpu().numpy()
        all_positions.append(body_pose[:, :3])
        all_quats.append(body_pose[:, 3:])
        idx = self.cfg.marker_indices[f"robot_{robo_id}_{body_name}_cur"]
        all_indices.extend([idx] * len(body_pose))
        
        # for object, it would be the object pose
        for obj_id in self.obj_buffer:
            asset_name = f"obstacle_{obj_id}"
            asset = self._env.scene[asset_name]
            wf_pose = asset.data.root_state_w[:, :7].detach().cpu().numpy()
            all_positions.append(wf_pose[:, :3])
            all_quats.append(wf_pose[:, 3:])
            idx = self.cfg.marker_indices[f"obstacle_{obj_id}_cur"]
            all_indices.extend([idx] * len(wf_pose))
            

        all_positions = np.concatenate(all_positions, axis=0)
        all_quats = np.concatenate(all_quats, axis=0)
        all_indices = np.array(all_indices)
        
        assert all_positions.shape[0] == all_quats.shape[0] == all_indices.shape[0]

        self.goal_marker_visualizer.visualize(
            translations=all_positions,
            orientations=all_quats,
            marker_indices=all_indices,
        )



@configclass
class MyGoalCommandCfg(CommandTermCfg):
    class_type: type = GoalCommand
    allowed_goal_configs_path: str = MISSING
    resampling_time_range = cfg.COMMAND.RESAMPLING_TIME_RANGE
    debug_vis: bool = cfg.COMMAND.DEBUG_VIS
    visualizer_cfg: VisualizationMarkersCfg = MISSING
    marker_indices: dict = {}
    robo_base_pose_dict: dict = {}
    ground_ht: float = 0.0
    robo_ee_names: dict = {}
    task: str = MISSING
    # obj_base_ht_dict: dict = {}


def get_robot_vis_cfg_dict(robot: eu.Robot, target='joint_pos', goal=True):
    goal_cfg_dict = {}
    current_cfg_dict = {}
    # each link get its own marker
    urdf_inst = URDF.load(robot.robot_urdf)
    fk = urdf_inst.link_fk()

    # create tmp usd files for links in the robot morphologies folder
    robot_urdf = Path(robot.robot_urdf)
    robot_mesh_dir = robot_urdf.parent / "usd_meshes"
    robot_mesh_dir.mkdir(exist_ok=True)
    
    # have two config dicts for each, one for goal and one for current configuration

    for idx, (link, pose) in enumerate(fk.items()):
        
        if not cfg.COMMAND.FULL_VISUALIZATION:
            # only visualize the end effector, otherwise skip
            if target == 'ee_pose' and link.name != robot.ee_link:
                continue
        
        name = link.name
        
        
        if goal:
            mesh = link.collision_mesh
            if mesh is None and target == 'joint_pos':
                continue 
            elif mesh is None and target == 'ee_pose':
                marker_type = 'frame'
            else:
                marker_type = cfg.GOAL_MARKER_TYPE
            
                
            if marker_type == 'mesh':
                # //////////////////////////////////////////
                # name = f"link_{idx}"
        
                # mesh = mesh.copy()
                # transform = np.eye(4)
                # transform[:3, :3] = R.from_euler("xyz", [-np.pi/2, 0, 0]).as_matrix()
                # mesh = mesh.apply_transform(transform)

                link_dir = robot_mesh_dir / name
                link_dir.mkdir(exist_ok=True)

                # get the usd path
                link_usd_path = link_dir / f"{name}.usd"
                if not link_usd_path.exists() or cfg.FORCE_ROBO_LINK_USD_CONVERSION:        
                    link_usd_path = str(link_usd_path)
                    tmp_obj_path = str(link_dir / f"{name}.obj")
                    trimesh.exchange.export.export_mesh(mesh, tmp_obj_path)
                    conversion_args = MeshArgsCli(
                        input=tmp_obj_path, output=link_usd_path, mass=1.0, 
                    )
                    mesh_to_usd_args(conversion_args)

                link_usd_path = str(link_usd_path)
                goal_cfg_dict[f"robot_{robot.robot_id}_{name}"] = sim_utils.UsdFileCfg(
                    usd_path=link_usd_path,
                    # scale of 100 is needed due to the obj to usd conversion
                    # scale=(100.0, 100.0, 100.0),
                    scale=(1.0, 1.0, 1.0),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=cfg.GOAL_MARKER_COLOR, opacity=0.1
                    ),
                )
                
            elif marker_type == 'frame':
                goal_cfg_dict[f"robot_{robot.robot_id}_{name}"] = sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                    scale=(0.1, 0.1, 0.1),
                )
            
        current_cfg_dict[f"robot_{robot.robot_id}_{name}_cur"] = sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(0.1, 0.1, 0.1),
            )


    combined_dict = {**goal_cfg_dict, **current_cfg_dict}
    return combined_dict


def get_obj_vis_cfg_dict(obj: eu.Prim):
    if obj.obj_type == "box":
        vis_cfg = sim_utils.CuboidCfg(
            size=obj.obj_shape,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=cfg.GOAL_MARKER_COLOR, opacity=0.1
            ),
        )
    elif obj.obj_type == "cylinder":
        vis_cfg = sim_utils.CylinderCfg(
            radius=obj.obj_shape[0],
            height=obj.obj_shape[1],
            axis="Z",
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=cfg.GOAL_MARKER_COLOR, opacity=0.1
            ),
        )

    else:
        raise NotImplementedError(
            f"Object type {obj.obj_type} not implemented for visualization"
        )

    current_vis_cfg = sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
        scale=(0.1, 0.1, 0.1),
    )

    return {"obstacle_" + str(obj.obj_id): vis_cfg, "obstacle_" + str(obj.obj_id) + "_cur": current_vis_cfg}


@configclass
class CommandsCfg:
    """Command specifications for the environment."""

    def __init__(self, prob: eu.ProblemSpec, unique_name: str, task: str):
        # create a dictionary of visualization configs
        # we create set of visualizers for each robotic link and each object
        # kind of follow the view robot from vis_server

        vis_cfg_dicts = {}
        robo_base_pose_dict = {}
        # obj_base_pose_dict = {}
        robo_paths = {}
        robo_ee_names = {}

        # beyond python 3.7, dict maintains insertion order, so we know the marker indices

        for robo_id in prob.goal_robo_state_dict:
            robo_base_pose_dict[str(robo_id)] = prob.robot_dict[robo_id].pose


        if cfg.COMMAND.DEBUG_VIS:
            # we create a visualizer only for robots and objects, for which a valid, not-None goal exists
            
            for robo_id in prob.robot_dict:
                # for robo_id, robo_state in prob.goal_robo_state_dict.items():
                if robo_id in prob.goal_robo_state_dict and prob.goal_robo_state_dict[robo_id].ee_pose is not None:
                    vis_cfg_dicts.update(get_robot_vis_cfg_dict(prob.robot_dict[robo_id], target='ee_pose', goal=True))
                else:
                    vis_cfg_dicts.update(get_robot_vis_cfg_dict(prob.robot_dict[robo_id], target='ee_pose', goal=False))
                    
                robo_paths[robo_id] = Path(prob.robot_dict[robo_id].robot_urdf).parent
            

            robo_ee_names = {str(robo_id): robot.ee_link for robo_id, robot in prob.robot_dict.items()}

            for obj_id, obj_state in prob.goal_obj_state_dict.items():
                # check if the goal is not None
                if obj_state.pose is not None:
                    vis_cfg_dicts.update(get_obj_vis_cfg_dict(prob.obj_dict[obj_id]))
                    # obj_base_ht_dict[obj_id] = prob.obj_dict[obj_id].pose[2, 3]


            vis_index_mapping = {k: idx for idx, k in enumerate(vis_cfg_dicts)}

            

            visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
                prim_path="/Visuals/Command/goal_marker", markers=vis_cfg_dicts
            )

        else:
            visualizer_cfg = None
            vis_index_mapping = {}



        # check if randomized configs exist
        if prob.additional_configs and (not cfg.FIXED_GOAL):
            # load from the
            self.goal_command = MyGoalCommandCfg(
                allowed_goal_configs_path=prob.additional_configs,
                visualizer_cfg=visualizer_cfg,
                marker_indices=vis_index_mapping,
                robo_base_pose_dict=robo_base_pose_dict,
                debug_vis=cfg.COMMAND.DEBUG_VIS,
                resampling_time_range=cfg.COMMAND.RESAMPLING_TIME_RANGE,
                # obj_base_ht_dict=obj_base_ht_dict,
                ground_ht=prob.ground,
                robo_ee_names=robo_ee_names,
                task=task
            )
            
            print("initialized command cfg", self.goal_command.debug_vis)

        else:
            
            eu.save_default_init_goal_configs(prob, save_dirname="default", save_fname=f"{unique_name}.pkl")

            # essentially fixed goal
            self.goal_command = MyGoalCommandCfg(
                allowed_goal_configs_path=f"default/{unique_name}.pkl",
                # allowed_goal_configs_path=f"{rand_name}.pkl",
                visualizer_cfg=visualizer_cfg,
                resampling_time_range=(10000, 10000),
                marker_indices=vis_index_mapping,
                robo_base_pose_dict=robo_base_pose_dict,
                ground_ht=prob.ground,
                robo_ee_names={},
                task=task   
            )
