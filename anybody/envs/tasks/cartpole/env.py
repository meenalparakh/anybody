# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch
import carb 
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.classic.cartpole.mdp as mdp

##
# Pre-defined configs
##
# from omni.isaac.lab_assets.cartpole import CARTPOLE_CFG  # isort:skip
from anybody.robot_morphologies import get_default_robo_cfg
from anybody.envs.isaac_env_components import funcs as ifunc
from anybody.cfg import cfg as global_cfg
import numpy as np
import omni.isaac.core.utils.prims as prim_utils

##
# Scene definition
##


@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    def __init__(self, robo_type, robo_name, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if prim_utils.is_prim_path_valid("/World/ground"):
            carb.log_info("Ground plane exists")
        else:   
            # ground plane
            self.ground = AssetBaseCfg(
                prim_path="/World/ground",
                spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
                collision_group=-1,
            )
            self.ground.init_state.pos = (0.0, 0.0, -3.0)

        self.robot = get_default_robo_cfg(robo_type, robot_name=robo_name, robo_id=0)
        # check if light prim exists
        if prim_utils.is_prim_path_valid("/World/Light"):
            carb.log_info("Light exists")
        else:
            # lights
            self.dome_light = AssetBaseCfg(
                prim_path="/World/Light",
                spawn=sim_utils.DomeLightCfg(
                    intensity=3000.0, color=(0.75, 0.75, 0.75)
                ),
            )
            self.dome_light.init_state.pos = (0.0, 0.0, 10.0)

##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    # no commands for this MDP
    null = mdp.NullCommandCfg()


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot", joint_names=["x"], scale=100.0
    )

@configclass
class PolicyCfg(ObsGroup):
    """Observations for policy group."""

    # observation terms (order preserved)
    joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True

@configclass
class MyPolicyCfg(ObsGroup):
    def __init__(self, robo_name, robo_id):
        super().__init__()
        
        # get pole length and mass
        idx = int(robo_name[1:])
        lengths = np.linspace(0.5, 1.5, 5)
        masses = np.linspace(0.5, 1.5, 5)
        pole_vec = torch.tensor([lengths[idx], masses[idx]])
    
        if global_cfg.OBSERVATION.MASK_ROBO_MORPH:
            pole_vec = torch.zeros(2)
    
    
        self.robo_id = ObsTerm(
            func=ifunc.return_vec,
            params={
                "vec": torch.tensor([robo_id]),
            }
        )
    
        self.pole_info = ObsTerm(
            func=ifunc.return_vec,
            params={
                "vec": pole_vec,
            },
        )
        self.joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        self.joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        
    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True
    

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    def __init__(self, robo_name, robo_id):
        super().__init__()
        # observation groups
        self.policy: MyPolicyCfg = MyPolicyCfg(robo_name, robo_id)

@configclass
class EventCfg:
    """Configuration for events."""

    # reset
    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["x"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.5, 0.5),
        },
    )

    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["theta"]),
            "position_range": (-0.25 * math.pi, 0.25 * math.pi),
            "velocity_range": (-0.25 * math.pi, 0.25 * math.pi),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # (3) Primary task: keep pole upright
    pole_pos = RewTerm(
        func=mdp.joint_pos_target_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["theta"]), "target": 0.0},
    )
    # (4) Shaping tasks: lower cart velocity
    cart_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["x"])},
    )
    # (5) Shaping tasks: lower pole angular velocity
    pole_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["theta"])},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Cart out of bounds
    cart_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["x"]),
            "bounds": (-3.0, 3.0),
        },
    )


@configclass
class CurriculumCfg:
    """Configuration for the curriculum."""

    pass


##
# Environment configuration
##


@configclass
class CartpoleEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    def __init__(
        self, robo_type, robo_name, robo_id, cfg, device, pos_offset=np.zeros(3), unique_name=""
    ):
        # assuming prob is the robo_name
        super().__init__()
        # Scene settings
        self.scene: CartpoleSceneCfg = CartpoleSceneCfg(
            robo_type,
            robo_name,
            env_prefix="cartpole",
            pos_offset=pos_offset,
            num_envs=cfg.TRAIN.NUM_ENVS,
            env_spacing=5.0
        )
        # Basic settings
        self.observations: ObservationsCfg = ObservationsCfg(robo_name=robo_name, robo_id=robo_id)
        self.actions: ActionsCfg = ActionsCfg()
        self.events: EventCfg = EventCfg()
        # MDP settings
        self.curriculum: CurriculumCfg = CurriculumCfg()
        self.rewards: RewardsCfg = RewardsCfg()
        self.terminations: TerminationsCfg = TerminationsCfg()
        # No command generator
        self.commands: CommandsCfg = CommandsCfg()

        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120

        self.sim.device = device