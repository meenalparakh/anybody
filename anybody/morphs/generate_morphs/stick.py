import os

import numpy as np
from collections import OrderedDict

# from anybody.utils.vis_server import VizServer
from anybody.utils.path_utils import get_robot_morphs_dir
from anybody.utils.utils import format_str
from anybody.utils.collision_utils import (
    get_collision_spheres_stick,
    convert_collisions_info_to_str,
)
from .utils import update_urdf

base_link = """
    <link name="base_link">
    <inertial>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <mass value="0.1"/>
        <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
    </link>
"""

stick_link = """
    
    <link name="link_prism_x">
    </link>
    
    <link name="link_rev_x">
    </link>
    
    <link name="link_rev_z">
        <inertial>
            <origin xyz="HALFLENGTH 0 0" rpy="0 0 0"/>
            <mass value="1.0"/>
            <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
        </inertial>
        <visual>
            <origin xyz="HALFLENGTH 0 0" rpy="0 0 0"/>
            <geometry>
                <box size="LENGTH 0.02 0.02"/>
            </geometry>
            <material name="blue">
                <color rgba="0 0 1 1"/>
            </material>
        </visual>
        <collision>
            <origin xyz="HALFLENGTH 0 0" rpy="0 0 0"/>
            <geometry>
                <box size="COLLISIONLENGTH 0.02 0.02"/>
            </geometry>
        </collision>
    </link>

    <!-- Link ball -->
    <link name="link_ball">
        <inertial>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <mass value="0.01"/>
            <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
        </inertial>
        <visual>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <sphere radius="0.03"/>
            </geometry>
            <material name="blue">
                <color rgba="0 0 1 1"/>
            </material>
        </visual>
        <collision>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <sphere radius="0.03"/>
            </geometry>
        </collision>
    </link>
    
"""


def get_obj_joint(length, radius=0.03):
    obj_joint = f"""
        <joint name="joint_px" type="prismatic">
            <parent link="base_link"/>
            <child link="link_prism_x"/>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <axis xyz="1 0 0"/>
            <limit lower="-100" upper="100" effort="100." velocity="1"/>
        </joint>

        <joint name="joint_rx" type="revolute">
            <parent link="link_prism_x"/>
            <child link="link_rev_x"/>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <axis xyz="1 0 0"/>
            <limit lower="-{2 * np.pi}" upper="{2 * np.pi}" effort="100." velocity="1"/>
        </joint>

        <joint name="joint_rz" type="revolute">
            <parent link="link_rev_x"/>
            <child link="link_rev_z"/>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <axis xyz="0 0 1"/>
            <limit lower="-{2 * np.pi}" upper="{2 * np.pi}" effort="100." velocity="1"/>
        </joint>
        
        <!-- Joint final -->
        <joint name="joint_final" type="fixed">
            <parent link="link_rev_z"/>
            <child link="link_ball"/>
            <origin xyz="{length + radius} 0 0" rpy="0 0 0"/>
        </joint>
"""

    return obj_joint


def get_urdf(length):
    final_urdf = f"""<?xml version="1.0"?>
<robot name="stick_robot">
{base_link}
{stick_link}
{get_obj_joint(length)}
</robot>
"""

    final_urdf = format_str(
        final_urdf,
        OrderedDict(
            [
                ("HALFLENGTH", str(length / 2.0)),
                ("COLLISIONLENGTH", str(length * 0.95)),
                ("LENGTH", str(length)),
            ]
        ),
    )

    # update joint info
    joint_names = ["joint_px", "joint_rx", "joint_rz"]
    joint_lb = [-2.0, -2 * np.pi, -2 * np.pi]
    joint_ub = [2.0, 2 * np.pi, 2 * np.pi]
    effort = [100.0, 100.0, 100.0]
    velocity = [5.0, 5.0, 5.0]
    joint_updates = [
        {"name": j, "lower": lb, "upper": ub, "effort": e, "velocity": v}
        for j, lb, ub, e, v in zip(joint_names, joint_lb, joint_ub, effort, velocity)
    ]

    final_urdf = update_urdf(final_urdf, joint_updates)

    collision_spheres = OrderedDict(
        [
            ("link_rev_z", get_collision_spheres_stick(length, 0.03, 0.02)),
            ("link_ball", [(np.array([0.0, 0.0, 0.0]), 0.03)]),
        ]
    )
    return final_urdf, collision_spheres


def save_urdfs(to_usd=False, randomized=False):
    
    if not randomized:
        lengths = np.linspace(0.2, 0.7, 5)
    else:
        lengths = np.linspace(0.1, 0.8, 100)

    # robo info
    joint_names = ["joint_px", "joint_rx", "joint_rz"]
    joint_default_vals = [0.0, 0.0, 0.0]
    collision_ignore = {"link_rev_z": ["link_ball"], "link_ball": ["link_rev_z"]}
    self_collision_buffer = {"link_rev_z": 0.0, "link_ball": 0.0}
    null_space_weight = [1.0] * len(joint_names)
    cspace_distance_weight = [1.0] * len(joint_names)

    (get_robot_morphs_dir() / "stick").mkdir(parents=True, exist_ok=True)
    robo_names = []

    for s, l in enumerate(lengths):
        final_urdf, collision_spheres_dict = get_urdf(l)
        
        robo_name = f"stick_{s}" if not randomized else f"r{s}"
        robot_dir = get_robot_morphs_dir() / "stick" / robo_name

        robo_names.append(str(robot_dir))
        robot_dir.mkdir(parents=True, exist_ok=True)
        urdf_path = robot_dir / f"{robo_name}.urdf"
        with open(urdf_path, "w") as f:
            f.write(final_urdf)

        if to_usd:
            usd_path = robot_dir / f"{robo_name}.usd"
            # check if usd exists
            from anybody.utils.to_usd import ArgsCli, main
            args_cli = ArgsCli(input=str(urdf_path), output=str(usd_path), headless=True)
            main(args_cli)

            # create collision spheres yml
            collision_spheres_yml = robot_dir / "collision_spheres.yml"
            collision_spheres_str = convert_collisions_info_to_str(
                collision_spheres_dict
            )

            # write collision spheres yml
            with open(collision_spheres_yml, "w") as f:
                f.write(collision_spheres_str)

            # # writing the robot config file
            # # load the template from robot_morphologies / 'template_cfg.yml'
            # with open(get_robot_morphs_dir() / "template_cfg.yml", "r") as f:
            #     template_cfg = yaml.safe_load(f)

            # assert template_cfg is not None
            # # assert 'robot_cfg' in template_cfg
            # # assert 'kinematics' in template_cfg['robot_cfg']
            # # assert 'cspace' in template_cfg['robot_cfg']['kinematics']

            # # write the cfg file
            # template_cfg["robot_cfg"]["kinematics"]["usd_path"] = str(usd_path)
            # template_cfg["robot_cfg"]["kinematics"]["usd_robot_root"] = f"/{robo_name}"
            # template_cfg["robot_cfg"]["kinematics"]["urdf_path"] = str(urdf_path)

            # template_cfg["robot_cfg"]["kinematics"]["asset_root_path"] = str(robot_dir)
            # template_cfg["robot_cfg"]["kinematics"]["base_link"] = "base_link"
            # template_cfg["robot_cfg"]["kinematics"]["ee_link"] = "link_ball"

            # template_cfg["robot_cfg"]["kinematics"]["collision_link_names"] = [
            #     "link_rev_z",
            #     "link_ball",
            # ]
            # template_cfg["robot_cfg"]["kinematics"]["collision_spheres"] = str(
            #     collision_spheres_yml
            # )

            # template_cfg["robot_cfg"]["kinematics"]["self_collision_ignore"] = (
            #     collision_ignore
            # )
            # template_cfg["robot_cfg"]["kinematics"]["self_collision_buffer"] = (
            #     self_collision_buffer
            # )

            # template_cfg["robot_cfg"]["kinematics"]["mesh_link_names"] = [
            #     "link_rev_z",
            #     "link_ball",
            # ]
            # template_cfg["robot_cfg"]["kinematics"]["cspace"]["joint_names"] = (
            #     joint_names
            # )
            # template_cfg["robot_cfg"]["kinematics"]["cspace"]["retract_config"] = (
            #     joint_default_vals
            # )
            # template_cfg["robot_cfg"]["kinematics"]["cspace"]["null_space_weight"] = (
            #     null_space_weight
            # )
            # template_cfg["robot_cfg"]["kinematics"]["cspace"][
            #     "cspace_distance_weight"
            # ] = cspace_distance_weight

            # curobo_cfg = robot_dir / "curobo_cfg.yml"
            # with open(curobo_cfg, "w") as f:
            #     yaml.safe_dump(template_cfg, f, sort_keys=False)

    return robo_names
