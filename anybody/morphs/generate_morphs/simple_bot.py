import os
import numpy as np
from collections import OrderedDict

import urdfpy
from anybody.utils.vis_server import VizServer
from anybody.utils.path_utils import get_robot_morphs_dir, get_tmp_mesh_storage
from anybody.utils.utils import format_str
from .utils import update_urdf


base_link = """
    <link name="base_link">
    <inertial>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <mass value="0.0"/>
        <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
    </link>
"""


bot_base = """

    <!-- Circular disc, axis along z-axis -->
    <link name="bot_base">
        <inertial>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <mass value="1.0"/>
            <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
        </inertial>
        <visual>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <cylinder length="0.05" radius="RADIUS" />
            </geometry>
            <material name="blue">
                <color rgba="0 0 1 1"/>
            </material>
        </visual>
        
        <collision>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <cylinder length="0.03" radius="RADIUS" />
            </geometry>
        </collision>

    </link>
    
    <joint name="joint_base_rev" type="revolute">
        <parent link="base_link"/>
        <child link="bot_base"/>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-1.57" upper="1.57" effort="1" velocity="1"/>
    </joint>
    
"""

stick_link = """

    <!-- Link LINKNAME -->
    
    <link name="link_LINKNAME">
        <inertial>
            <origin xyz="HALFLENGTH 0 0" rpy="0 0 0"/>
            <mass value="0.05"/>
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
                <box size="COLLISIONLENGTH 0.015 0.015"/>
            </geometry>
        </collision>
    </link>
    """

ee_base_link = """

    <!-- Link dummy for ee -->
    <link name="ee_link_base">
        <inertial>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <mass value="0.005"/>
            <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
        </inertial>
        <visual>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <sphere radius="0.005"/>
            </geometry>
            <material name="red">   
                <color rgba="1 0 0 1"/> 
            </material>
        </visual>
    </link>
"""

ee_base_joint = """ 
    <joint name="joint_ee_LINKNAME_rev" type="revolute">
        <parent link="link_LINKNAME"/>
        <child link="ee_link_base"/>
        <origin xyz="LENGTH 0 0" rpy="0 0 0"/>
        <axis xyz="1 0 0"/>
        <limit lower="-1.57" upper="1.57" effort="1" velocity="1"/>
    </joint>
"""


stick_base_joint = f"""
    <joint name="joint_LINKNAME_rev" type="revolute">
        <parent link="bot_base"/>
        <child link="link_LINKNAME"/>
        <origin xyz="0 0 0.045" rpy="0 0 0"/>
        <axis xyz="0 1 0"/>
        <limit lower="{-5 * np.pi/6}" upper="{0}" effort="1" velocity="1"/>
    </joint>
    
"""

stick_stick_joint = """
    <joint name="joint_LINKNAME_rev" type="revolute">
        <parent link="link_PARENT"/>
        <child link="link_LINKNAME"/>
        <origin xyz="LENGTH 0 0" rpy="0 0 0"/>
        <axis xyz="0 1 0"/>
        <limit lower="-1.57" upper="1.57" effort="1" velocity="1"/>
    </joint>
"""


def ee_link(collision=True):
    ee_link = """
        <link name="ee_link_SIDE">
            <inertial>
                <origin xyz="HALFEESIZE 0 0" rpy="0 0 0"/>
                <mass value="0.010"/>
                <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
            </inertial>
            <visual>
                <origin xyz="HALFEESIZE 0 0" rpy="0 0 0"/>
                <geometry>
                    <box size="EESIZE 0.005 0.02"/>
                </geometry>
                <material name="blue">
                    <color rgba="0 0 1 1"/>
                </material>
            </visual>
"""
    if collision:
        ee_link += """
            <collision>
                <origin xyz="HALFEESIZE 0 0" rpy="0 0 0"/>
                <geometry>
                    <box size="EECOLLISION 0.005 0.02"/>
                </geometry>
            </collision>
        </link>     
"""
    else:
        ee_link += """
        </link>
"""
    return ee_link


ee_arm_joint = """
    <joint name="joint_ee_SIDE_rev" type="revolute">
        <parent link="ee_link_base"/>
        <child link="ee_link_SIDE"/>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-1.57" upper="1.57" effort="1" velocity="1"/>
    </joint>
"""

ee_ee_joint = """
    <joint name="joint_ee_SIDE_rev" type="revolute">
        <parent link="ee_link_PARENT"/>
        <child link="ee_link_SIDE"/>
        <origin xyz="EESIZE 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-1.57" upper="1.57" effort="1" velocity="1"/>
    </joint>
"""

obj_urdf = f"""<?xml version="1.0"?>
<robot name="obj">
    {base_link}
    {bot_base}
    {stick_link.replace("LINKNAME", "1")}
    {stick_base_joint.replace("LINKNAME", "1")}
    {stick_link.replace("LINKNAME", "2")}
    {stick_stick_joint.replace("LINKNAME", "2").replace("PARENT", "1")}
    {stick_link.replace("LINKNAME", "3")}
    {stick_stick_joint.replace("LINKNAME", "3").replace("PARENT", "2")}
    {ee_base_link}
    {ee_base_joint.replace("LINKNAME", "3")}
    {ee_link(collision=False).replace("SIDE", "left1")}
    {ee_link(collision=False).replace("SIDE", "right1")}
    {ee_arm_joint.replace("LINKNAME", "3").replace("SIDE", "left1")}    
    {ee_arm_joint.replace("LINKNAME", "3").replace("SIDE", "right1")}    
    {ee_link(collision=True).replace('SIDE', "left2")}
    {ee_link(collision=True).replace('SIDE', "right2")}
    {ee_ee_joint.replace("PARENT", "left1").replace("SIDE", 'left2')}
    {ee_ee_joint.replace("PARENT", "right1").replace("SIDE", 'right2')}
    
</robot>
"""


def save_urdfs(to_usd=False, randomized=False):
    
    if not randomized:
        radii = np.linspace(0.05, 0.1, 4)
        length = np.linspace(0.2, 0.4, 4)
        half_length = length / 2
        collision_length = length * 0.85
        ee_size = np.linspace(0.03, 0.06, 4)
        ee_collision = ee_size * 0.85
        half_ee = ee_size / 2

        names = [f"bot_{i}" for i in range(4)]

    else:
        val_ranges = {
            "radii": (0.05, 0.1),
            "length": (0.1, 0.5),
            "ee_size": (0.02, 0.07),
        }

        radii = np.random.uniform(*val_ranges["radii"], 100)
        length = np.random.uniform(*val_ranges["length"], 100)
        half_length = length / 2
        collision_length = length * 0.85
        ee_size = np.random.uniform(*val_ranges["ee_size"], 100)
        ee_collision = ee_size * 0.85
        half_ee = ee_size / 2
        
        names = [f"r{i}" for i in range(100)]


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
    # default_jvals = [np.pi/2, -1.0, 0.7, 1.0, 0.0, 0.5, -0.5, -0.5, 0.5]

    jvals_low = [
        0,
        -5 * np.pi / 6,
        -np.pi / 2,
        -np.pi / 2,
        -np.pi / 2,
        0,
        -np.pi / 2,
        -np.pi / 2,
        0,
    ]
    jvals_high = [np.pi, 0, np.pi, np.pi, np.pi / 2, np.pi / 2, 0, 0, np.pi / 2]

    j_efforts = [100.0] * 9
    j_vels = [5.0] * 9

    joint_updates = [
        {"name": j, "lower": lb, "upper": ub, "effort": e, "velocity": v}
        for j, lb, ub, e, v in zip(
            joint_names, jvals_low, jvals_high, j_efforts, j_vels
        )
    ]


    # n = 4 if not randomized else 100

    # (get_robot_morphs_dir() / "simple_bot").mkdir(parents=True, exist_ok=True)
    for i in range(len(radii)):
        
        urdf = format_str(
            obj_urdf,
            OrderedDict(
                [
                    ("RADIUS", radii[i]),
                    ("HALFLENGTH", half_length[i]),
                    ("COLLISIONLENGTH", collision_length[i]),
                    ("LENGTH", length[i]),
                    ("HALFEESIZE", half_ee[i]),
                    ("EECOLLISION", ee_collision[i]),
                    ("EESIZE", ee_size[i]),
                ]
            ),
        )

        urdf = update_urdf(urdf, joint_updates)

        robo_dir = get_robot_morphs_dir() / "simple_bot" / f"{names[i]}"
        robo_dir.mkdir(parents=True, exist_ok=True)

        urdf_path = robo_dir / f"{names[i]}.urdf"

        with open(urdf_path, "w") as f:
            f.write(urdf)

        if to_usd:
            usd_path = robo_dir / f"{names[i]}.usd"
            # if not os.path.exists(usd_path):
            from anybody.utils.to_usd import ArgsCli, main
            args_cli = ArgsCli(input=str(urdf_path), output=str(usd_path), headless=True)
            main(args_cli)



if __name__ == "__main__":
    # save_urdfs()
    # exit()

    obj_urdf = format_str(
        obj_urdf,
        OrderedDict(
            [
                ("RADIUS", 0.1),
                ("HALFLENGTH", 0.15),
                ("COLLISIONLENGTH", 0.25),
                ("LENGTH", 0.3),
                ("HALFEESIZE", 0.02),
                ("EECOLLISION", 0.03),
                ("EESIZE", 0.04),
            ]
        ),
    )

    get_tmp_mesh_storage().mkdir(parents=True, exist_ok=True)
    urdf_path = get_tmp_mesh_storage() / f"simple_bot.urdf"

    with open(urdf_path, "w") as f:
        f.write(obj_urdf)

    print("obj.urdf file created successfully!")
    # joint_vals = [np.random.uniform(-5, 5) for _ in range(2)]

    joint_names = [
        "joint_base_rev",
        "joint_1_rev",
        "joint_2_rev",
        "joint_3_rev",
        "joint_ee_3_rev",
        "joint_ee_left1_rev",
        "joint_ee_right1_rev",
        "joint_ee_left2_rev",
        "joint_ee_right2_rev",
    ]

    joint_vals = np.random.uniform(-1, 1, len(joint_names))
    joint_vals = [0.0, -1.0, 0.7, 1.0, -1.0, 0.5, -0.5, -0.5, 0.5]

    print("joint_vals: ", joint_vals)
    urdfpy_robot = urdfpy.URDF.load(urdf_path)

    vis = VizServer()
    vis.view_robot(
        urdfpy_robot, joint_names, joint_vals, "obj_robot", 0x00FF00, mesh_type="visual"
    )
    vis.view_robot(
        urdfpy_robot,
        joint_names,
        joint_vals,
        "obj_robot_col",
        0x0000FF,
        mesh_type="collision",
    )
