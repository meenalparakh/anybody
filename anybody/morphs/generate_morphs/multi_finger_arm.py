import os
import numpy as np
import urdfpy
from collections import OrderedDict

from anybody.utils.vis_server import VizServer
from anybody.utils.path_utils import get_robot_morphs_dir, get_tmp_mesh_storage
from anybody.utils.utils import format_str
from .utils import update_urdf, update_joint_params
from .multi_finger_v2 import get_hand_urdf


base_link = """
    <link name="base_link">
    <inertial>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <mass value="1.0"/>
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


def get_stick_link(linkname, length, collision_length):
    stick_link = f"""

    <!-- Link {linkname} -->
    
    <link name="link_{linkname}">
        <inertial>
            <origin xyz="{length /2} 0 0" rpy="0 0 0"/>
            <mass value="1.0"/>
            <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
        </inertial>
        <visual>
            <origin xyz="{length / 2} 0 0" rpy="0 0 0"/>
            <geometry>
                <box size="{length} 0.02 0.02"/>
            </geometry>
            <material name="blue">
                <color rgba="0 0 1 1"/>
            </material>
        </visual>
        <collision>
            <origin xyz="{length / 2} 0 0" rpy="0 0 0"/>
            <geometry>
                <box size="{collision_length} 0.015 0.015"/>
            </geometry>
        </collision>
    </link>
    """

    return stick_link


stick_base_joint = f"""
    <joint name="joint_LINKNAME_rev" type="revolute">
        <parent link="bot_base"/>
        <child link="link_LINKNAME"/>
        <origin xyz="0 0 0.045" rpy="0 0 0"/>
        <axis xyz="0 1 0"/>
        <limit lower="{-5 * np.pi/6}" upper="{0}" effort="1" velocity="1"/>
    </joint>
    
"""


def get_stick_stick_joint(linkname, parent_name, length):
    stick_stick_joint = f"""
    <joint name="joint_{linkname}_rev" type="revolute">
        <parent link="link_{parent_name}"/>
        <child link="link_{linkname}"/>
        <origin xyz="{length} 0 0" rpy="0 0 0"/>
        <axis xyz="0 1 0"/>
        <limit lower="-1.57" upper="1.57" effort="1" velocity="1"/>
    </joint>
    """
    return stick_stick_joint


def get_urdf(
    per_finger_lengths,
    base_radius=0.07,
    rad=0.03,
    arm_length=0.3,
    last_link_length=0.05,
    finger_params={},
    eq_distant=True,
):
    # base_radius = 0.07
    # rad = 0.03
    # arm_length = 0.3
    # last_link_length = 0.05

    # per_finger_lengths = [[0.05, 0.03, 0.02]] * 3 + [[0.04, 0.02, 0.02]]
    n_fingers = len(per_finger_lengths)
    hand_urdf, hand_basename = get_hand_urdf(
        n_fingers,
        per_finger_lengths=per_finger_lengths,
        equally_distanced=eq_distant,
        base_link_origin=[0, 0, 0],
        base_radius=rad,
        finger_params=finger_params,
    )

    wrist_joint = f"""
    <joint name="joint_wrist_rev" type="fixed">
        <parent link="link_3"/>
        <child link="{hand_basename}"/>
        <origin xyz="{last_link_length} 0 0" rpy="0 {-np.pi/2} 0"/>
    </joint>
    """

    obj_urdf = f"""<?xml version="1.0"?>
<robot name="obj">
    {base_link}
    {bot_base}
    {get_stick_link("1", length=arm_length, collision_length = arm_length*0.85)}
    {stick_base_joint.replace("LINKNAME", "1")}
    {get_stick_link("2", length=arm_length, collision_length = arm_length*0.85)}
    {get_stick_stick_joint("2", "1", arm_length)}
    {get_stick_link("3", length=last_link_length, collision_length = last_link_length*0.85)}
    {get_stick_stick_joint("3", "2", arm_length)}
    {hand_urdf}
    {wrist_joint}
</robot>
"""

    new_urdf = format_str(
        obj_urdf,
        OrderedDict(
            [
                ("RADIUS", base_radius),
                ("HALFLENGTH", arm_length / 2),
                ("COLLISIONLENGTH", arm_length * 0.85),
                ("LENGTH", arm_length),
            ]
        ),
    )

    return new_urdf


def save_urdfs(to_usd=False):
    radii = np.linspace(0.05, 0.1, 4)
    length = np.linspace(0.2, 0.4, 4)

    n_fingers = [3, 3, 4, 4]

    finger_link_lengths1 = [0.04, 0.03, 0.02]
    finger_link_lengths2 = [0.03, 0.05]

    equidistant = [True, False, True, False]


    joint_names = [
        "joint_base_rev",
        "joint_1_rev",
        "joint_2_rev",
        "joint_3_rev",
    ]
    # default_jvals = [np.pi/2, -1.0, 0.7, 1.0, 0.0, 0.5, -0.5, -0.5, 0.5]

    jvals_low = [
        0,
        -5 * np.pi / 6,
        -np.pi / 2,
        -np.pi / 2,
        # -np.pi / 2,
        # 0,
        # -np.pi / 2,
        # -np.pi / 2,
        # 0,
    ]
    jvals_high = [
        np.pi,
        0,
        np.pi,
        np.pi,
        #   np.pi / 2, np.pi / 2, 0, 0, np.pi / 2
    ]

    j_efforts = [100.0] * len(joint_names)
    j_vels = [5.0] * len(joint_names)

    joint_updates = [
        {"name": j, "lower": lb, "upper": ub, "effort": e, "velocity": v}
        for j, lb, ub, e, v in zip(
            joint_names, jvals_low, jvals_high, j_efforts, j_vels
        )
    ]

    # (get_robot_morphs_dir() / "simple_bot").mkdir(parents=True, exist_ok=True)
    for i in range(4):
        for fl_idx, fl in enumerate([finger_link_lengths1, finger_link_lengths2]):
            finger_params = {
                # "pitch": np.pi / 6,
                "link_thickness": 0.01,
                "link_width": 0.01,
            }

            fls = [fl] * n_fingers[i]

            urdf = get_urdf(
                fls,
                base_radius=radii[i],
                arm_length=length[i],
                finger_params=finger_params,
                eq_distant=equidistant[i],
            )

            urdf = update_urdf(urdf, joint_updates)

            name = f"mf_arm_{i}_{fl_idx}"

            robo_dir = get_robot_morphs_dir() / "mf_arm" / f"{name}"
            robo_dir.mkdir(parents=True, exist_ok=True)

            urdf_path = robo_dir / f"{name}.urdf"

            with open(urdf_path, "w") as f:
                f.write(urdf)

            urdf = update_joint_params(urdf_path)

            if to_usd:
                usd_path = robo_dir / f"{name}.usd"
                if not os.path.exists(usd_path):
                    from anybody.utils.to_usd import ArgsCli, main
                    args_cli = ArgsCli(input=str(urdf_path), output=str(usd_path), headless=True)
                    main(args_cli)


if __name__ == "__main__":
    # write urdf to file

    finger_link_lengths2 = [0.03, 0.05]
    fls = [finger_link_lengths2] * 3

    my_urdf = get_urdf(fls)
    urdf_path = get_tmp_mesh_storage() / "mf_hand.urdf"
    with open(urdf_path, "w") as f:
        f.write(my_urdf)

    urdfpy_robot = urdfpy.URDF.load(urdf_path)
    vis = VizServer()

    joint_names = [j.name for j in urdfpy_robot.joints if j.joint_type != "fixed"]

    # joint_names = ["joint_wrist_rev"]

    jlimits = {j.name: j.limit for j in urdfpy_robot.joints}
    joint_lb = [jlimits[jname].lower for jname in joint_names]
    joint_ub = [jlimits[jname].upper for jname in joint_names]

    for _ in range(20):
        # joint_vals = np.zeros(len(joint_names))
        joint_vals = np.random.uniform(joint_lb, joint_ub)
        # joint_vals[:3] = [0, 0.0, 0.0]

        vis.view_robot(
            urdfpy_robot,
            joint_names,
            joint_vals,
            "obj_robot",
            0x00FF00,
            mesh_type="visual",
        )
        vis.view_robot(
            urdfpy_robot,
            joint_names,
            joint_vals,
            "obj_robot_col",
            0x0000FF,
            mesh_type="collision",
        )
        input("Press Enter to continue...")


# if __name__ == "__main__":
#     save_urdfs()
#     exit()

# obj_urdf = format_str(
#     obj_urdf,
#     OrderedDict(
#         [
#             ("RADIUS", 0.1),
#             ("HALFLENGTH", 0.15),
#             ("COLLISIONLENGTH", 0.25),
#             ("LENGTH", 0.3),
#             ("HALFEESIZE", 0.02),
#             ("EECOLLISION", 0.03),
#             ("EESIZE", 0.04),
#         ]
#     ),
# )

# get_tmp_mesh_storage().mkdir(parents=True, exist_ok=True)
# urdf_path = get_tmp_mesh_storage() / f"simple_bot.urdf"

# with open(urdf_path, "w") as f:
#     f.write(obj_urdf)

# print("obj.urdf file created successfully!")
# # joint_vals = [np.random.uniform(-5, 5) for _ in range(2)]

# joint_names = [
#     "joint_base_rev",
#     "joint_1_rev",
#     "joint_2_rev",
#     "joint_3_rev",
#     "joint_ee_3_rev",
#     "joint_ee_left1_rev",
#     "joint_ee_right1_rev",
#     "joint_ee_left2_rev",
#     "joint_ee_right2_rev",
# ]

# joint_vals = np.random.uniform(-1, 1, len(joint_names))
# joint_vals = [0.0, -1.0, 0.7, 1.0, -1.0, 0.5, -0.5, -0.5, 0.5]

# print("joint_vals: ", joint_vals)
# urdfpy_robot = urdfpy.URDF.load(urdf_path)

# vis = VizServer()
# vis.view_robot(
#     urdfpy_robot, joint_names, joint_vals, "obj_robot", 0x00FF00, mesh_type="visual"
# )
# vis.view_robot(
#     urdfpy_robot,
#     joint_names,
#     joint_vals,
#     "obj_robot_col",
#     0x0000FF,
#     mesh_type="collision",
# )
