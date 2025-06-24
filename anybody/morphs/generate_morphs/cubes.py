import os, sys

sys.path.append(os.environ["PROJECT_PATH"])

import numpy as np
import urdfpy
from collections import OrderedDict

from anybody.utils.vis_server import VizServer
from anybody.utils.path_utils import get_tmp_mesh_storage, get_robot_morphs_dir
from anybody.utils.utils import format_str
import sys

from .utils import update_urdf

base_link = """
    <link name="base_link">
    <inertial>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <mass value="1.0"/>
        <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
    </link>
"""

cube_link = """
    <!-- Link LINKNAME -->
    
    <link name="link_LINKNAME_dummy">
    </link>

    <link name="link_LINKNAME">
        <inertial>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <mass value="1.0"/>
            <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
        </inertial>
        <visual>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <box size="SIZE SIZE SIZE"/>
            </geometry>
            <material name="blue">
                <color rgba="0 0 1 1"/>
            </material>
        </visual>
        <collision>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <box size="COLLISIONSIZE COLLISIONSIZE COLLISIONSIZE"/>
            </geometry>
        </collision>
    </link>
"""

obj_joint = """
    <!-- Joint OBJIDX -->
    <joint name="joint_OBJIDX_x" type="prismatic">
        <parent link="base_link"/>
        <child link="link_OBJIDX_dummy"/>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <axis xyz="1 0 0"/>
        <limit lower="-100" upper="100" effort="1" velocity="1"/>
    </joint>

    <joint name="joint_OBJIDX_y" type="prismatic">
        <parent link="link_OBJIDX_dummy"/>
        <child link="link_OBJIDX"/>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <axis xyz="0 1 0"/>
        <limit lower="-100" upper="100" effort="1" velocity="1"/>
    </joint>
"""


def create_obj_planar(n, size=0.1):
    assert n >= 1, "n must be greater than or equal to 1"

    links = base_link

    for link_idx in range(n):
        links += cube_link.replace("LINKNAME", str(link_idx))
        links += "\n"

    joints = ""
    for joint_idx in range(n):
        joints += obj_joint.replace("OBJIDX", str(joint_idx))
        joints += "\n"

    n_link_urdf = f"""<?xml version="1.0"?>
<robot name="n_link_robot">
{links}
{joints}
</robot>
"""
    n_link_urdf = format_str(
        n_link_urdf,
        OrderedDict(
            [
                ("COLLISIONSIZE", f"{size}"),
                ("HALFSIZE", f"{size/2}"),
                ("SIZE", f"{size}"),
            ]
        ),
    )

    joint_names = [f"joint_{i}_x" for i in range(n)] + [
        f"joint_{i}_y" for i in range(n)
    ]
    joint_lower = [-1.0] * n + [-1.0] * n
    joint_upper = [1.0] * n + [1.0] * n
    joint_effort = [50] * 2 * n
    joint_velocity = [5] * 2 * n
    joint_updates = [
        {"name": j, "lower": lb, "upper": ub, "effort": e, "velocity": v}
        for j, lb, ub, e, v in zip(
            joint_names, joint_lower, joint_upper, joint_effort, joint_velocity
        )
    ]
    n_link_urdf = update_urdf(n_link_urdf, joint_updates)

    return n_link_urdf


def save_urdfs(to_usd=False, randomized=False):
    # n_cubes = [1, 2, 3, 4, 5]
    # sizes = np.random.uniform(0.05, 0.2, 2 * len(n_cubes))
    
    n = 100 if randomized else 5    
    if randomized:
        sizes = np.linspace(0.02, 0.2, n)
    else:
        sizes = np.linspace(0.05, 0.1, n)
    

    for idx in range(n):
        
        robo_name = f"cube_{idx}" if not randomized else f"r{idx}"
        
        robo_dir = get_robot_morphs_dir() / "cubes" / robo_name
        robo_dir.mkdir(parents=True, exist_ok=True)

        urdf = create_obj_planar(1, size=sizes[idx])

        urdf_path = robo_dir / f"{robo_name}.urdf"
        with open(urdf_path, "w") as f:
            f.write(urdf)

        if to_usd:
            usd_path = robo_dir / f"{robo_name}.usd"
            if not os.path.exists(usd_path):
                from anybody.utils.to_usd import ArgsCli, main
                args_cli = ArgsCli(input=str(urdf_path), output=str(usd_path), headless=True)
                main(args_cli)



# if __name__ == "__main__":

#     # create_urdfs()
#     # exit()


#     if len(sys.argv) > 1:
#         n = int(sys.argv[1])
#     else:
#         n = 1

#     obj_urdf = create_obj_planar(n)
#     get_tmp_mesh_storage().mkdir(parents=True, exist_ok=True)
#     urdf_path = get_tmp_mesh_storage() / f"{n}_cube.urdf"

#     with open(urdf_path, "w") as f:
#         f.write(obj_urdf)

#     print("obj.urdf file created successfully!")
#     # joint_vals = [np.random.uniform(-5, 5) for _ in range(2)]

#     joint_names = []
#     for i in range(n):
#         joint_names.append(f"joint_{i}_x")
#         joint_names.append(f"joint_{i}_y")

#     joint_vals = np.random.uniform(-1, 1, len(joint_names))

#     vis = VizServer()
#     vis.view_robot(
#         urdfpy.URDF.load(urdf_path),
#         joint_names,
#         joint_vals,
#         "obj_robot",
#         0x00FF00,
#         mesh_type="visual",
#     )
#     vis.view_robot(
#         urdfpy.URDF.load(urdf_path),
#         joint_names,
#         joint_vals,
#         "obj_robot_col",
#         0x0000FF,
#         mesh_type="collision",
#     )
