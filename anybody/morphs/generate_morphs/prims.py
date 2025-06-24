import os
import numpy as np
import urdfpy
from collections import OrderedDict

from anybody.utils.vis_server import VizServer
from anybody.utils.path_utils import get_tmp_mesh_storage, get_robot_morphs_dir
from anybody.utils.utils import format_str
from .utils import update_urdf


base_link = """
    <link name="base_link">
        <inertial>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <mass value="0.1"/>
            <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
        </inertial>
    </link>
"""

dummy_links = """
    <link name="link_LINKNAME_px"/>
    <link name="link_LINKNAME_py"/>
    <link name="link_LINKNAME_pz"/>
    <link name="link_LINKNAME_rz"/>
"""

planar_link =  """   
    <link name="link_LINKNAME_ry">
        <inertial>
            <origin xyz="0 0 -HALFLENGTH" rpy="0 0 0"/>
            <mass value="0.1"/>
            <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
        </inertial>
        <visual>
            <origin xyz="0 0 -HALFLENGTH" rpy="0 0 0"/>
            <geometry>
                <box size="0.01 LENGTH LENGTH"/>
            </geometry>
            <material name="blue">
                <color rgba="0 0 1 1"/>
            </material>
        </visual>
        <collision>
            <origin xyz="0 0 -HALFLENGTH" rpy="0 0 0"/>
            <geometry>
                <box size="0.01 LENGTH LENGTH"/>
            </geometry>
        </collision>
    </link>
"""


spherical_link = """
    <link name="link_LINKNAME_ry">
        <inertial>
            <origin xyz="0 0 -HALFLENGTH" rpy="0 0 0"/>
            <mass value="0.1"/>
            <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
        </inertial>
        <visual>
            <origin xyz="0 0 -HALFLENGTH" rpy="0 0 0"/>
            <geometry>
                <sphere radius="HALFLENGTH"/>
            </geometry>
            <material name="blue">
                <color rgba="0 0 1 1"/>
            </material>
        </visual>
        <collision>
            <origin xyz="0 0 -HALFLENGTH" rpy="0 0 0"/>
            <geometry>
                <sphere radius="HALFLENGTH"/>
            </geometry>
        </collision>
    </link>
"""

cylindrical_link = """
    <link name="link_LINKNAME_ry">
        <inertial>
            <origin xyz="0 0 -HALFLENGTH" rpy="0 0 0"/> 
            <mass value="0.1"/>
            <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
        </inertial>
        <visual>
            <origin xyz="0 0 -HALFLENGTH" rpy="0 0 0"/> 
            <geometry>
                <cylinder radius="0.005" length="LENGTH"/>
            </geometry>
            <material name="blue">
                <color rgba="0 0 1 1"/>
            </material>
        </visual>
        <collision>
            <origin xyz="0 0 -HALFLENGTH" rpy="0 0 0"/>
            <geometry>
                <cylinder radius="0.005" length="LENGTH"/>
            </geometry>
        </collision>
    </link>
"""

cuboidal_link = """
    <link name="link_LINKNAME_ry">
        <inertial>
            <origin xyz="0 0 -HALFLENGTH" rpy="0 0 0"/> 
            <mass value="0.1"/>
            <inertia ixx="0.001" ixy="0.0" ixz="0.0" iyy="0.001" iyz="0.0" izz="0.001"/>
        </inertial>
        <visual>
            <origin xyz="0 0 -HALFLENGTH" rpy="0 0 0"/>
            <geometry>
                <box size="LENGTH LENGTH LENGTH"/>
            </geometry>
            <material name="blue">
                <color rgba="0 0 1 1"/>
            </material>
        </visual>
        <collision>
            <origin xyz="0 0 -HALFLENGTH" rpy="0 0 0"/>
            <geometry>
                <box size="LENGTH LENGTH LENGTH"/>
            </geometry>
        </collision>
    </link>
"""

   
final_link = """
    <link name="link_LINKNAME_end">
        <!-- sphere -->
        <inertial>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <mass value="0.1"/>
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


joint_5dof = """
    <joint name="joint_LINKNAME_px" type="prismatic">
        <parent link="base_link"/>
        <child link="link_LINKNAME_px"/>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <axis xyz="1 0 0"/>
        <limit lower="-100" upper="100" effort="1" velocity="1"/>
    </joint>
    
    <joint name="joint_LINKNAME_py" type="prismatic">
        <parent link="link_LINKNAME_px"/>
        <child link="link_LINKNAME_py"/>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <axis xyz="0 1 0"/>
        <limit lower="-100" upper="100" effort="1" velocity="1"/>
    </joint>
    
    <joint name="joint_LINKNAME_pz" type="prismatic">
        <parent link="link_LINKNAME_py"/>
        <child link="link_LINKNAME_pz"/>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-100" upper="100" effort="1" velocity="1"/>
    </joint>

    <joint name="joint_LINKNAME_rz" type="revolute">
        <parent link="link_LINKNAME_pz"/>
        <child link="link_LINKNAME_rz"/>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
    </joint>

    <joint name="joint_LINKNAME_ry" type="revolute">
        <parent link="link_LINKNAME_rz"/>
        <child link="link_LINKNAME_ry"/>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <axis xyz="0 1 0"/>
        <limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
    </joint>

    <joint name="joint_LINKNAME_end" type="fixed">
        <parent link="link_LINKNAME_ry"/>
        <child link="link_LINKNAME_end"/>
        <origin xyz="0 0 -LENGTH" rpy="0 0 0"/>
    </joint>
"""


def get_urdf(link_type="planar"):
    if link_type == "planar":
        prim_link = planar_link.replace("LINKNAME", "0")
    elif link_type == "spherical":
        prim_link = spherical_link.replace("LINKNAME", "0")
    elif link_type == "cylindrical":
        prim_link = cylindrical_link.replace("LINKNAME", "0")
    elif link_type == "cuboidal":
        prim_link = cuboidal_link.replace("LINKNAME", "0")
        
    final_urdf = f"""<?xml version="1.0"?>
<robot name="prim_robot">
{base_link}
{dummy_links.replace("LINKNAME", "0")}
{prim_link}
{final_link.replace("LINKNAME", "0")}
{joint_5dof.replace("LINKNAME", "0")}
</robot>
"""

    return final_urdf


def save_urdfs(to_usd=False, link_type="all", randomized=False):
    
    if link_type == "all":
        save_urdfs(to_usd, "planar", randomized)
        save_urdfs(to_usd, "cuboidal", randomized)
        save_urdfs(to_usd, "cylindrical", randomized)
        save_urdfs(to_usd, "spherical", randomized)
        return
    
    
    if link_type not in ['planar', 'cuboidal', 'cylindrical', 'spherical']:
        raise ValueError(f"Unknown link type: {link_type}")

    if link_type in ['planar', 'cyindrical']:
        if not randomized:
            lengths = np.linspace(0.4, 0.8, 4)
        else:
            lengths = np.linspace(0.4, 0.8, 25)
        
    else:
        # spherical and cuboidal
        if not randomized:
            lengths = np.linspace(0.2, 0.4, 3)
        else:
            lengths = np.linspace(0.2, 0.4, 25)    


    if not randomized:
        names = [f"{link_type}_{i}" for i in range(len(lengths))]
    else:
        names = [f"{link_type[:2]}{i}" for i in range(len(lengths))]

    joint_names = [
        "joint_0_px",
        "joint_0_py",
        "joint_0_pz",
        "joint_0_rz",
        "joint_0_ry",
    ]
    jvals_low = [-0.5, -0.5, 0.2, -2 * np.pi, -2 * np.pi]
    jvals_high = [0.5, 0.5, 0.8, 2 * np.pi, 2 * np.pi]
    # j_effort = [50.0, 50.0, 50.0, 50.0, 50.0]
    # j_vels = [5.0, 5.0, 5.0, 5.0, 5.0]
    j_vels = [10.0, 10.0, 10.0, 10.0, 10.0]
    j_effort = [500.0, 500.0, 500.0, 500.0, 500.0]    

    joint_updates = [
        {"name": j, "lower": lb, "upper": ub, "effort": e, "velocity": v}
        for j, lb, ub, e, v in zip(joint_names, jvals_low, jvals_high, j_effort, j_vels)
    ]

    for i, length in enumerate(lengths):
        print("Creating URDF for ", names[i])
        robo_dir = get_robot_morphs_dir() / "prims" / f"{names[i]}"
        robo_dir.mkdir(parents=True, exist_ok=True)
        urdf_path = robo_dir / f"{names[i]}.urdf"

        urdf = format_str(
            get_urdf(link_type),
            OrderedDict([("HALFLENGTH", f"{length / 2.}"), ("LENGTH", str(length))]),
        )

        urdf = update_urdf(urdf, joint_updates)


        with open(urdf_path, "w") as f:
            f.write(urdf)

        if to_usd:
            usd_path = robo_dir / f"{names[i]}.usd"
            if not os.path.exists(usd_path):
                from anybody.utils.to_usd import ArgsCli, main
                args_cli = ArgsCli(input=str(urdf_path), output=str(usd_path), headless=True)
                main(args_cli)



if __name__ == "__main__":
    
    # save_urdfs(to_usd=False, link_type="planar")
    # save_urdfs(to_usd=False, link_type="cuboidal")
    # save_urdfs(to_usd=False, link_type="cylindrical")
    # save_urdfs(to_usd=False, link_type="spherical")
    
    # planar urdf paths
    name = "planar_0"
    
    urdf_path = get_robot_morphs_dir() / "prims" / f"{name}" / f"{name}.urdf"

    urdfpy_robot = urdfpy.URDF.load(urdf_path)
    vis = VizServer()

    for _ in range(20):
        joint_names = [
            "joint_0_px",
            "joint_0_py",
            "joint_0_pz",
            "joint_0_rz",
            "joint_0_ry",
        ]
        joint_vals = np.random.uniform(-0.5, 0.5, len(joint_names))
        # joint_vals = [0., 0., .5, 1.52, 0.5]
        joint_vals[:3] = [0, 0.0, 0.5]

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
