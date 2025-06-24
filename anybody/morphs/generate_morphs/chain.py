import os
import numpy as np
import urdfpy

from anybody.utils.vis_server import VizServer
from anybody.utils.path_utils import get_tmp_mesh_storage, get_robot_morphs_dir
from anybody.utils.utils import format_str
from .utils import update_urdf, update_joint_params, get_5dof_base_link


# bead link

def get_urdf(n_beads=4, radius=0.01, length=0.001, cylinder_radius=0.001):
    margin = 0.0001

    base_link = f"""
    <link name="base_link"/>
    
    <link name="movable_base_link">
        <inertial>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <mass value="1.0"/>
            <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
        </inertial>
        <visual>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <sphere radius="{radius}"/>
            </geometry>
            <material name="blue">
                <color rgba="0 0 1 1"/>
            </material>
        </visual>
        <collision>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <sphere radius="{radius}"/>
            </geometry>
        </collision>
    </link>
    """

    pos_only_base_movement = get_5dof_base_link(base_link_name="movable_base_link", pos_only=True)


    bead_link = f"""
    
    <link name="link_LINKNAME_cylinder_revx"/>
    
    <link name="link_LINKNAME_cylinder">
        <inertial>
            <origin xyz="{radius + length / 2 + margin} 0 0" rpy="0 {np.pi/2} 0"/>
            <mass value="1.0"/>
            <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
        </inertial>
        <visual>
            <origin xyz="{radius + length / 2 + margin} 0 0" rpy="0 {np.pi/2} 0"/>
            <geometry>
                <cylinder radius="{cylinder_radius}" length="{length}"/>
            </geometry>
        </visual>
        <collision>
            <origin xyz="{radius + length / 2 + margin} 0 0" rpy="0 {np.pi/2} 0"/>
            <geometry>
                <cylinder radius="{cylinder_radius}" length="{length}"/>
            </geometry>
        </collision>
    </link>
    
    <link name="link_LINKNAME_sphere">
        <inertial>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <mass value="1.0"/>
            <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
        </inertial>
        <visual>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <sphere radius="{radius}"/>
            </geometry>
        </visual>
        <collision>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <sphere radius="{radius}"/>
            </geometry>
        </collision>    
    </link>
    
    <joint name="joint_LINKNAME_revx" type="continuous">
        <parent link="PARENT_LINK"/>
        <child link="link_LINKNAME_cylinder_revx"/>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <axis xyz="1 0 0"/>
        <limit effort="50" velocity="5"/>
    </joint>
    
    <joint name="joint_LINKNAME_revz" type="continuous">
        <parent link="link_LINKNAME_cylinder_revx"/>
        <child link="link_LINKNAME_cylinder"/>  
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit effort="50" velocity="5"/>
    </joint>
    
    <joint name="joint_LINKNAME_x" type="fixed">
        <parent link="link_LINKNAME_cylinder"/>
        <child link="link_LINKNAME_sphere"/>
        <origin xyz="{2 * radius + length + margin} 0 0" rpy="0 0 0"/>
    </joint>
    
"""
        
    links_and_joints = base_link + pos_only_base_movement
    
    parent_link_name = "movable_base_link"
    for i in range(n_beads):
        links_and_joints += bead_link.replace("LINKNAME", str(i)).replace("PARENT_LINK", parent_link_name)
        parent_link_name = f"link_{i}_sphere"
        
    urdf = f"""<?xml version="1.0"?>
<robot name="bead_chain">
{links_and_joints}
</robot>
"""

    return urdf



def save_urdfs(to_usd=False):
    
    n_beads = [2, 3, 4, 5, 6, 7]

    radius = 0.01
    lengths = np.linspace(0.001, 0.005, 5)
    cylinder_radius = 0.001
    

    for n in n_beads:

        # joint_names = [f"joint_{i}_revx" for i in range(n)] + [f"joint_{i}_revz" for i in range(n)]
        # joint_names += ["joint_base_px", "joint_base_py", "joint_base_pz"]

        # jvals_low = [-0.6, -0.6, 0.2, -2 * np.pi, -2 * np.pi]
        # jvals_high = [0.6, 0.6, 1.0, 2 * np.pi, 2 * np.pi]
        # j_effort = [50.0] * 2 * n + [50.0] * 3
        # j_vels = [5.0] * 2 * n + [5.0] * 3

        # joint_updates = [
        #     {"name": j, "effort": e, "velocity": v}
        #     for j, e, v in zip(joint_names, j_effort, j_vels)
        # ]

        for l_idx, length in enumerate(lengths):
            robo_name = f"chain_{n}b_{l_idx}"
            robo_dir = get_robot_morphs_dir() / "chain" / robo_name
            robo_dir.mkdir(parents=True, exist_ok=True)
            urdf_path = robo_dir / f"{robo_name}.urdf"

            chain_urdf = get_urdf(n_beads=n, radius=radius, length=length, cylinder_radius=cylinder_radius)

            
            
            with open(urdf_path, "w") as f:
                f.write(chain_urdf)

            chain_urdf = update_joint_params(urdf_path)

            if to_usd:
                usd_path = robo_dir / f"{robo_name}.usd"
                if not os.path.exists(usd_path):
                    from anybody.utils.to_usd import ArgsCli, main
                    args_cli = ArgsCli(input=str(urdf_path), output=str(usd_path), headless=True)
                    main(args_cli)





if __name__ == "__main__":
    
    # write urdf to file
    
    my_urdf = get_urdf(n_beads=10)
    
    urdf_path = get_tmp_mesh_storage() / "bead_chain.urdf"
    with open(urdf_path, "w") as f:
        f.write(my_urdf)
    
    urdfpy_robot = urdfpy.URDF.load(urdf_path)
    vis = VizServer()

    # get all joint names, and their limits, and sample random joint values
    
    joint_names = urdfpy_robot.joints
    
    # filter out the fixed joints
    
    joint_names = [j.name for j in joint_names if j.joint_type != "fixed"]    
    
    joint_lb = [-np.pi] * len(joint_names)
    joint_ub = [np.pi] * len(joint_names)
    
    # jlimits = {j.name: j.limit for j in urdfpy_robot.joints}
    # joint_lb = [jlimits[jname].lower for jname in joint_names]
    # joint_ub = [jlimits[jname].upper for jname in joint_names]

    for _ in range(20):

        joint_vals = np.zeros(len(joint_names))
        # joint_vals = np.random.uniform(joint_lb, joint_ub)

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
