import os
import numpy as np
import urdfpy

from anybody.utils.vis_server import VizServer
from anybody.utils.path_utils import get_tmp_mesh_storage, get_robot_morphs_dir
from anybody.utils.utils import format_str
from .utils import update_urdf, get_5dof_base_link, update_joint_params 

def get_flap_link(flap_length, flap_thickness, width, cylinder_radius, margin, collision_margin):
    
    flap_link = f"""
    <link name="FLAPNAME">
        <inertial>
            <origin xyz="{cylinder_radius + flap_length/2 + margin} 0 0" rpy="0 0 0"/>
            <mass value="1.0"/>
            <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
        </inertial>
        <visual>
            <origin xyz="{cylinder_radius + flap_length/2 + margin} 0 0" rpy="0 0 0"/>
            <geometry>
                <box size="{flap_length} {flap_thickness} {width}"/>
            </geometry>
        </visual>
        <collision>
            <origin xyz="{cylinder_radius + flap_length/2 + margin} 0 0" rpy="0 0 0"/>
            <geometry>
                <box size="{flap_length - collision_margin} {flap_thickness} {width}"/>
            </geometry>
        </collision>
    </link>
"""

    return flap_link


def get_urdf(width=0.01, flap_length=0.05, flap_thickness=0.001, bend_theta=np.pi/6, 
             flap_length2=0.1):

    # cylindrical base link
    cylinder_radius = 0.001
    margin = 0.0001
    open_margin = np.pi/6
    smaller_flap_length = min(flap_length / 2, 0.05)
    fixed_theta = np.pi/6
    collision_margin = 0.005
    
    world_link = """
    <link name="base_link"/>
"""
    
    base_link = f"""
    <link name="cylinder_baselink">
        <inertial>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <mass value="1.0"/>
            <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
        </inertial>
        <visual>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <cylinder radius="{cylinder_radius}" length="{width}"/>
            </geometry>
            <material name="blue">
                <color rgba="0 0 1 1"/>
            </material>
        </visual>
        <collision>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <cylinder radius="{cylinder_radius}" length="{width}"/>
            </geometry>
        </collision>
    </link>
"""
    dof5_movement = get_5dof_base_link(base_link_name="cylinder_baselink")

    # flap is a flat cuboid of width 'width' and length 'flap_length' and thickness 'flap_thickness'
    # it shares the width axis with the cylinder base link  
    

    flap_fixed = f"""
    <link name="FLAPNAME">
        <inertial>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <mass value="1.0"/>
            <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
        </inertial>
        <visual>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <box size="{smaller_flap_length} {flap_thickness} {width}"/>
            </geometry>
        </visual>
        <collision>
            <origin xyz="0 0 0" rpy="0 0 0"/>   
            <geometry>
                <box size="{smaller_flap_length} {flap_thickness} {width}"/>
            </geometry>
        </collision>
    </link>
"""
    

    flap_joint1 = """
    <joint name="joint_FLAPNAME_rev" type="revolute">
        <parent link="PARENT"/>
        <child link="FLAPNAME"/>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="LOWER_THETA" upper="UPPER_THETA" effort="1" velocity="1"/>
    </joint>
"""

    flap_joint1_2 = f"""
    <joint name="joint_FLAPNAME_rev" type="revolute">
        <parent link="PARENT"/>
        <child link="FLAPNAME"/>
        <origin xyz="{cylinder_radius + flap_length + margin} 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="LOWER_THETA" upper="UPPER_THETA" effort="1" velocity="1"/>
    </joint>
"""

    flap_joint2 = f"""
    <joint name="joint_FLAPNAME" type="fixed">   
        <parent link="PARENT"/>
        <child link="FLAPNAME"/>
        <origin xyz="{flap_length2 + cylinder_radius + margin} 0 0" rpy="0 0 THETA"/>
    </joint>
"""

    flap_link = get_flap_link(flap_length, flap_thickness, width, cylinder_radius, margin, collision_margin)

    flap_left1 = format_str(flap_link + flap_joint1, {
        'FLAPNAME': 'flap_left1',
        'PARENT': 'cylinder_baselink',
        'LOWER_THETA': str(bend_theta),
        'UPPER_THETA': str(bend_theta + open_margin)
    })
        
    flap_right1 = format_str(flap_link + flap_joint1, {
        'FLAPNAME': 'flap_right1',
        'PARENT': 'cylinder_baselink',
        'LOWER_THETA': str(-bend_theta - open_margin),
        'UPPER_THETA': str(-bend_theta) 
    })
    
    flap_link2 = get_flap_link(flap_length2, flap_thickness, width, cylinder_radius, margin, collision_margin)  
    
    flap_left2 = format_str(flap_link2 + flap_joint1_2, {
        'FLAPNAME': 'flap_left2',
        'PARENT': 'flap_left1',
        'LOWER_THETA': str(- bend_theta - open_margin - fixed_theta),
        'UPPER_THETA': str(- bend_theta - fixed_theta)
    })

    flap_right2 = format_str(flap_link2 + flap_joint1_2, {
        'FLAPNAME': 'flap_right2',
        'PARENT': 'flap_right1',
        'LOWER_THETA': str(bend_theta + fixed_theta),
        'UPPER_THETA': str(bend_theta + open_margin + fixed_theta)
    })
    
    flap_left_fixed = format_str(flap_fixed + flap_joint2, {
        'FLAPNAME': 'flap_left_fixed',
        'PARENT': 'flap_left2',
        'THETA': str(fixed_theta)
    })
    
    flap_right_fixed = format_str(flap_fixed + flap_joint2, {
        'FLAPNAME': 'flap_right_fixed',
        'PARENT': 'flap_right2',
        'THETA': str(-fixed_theta)
    })
    
    
    urdf = f"""<?xml version="1.0"?>
<robot name="tongs">
{world_link}
{dof5_movement}
{base_link}
{flap_left1}
{flap_right1}
{flap_left2}
{flap_right2}
{flap_left_fixed}
{flap_right_fixed}
</robot>
"""
    return urdf



def save_urdfs(to_usd=False):
    
    lengths1 = np.linspace(0.05, 0.1, 3)
    lengths2 = np.linspace(0.1, 0.05, 3)

    # joint_names = ['joint_cylinder_baselink_px', 
    #                 'joint_cylinder_baselink_py',
    #                 'joint_cylinder_baselink_pz',
    #                 'joint_cylinder_baselink_rz',
    #                 'joint_cylinder_baselink_ry',
    #                 'joint_flap_left1_rev',
    #                 'joint_flap_right1_rev',
    #                 'joint_flap_left2_rev',
    #                 'joint_flap_right2_rev',
    #             ]
    
    # jvals_low = [-0.6, -0.6, 0.2, -2 * np.pi, -2 * np.pi]
    # jvals_high = [0.6, 0.6, 1.0, 2 * np.pi, 2 * np.pi]
    # j_effort = [50.0] * 5 + [50.0] * 4
    # j_vels = [5.0] * 5 + [5.0] * 4

    # joint_updates = [
    #     {"name": j, "effort": e, "velocity": v}
    #     for j, e, v in zip(joint_names, j_effort, j_vels)
    # ]

    for l_idx, (l1, l2) in enumerate(zip(lengths1, lengths2)):

        robo_name = f"tongs_{l_idx}"
        robo_dir = get_robot_morphs_dir() / "tongs_v1" / robo_name
        robo_dir.mkdir(parents=True, exist_ok=True)
        urdf_path = robo_dir / f"{robo_name}.urdf"

        urdf = get_urdf(flap_length=l1, flap_length2=l2, bend_theta=np.pi/18)
        # urdf = update_urdf(tongs_urdf, joint_updates)

        with open(urdf_path, "w") as f:
            f.write(urdf)

        urdf = update_joint_params(urdf_path)

        if to_usd:
            usd_path = robo_dir / f"{robo_name}.usd"
            if not os.path.exists(usd_path):
                from anybody.utils.to_usd import ArgsCli, main
                args_cli = ArgsCli(input=str(urdf_path), output=str(usd_path), headless=True)
                main(args_cli)



if __name__ == "__main__":
    
    # write urdf to file
    
    my_urdf = get_urdf(bend_theta=0.1)
    
    urdf_path = get_tmp_mesh_storage() / "tongs.urdf"
    with open(urdf_path, "w") as f:
        f.write(my_urdf)
    
    urdfpy_robot = urdfpy.URDF.load(urdf_path)
    vis = VizServer()

    # get all joint names, and their limits, and sample random joint values
    
    # joint_names = urdfpy_robot.joints
    
    # # filter out the fixed joints
    # 
    # joint_names = [j.name for j in joint_names if j.joint_type != "fixed"]    
    
    joint_names = ['joint_cylinder_baselink_px', 
                   'joint_cylinder_baselink_py',
                     'joint_cylinder_baselink_pz',
                        'joint_cylinder_baselink_rz',
                        'joint_cylinder_baselink_ry',
                        'joint_flap_left1_rev',
                        'joint_flap_right1_rev',
                        'joint_flap_left2_rev',
                        'joint_flap_right2_rev',
                   ]
    
    # joint_lb = [-np.pi] * len(joint_names)
    # joint_ub = [np.pi] * len(joint_names)
    
    jlimits = {j.name: j.limit for j in urdfpy_robot.joints}
    joint_lb = [jlimits[jname].lower for jname in joint_names]
    joint_ub = [jlimits[jname].upper for jname in joint_names]

    

    for _ in range(20):

        # joint_vals = np.zeros(len(joint_names))
        joint_vals = np.random.uniform(joint_lb, joint_ub)
        joint_vals[:3] = [0, 0.0, 0.0]

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
