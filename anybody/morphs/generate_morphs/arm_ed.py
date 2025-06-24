import os
import numpy as np
import urdfpy

from anybody.utils.vis_server import VizServer
from anybody.utils.path_utils import get_tmp_mesh_storage, get_robot_morphs_dir
from .utils import get_3dof_base_link, update_joint_params

from .arm_ur5 import get_L_link, get_ee_link



def get_cylinder_link(
    linkname,
    origin=(0, 0, 0),
    length=0.05,
    radius=0.05,
    color=[0, 0, 1],
):
    collision_margin = 0.005
    return f"""
    <link name="{linkname}">
        <inertial>
            <origin xyz="{origin[0]} {origin[1]} {origin[2]}" rpy="0 0 0"/>
            <mass value="1.0"/>
            <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
        </inertial>
        <visual>
            <origin xyz="{origin[0]} {origin[1]} {origin[2]}" rpy="0 0 0"/>
            <geometry>
                <cylinder length="{length}" radius="{radius}"/>
            </geometry>
            <material name="blue">
                <color rgba="{color[0]} {color[1]} {color[2]} 1"/>   
            </material>
        </visual>
        <collision>
            <origin xyz="{origin[0]} {origin[1]} {origin[2]}" rpy="0 0 0"/>
            <geometry>
                <cylinder length="{length - collision_margin}" radius="{radius}"/>
            </geometry>
        </collision>
    </link>
    """
    

def get_S_link(link_idx, l1, l2, rad, l3=None):
    
    if l3 is None:
        l3 = l2
    
    collision_margin = 0.005

    # need to add a segment 0 and another corner link
    additional_segment = f"""
    <link name="segment_{link_idx}_0">
        <visual>
        <geometry>
            <cylinder length="{l3}" radius="{rad}" />
        </geometry>
        <origin xyz="0 {l3/2} 0" rpy="{np.pi/2} 0 0"/>
        <material name="blue">
            <color rgba="0 0 1 1"/>
        </material>
        </visual>
        <collision>
            <geometry>
                <cylinder length="{l3 - collision_margin}" radius="{rad}" />
            </geometry>
            <origin xyz="0 {l3/2} 0" rpy="{np.pi/2} 0 0"/>
        </collision>
        <inertial>
            <mass value="1.0"/>
            <inertia ixx="1.0" iyy="1.0" izz="1.0" ixy="0.0" ixz="0.0" iyz="0.0"/>
            <origin xyz="0 {l3/2} 0" rpy="{np.pi/2} 0 0"/>
        </inertial>
    </link>
    
  <!-- A box to represent the corner connection -->
  <link name="corner_{link_idx}_0">
    <visual>
      <geometry>
        <box size="{rad*2} {rad*2} {rad*2}"/>
      </geometry>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <material name="green">
        <color rgba="0 1 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="{rad*2 - collision_margin} {rad*2 - collision_margin} {rad*2 - collision_margin}"/>
      </geometry>
      <origin xyz="0 0 0" rpy="0 0 0"/>
    </collision>
    <inertial>
      <mass value="0.1"/>
      <inertia ixx="0.1" iyy="0.1" izz="0.1" ixy="0.0" ixz="0.0" iyz="0.0"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
    </inertial>
  </link>
  
  <joint name="corner_joint_{link_idx}_0" type="fixed">
    <parent link="segment_{link_idx}_0"/>
    <child link="corner_{link_idx}_0"/>
    <origin xyz="0 {l3 + rad} 0" rpy="0 0 {np.pi}"/>
  </joint>
  
  <joint name="corner_joint_{link_idx}_0_1" type="fixed">
    <parent link="corner_{link_idx}_0"/>
    <child link="segment_{link_idx}_1"/>
    <origin xyz="0 0 {rad}" rpy="0 0 {np.pi}"/> 
  </joint>
  
"""

    l_link, _ = get_L_link(link_idx, l1, l2, rad)
    c_link = additional_segment + l_link
    info = {"start": f"segment_{link_idx}_0", "end": f"segment_{link_idx}_2"}

    return c_link, info


def get_urdf(v1=0.07, v2=0.03, v3=0.3, v4=0.2, v5=0.1, v6=0.05, margin=0.01, rad=0.03, ee_rad=0.02):
    
    l0 = """
    <link name="base_link"/>
    """
    
    l1, cur_info = get_L_link(1, v1, v2 / 2, rad)
    _l1 = get_3dof_base_link(cur_info['start'])
    # _l1 = get_fixed_base_link(cur_info['start'])

    prev_info = cur_info

    l2, cur_info = get_S_link(2, v3, v2/2, rad)
    joint_1_2 = f"""
    <joint name="joint_1_2" type="revolute">
        <parent link="{prev_info['end']}"/>
        <child link="{cur_info['start']}"/>
        <origin xyz="0 {v2/2} 0" rpy="0 0 0"/>
        <axis xyz="0 1 0"/>
        <limit lower="{-np.pi/2}" upper="{np.pi/2}" effort="100" velocity="0.1"/>
    </joint>
    """
    prev_info = cur_info
    
    l3, cur_info = get_S_link(3, v4, margin, rad, l3=v2/2)

    joint_2_3 = f"""
    <joint name="joint_2_3" type="revolute">
        <parent link="{prev_info['end']}"/>
        <child link="{cur_info['start']}"/>
        <origin xyz="0 {v2/2} 0" rpy="0 0 0"/>
        <axis xyz="0 1 0"/>
        <limit lower="{-np.pi/2}" upper="{np.pi/2}" effort="100" velocity="0.1"/>
    </joint>
    """
    
    prev_info = cur_info
    
    l4 = get_cylinder_link(linkname="wrist", length = v5, radius = ee_rad, origin=(0, 0, -rad))

    joint_3_4 = f"""
    <joint name="joint_3_4" type="revolute">
        <parent link="{prev_info['end']}"/>
        <child link="wrist"/>
        <origin xyz="0 {margin + ee_rad} 0" rpy="0 {np.pi/2} 0"/>
        <axis xyz="0 1 0"/>
        <limit lower="{-np.pi/2}" upper="{np.pi/2}" effort="100" velocity="0.1"/>
    </joint>
    """
    
    ee_link = get_ee_link(
        parent_linkname="wrist",
        ee_length=v6,
        joint_origin=(0, 0, -v5 -rad),
        joint_rpy=(0, 0, np.pi / 2),
        displacement=ee_rad * 0.95,
    )
    
    urdf = f"""<?xml version="1.0"?>
<robot name="ed">
{l0}
{l1}
{_l1}
{l2}
{l3}
{l4}
{joint_1_2}
{joint_2_3}
{joint_3_4}
{ee_link}
</robot>
"""
    return urdf



def save_urdfs(to_usd=False, randomized=False):
    vals1 = {
        "v1": 0.07, 
        "v2": 0.03,
        "v3": 0.3,
        "v4": 0.2,
        "v5": 0.1,
        "v6": 0.05,
        "rad": 0.03,
        "ee_rad": 0.03 * 0.7,
        "margin": 0.01,
    }
    vals2 = {
        "v1": 0.1, 
        "v2": 0.03,
        "v3": 0.2,
        "v4": 0.15,
        "v5": 0.1,
        "v6": 0.08,
        "rad": 0.03,
        "ee_rad": 0.03 * 0.7,
        "margin": 0.02,
    }
    vals3 = {
        "v1": 0.15, 
        "v2": 0.07,
        "v3": 0.5,
        "v4": 0.25,
        "v5": 0.2,
        "v6": 0.1,
        "rad": 0.05,
        "ee_rad": 0.05 * 0.7,
        "margin": 0.01,
    }


    value_ranges = {
        "v1": [0.07, 0.2],
        "v2": [0.03, 0.1],
        "v3": [0.2, 0.5],
        "v4": [0.15, 0.3],
        "v5": [0.1, 0.25],
        "v6": [0.05, 0.15],
        "rad": [0.03, 0.06],
        "ee_rad": [0.02, 0.05],
        "margin": [0.01, 0.02],
    }


    if not randomized:

        for idx, val in enumerate([vals1, vals2, vals3]):
            my_urdf = get_urdf(**val)
            
            robo_name = f"arm_ed_{idx}"
            robo_dir = get_robot_morphs_dir() / "arm_ed" / robo_name
            robo_dir.mkdir(parents=True, exist_ok=True)

            urdf_path = robo_dir / f"{robo_name}.urdf"

            with open(urdf_path, "w") as f:
                f.write(my_urdf)

            my_urdf = update_joint_params(urdf_path)

            if to_usd:
                usd_path = robo_dir / f"{robo_name}.usd"
                if not os.path.exists(usd_path):
                    from anybody.utils.to_usd import ArgsCli, main
                    args_cli = ArgsCli(input=str(urdf_path), output=str(usd_path), headless=True)
                    main(args_cli)
    
    else:
        
        for idx in range(100):
            
            # sample values
            vals = {k: np.random.uniform(v[0], v[1]) for k, v in value_ranges.items()}
            my_urdf = get_urdf(**vals)
            
            robo_name = f"r{idx}"
            robo_dir = get_robot_morphs_dir() / "arm_ed" / robo_name
            robo_dir.mkdir(parents=True, exist_ok=True)
            
            urdf_path = robo_dir / f"{robo_name}.urdf"
            
            with open(urdf_path, "w") as f:
                f.write(my_urdf)
                
            my_urdf = update_joint_params(urdf_path)
            
            if to_usd:
                usd_path = robo_dir / f"{robo_name}.usd"
                if not os.path.exists(usd_path):
                    from anybody.utils.to_usd import ArgsCli, main
                    args_cli = ArgsCli(input=str(urdf_path), output=str(usd_path), headless=True)
                    main(args_cli)
                                    



if __name__ == "__main__":

    vals1 = {
        "v1": 0.07, 
        "v2": 0.03,
        "v3": 0.3,
        "v4": 0.2,
        "v5": 0.1,
        "v6": 0.05,
        "rad": 0.03,
        "ee_rad": 0.03 * 0.7,
        "margin": 0.01,
    }
    vals2 = {
        "v1": 0.1, 
        "v2": 0.03,
        "v3": 0.2,
        "v4": 0.15,
        "v5": 0.1,
        "v6": 0.08,
        "rad": 0.03,
        "ee_rad": 0.03 * 0.7,
        "margin": 0.02,
    }
    vals3 = {
        "v1": 0.15, 
        "v2": 0.07,
        "v3": 0.5,
        "v4": 0.25,
        "v5": 0.2,
        "v6": 0.1,
        "rad": 0.05,
        "ee_rad": 0.05 * 0.7,
        "margin": 0.01,
    }
    
    
    my_urdf = get_urdf(**vals3)

    urdf_path = get_tmp_mesh_storage() / "ed.urdf"
    with open(urdf_path, "w") as f:
        f.write(my_urdf)

    urdfpy_robot = urdfpy.URDF.load(urdf_path)
    vis = VizServer()

    joint_names = [j.name for j in urdfpy_robot.joints if j.joint_type != 'fixed']

    # joint_names = ["joint_3_4", "joint_ee_left", "joint_ee_right"]

    jlimits = {j.name: j.limit for j in urdfpy_robot.joints}
    joint_lb = [jlimits[jname].lower for jname in joint_names]
    joint_ub = [jlimits[jname].upper for jname in joint_names]

    for _ in range(20):
        # joint_vals = np.zeros(len(joint_names))
        # joint_vals = np.random.uniform(joint_lb, joint_ub)
        # joint_vals[:3] = [0, 0.0, 0.0]

        joint_names = ['joint_1_2', 'joint_2_3', 'joint_3_4', 'joint_ee_left', 'joint_ee_right']
        joint_vals = [ 1.5708,  1.5708,  1.5708, -0.0050,  0.0199]        

        # clip joint values to joint limits
        joint_vals = np.clip(joint_vals, joint_lb, joint_ub)


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
