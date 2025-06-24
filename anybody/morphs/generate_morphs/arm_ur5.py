import os
import numpy as np
import urdfpy

from anybody.utils.vis_server import VizServer
from anybody.utils.path_utils import get_tmp_mesh_storage, get_robot_morphs_dir
from .utils import update_joint_params


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
            <origin xyz="{0} {0} {length/2}" rpy="0 0 0"/>
            <mass value="1.0"/>
            <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
        </inertial>
        <visual>
            <origin xyz="{0} {0} {length/2}" rpy="0 0 0"/>
            <geometry>
                <cylinder length="{length}" radius="{radius}"/>
            </geometry>
            <material name="blue">
                <color rgba="{color[0]} {color[1]} {color[2]} 1"/>   
            </material>
        </visual>
        <collision>
            <origin xyz="{0} {0} {length/2}" rpy="0 0 0"/>
            <geometry>
                <cylinder length="{length - collision_margin}" radius="{radius}"/>
            </geometry>
        </collision>
    </link>
    """


def get_ee_link(
    parent_linkname,
    ee_length,
    joint_origin,
    joint_rpy,
    displacement=0.05,
    ee_width=0.02,
    ee_thickness=0.005,
):
    ee_link = f"""
        <link name="ee_link_SIDE">
            <inertial>
                <origin xyz="0 0 {ee_length/2}" rpy="0 0 0"/>
                <mass value="1.0"/>
                <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
            </inertial>
            <visual>
                <origin xyz="0 0 {ee_length / 2}" rpy="0 0 0"/>
                <geometry>
                    <box size="{ee_thickness} {ee_width} {ee_length}"/>
                </geometry>
                <material name="blue">
                    <color rgba="0 0 1 1"/>
                </material>
            </visual>
            <collision>
                <origin xyz="0 0 {ee_length / 2}" rpy="0 0 0"/>
                <geometry>
                    <box size="{ee_thickness} {ee_width} {ee_length}"/>
                </geometry>
            </collision>
        </link>     
    """
    ee_joint = f"""
        <joint name="joint_ee_SIDE" type="prismatic">
            <parent link="{parent_linkname}"/>
            <child link="ee_link_SIDE"/>
        <origin xyz="{joint_origin[0]} {joint_origin[1]} {joint_origin[2]}" rpy="{joint_rpy[0]} {joint_rpy[1]} {joint_rpy[2]}"/>
            <axis xyz="1 0 0"/>
            <limit lower="-0.05" upper="-0.005" effort="1" velocity="1"/>
        </joint>
    """

    ee_left_link = ee_link.replace("SIDE", "left")
    ee_left_joint = ee_joint.replace("SIDE", "left")     #.replace("LINKNAME", "link_8")
    ee_left_joint = ee_left_joint.replace(
        'lower="-0.05" upper="-0.005"', f'lower="-{displacement}" upper="-0.005"'
    )

    ee_right_link = ee_link.replace("SIDE", "right")
    ee_right_joint = ee_joint.replace("SIDE", "right")    #.replace("LINKNAME", "link_8")
    ee_right_joint = ee_right_joint.replace(
        'lower="-0.05" upper="-0.005"', f'lower="0.005" upper="{displacement}"'
    )

    return ee_left_link + ee_right_link + ee_left_joint + ee_right_joint


def get_L_link(link_idx, l1, l2, rad):
    collision_margin = 0.005
    assert l1 > collision_margin
    assert l2 > collision_margin

    link_urdf = f"""
  <link name="segment_{link_idx}_1">
    <visual>
      <geometry>
        <cylinder length="{l1}" radius="{rad}" />
      </geometry>
      <origin xyz="0 0 {l1/2}" rpy="0 0 0"/>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="{l1 - collision_margin}" radius="{rad}" />
      </geometry>
      <origin xyz="0 0 {l1/2}" rpy="0 0 0"/>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="1.0" iyy="1.0" izz="1.0" ixy="0.0" ixz="0.0" iyz="0.0"/>
      <origin xyz="0 0 {l1/2}" rpy="0 0 0"/>
    </inertial>
  </link>

  <!-- Second segment of the L-shape -->
  <link name="segment_{link_idx}_2">
    <visual>
      <geometry>
        <cylinder length="{l2}" radius="{rad}" />
      </geometry>
      <origin xyz="0 {l2/2} 0" rpy="{np.pi/2} 0 0"/>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder length="{l2 - collision_margin}" radius="{rad}" />
      </geometry>
      <origin xyz="0 {l2/2} 0" rpy="{np.pi/2} 0 0"/>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="1.0" iyy="1.0" izz="1.0" ixy="0.0" ixz="0.0" iyz="0.0"/>
      <origin xyz="0 {l2/2} 0" rpy="{np.pi/2} 0 0"/>
    </inertial>
  </link>

  <!-- A box to represent the corner connection -->
  <link name="corner_{link_idx}">
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

  <!-- Fixing the corner link to the second segment -->
  <joint name="corner_joint_{link_idx}_1" type="fixed">
    <parent link="segment_{link_idx}_1"/>
    <child link="corner_{link_idx}"/>
    <origin xyz="0 0 {l1 + rad}" rpy="0 0 0"/>
  </joint>
  
  <joint name="corner_joint_{link_idx}_1_1" type="fixed">
    <parent link="corner_{link_idx}"/>
    <child link="segment_{link_idx}_2"/>
    <origin xyz="0 {rad} 0" rpy="0 0 0"/>
  </joint>
"""
    return link_urdf, {"start": f"segment_{link_idx}_1", "end": f"segment_{link_idx}_2"}


def get_C_link(link_idx, l1, l2, rad):
    collision_margin = 0.005

    # need to add a segment 0 and another corner link
    additional_segment = f"""
    <link name="segment_{link_idx}_0">
        <visual>
        <geometry>
            <cylinder length="{l2}" radius="{rad}" />
        </geometry>
        <origin xyz="0 {l2/2} 0" rpy="{np.pi/2} 0 0"/>
        <material name="blue">
            <color rgba="0 0 1 1"/>
        </material>
        </visual>
        <collision>
            <geometry>
                <cylinder length="{l2 - collision_margin}" radius="{rad}" />
            </geometry>
            <origin xyz="0 {l2/2} 0" rpy="{np.pi/2} 0 0"/>
        </collision>
        <inertial>
            <mass value="1.0"/>
            <inertia ixx="1.0" iyy="1.0" izz="1.0" ixy="0.0" ixz="0.0" iyz="0.0"/>
            <origin xyz="0 {l2/2} 0" rpy="{np.pi/2} 0 0"/>
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
    <origin xyz="0 {l2 + rad} 0" rpy="0 0 0"/>
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


def get_urdf(v1=0.05, v2=0.1, v3=0.3, v4=0.1, v5=0.15, v6=0.1, v7=0.05, rad=0.035):
    l0 = """
    <link name="base_link"/>
    """

    l1, cur_info = get_L_link(1, v1, v2 / 2, rad)

    joint_0_1 = f"""
    <joint name="joint_0_1" type="revolute">
        <parent link="base_link"/>
        <child link="{cur_info['start']}"/>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-{np.pi/2}" upper="{np.pi/2}" effort="100" velocity="0.1"/>
    </joint>
    """
    prev_info = cur_info

    l2, cur_info = get_C_link(2, v3, v4 / 2, rad)
    joint_1_2 = f"""
    <joint name="joint_1_2" type="revolute">
        <parent link="{prev_info['end']}"/>
        <child link="{cur_info['start']}"/>
        <origin xyz="0 {v2/2} 0" rpy="0 0 0"/>
        <axis xyz="0 1 0"/>
        <limit lower="-{np.pi/2}" upper="{np.pi/2}" effort="100" velocity="0.1"/>
    </joint>
    """
    prev_info = cur_info

    l3, cur_info = get_C_link(3, v3, v4 / 2, rad)

    joint_2_3 = f"""
    <joint name="joint_2_3" type="revolute">
        <parent link="{prev_info['end']}"/>
        <child link="{cur_info['start']}"/>
        <origin xyz="0 {v4/2} 0" rpy="0 0 0" />
        <axis xyz="0 1 0"/>
        <limit lower="{-np.pi/2}" upper="{np.pi/2}" effort="100" velocity="0.1"/>
    </joint>
    """
    prev_info = cur_info

    l4, cur_info = get_L_link(4, v4 / 2, v5 / 2, rad)

    joint_3_4 = f"""    
    <joint name="joint_3_4" type="revolute">
        <parent link="{prev_info['end']}"/>
        <child link="{cur_info['start']}"/>
        <origin xyz="0 {v4/2} 0" rpy="{np.pi/2} {0} {np.pi}"/>
        <axis xyz="0 0 1"/>
        <limit lower="{-np.pi/2}" upper="{np.pi}" effort="100" velocity="0.1"/>
    </joint>
    """
    prev_info = cur_info

    l5, cur_info = get_L_link(5, v5 / 2, v6 / 2, rad)

    joint_4_5 = f"""
    <joint name="joint_4_5" type="revolute">
        <parent link="{prev_info['end']}"/>
        <child link="{cur_info['start']}"/>
        <origin xyz="0 {v5/2} 0" rpy="{-np.pi/2} 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="{-np.pi/2}" upper="{np.pi/2}" effort="100" velocity="0.1"/>
    </joint>
    """
    prev_info = cur_info

    l6 = get_cylinder_link("ee_link", origin=(0, v6 / 4, 0), length=v6 / 2, radius=rad)

    joint_5_6 = f"""
    <joint name="joint_5_6" type="revolute">
        <parent link="{prev_info['end']}"/>
        <child link="ee_link"/>
        <origin xyz="0 {v6/2} 0" rpy="{-np.pi/2} 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-{np.pi}" upper="{np.pi}" effort="100" velocity="0.1"/>
    </joint>
    """

    ee_link = get_ee_link(
        parent_linkname="ee_link",
        ee_length=v7,
        joint_origin=(0, 0, v6 / 2),
        joint_rpy=(0, 0, -np.pi / 2),
        displacement=rad * 0.95,
    )

    urdf = f"""<?xml version="1.0"?>
<robot name="ur5">
{l0}
{l1}
{l2}
{l3}
{l4}
{l5}
{l6}
{joint_0_1}
{joint_1_2}
{joint_2_3}
{joint_3_4}
{joint_4_5}
{joint_5_6}
{ee_link}
</robot>
"""
    return urdf


def save_urdfs(to_usd=False, randomized=False):
    vals1 = {
        "v1": 0.05,
        "v2": 0.1,
        "v3": 0.3,
        "v4": 0.1,
        "v5": 0.15,
        "v6": 0.1,
        "v7": 0.05,
        "rad": 0.035,
    }
    vals2 = {
        "v1": 0.05,
        "v2": 0.1,
        "v3": 0.2,
        "v4": 0.1,
        "v5": 0.1,
        "v6": 0.1,
        "v7": 0.05,
        "rad": 0.025,
    }
    vals3 = {
        "v1": 0.03,
        "v2": 0.07,
        "v3": 0.2,
        "v4": 0.07,
        "v5": 0.1,
        "v6": 0.1,
        "v7": 0.03,
        "rad": 0.025,
    }
    
    
    val_ranges = {
      "v1": [0.03, 0.06],
      "v2": [0.07, 0.14],
      "v3": [0.2, 0.4],
      "v4": [0.07, 0.14],
      "v5": [0.1, 0.2],
      "v6": [0.1, 0.2],
      "v7": [0.03, 0.06],
      "rad": [0.025, 0.05],
    }

    if not randomized:

      for idx, val in enumerate([vals1, vals2, vals3]):
          my_urdf = get_urdf(**val)

          robo_name = f"arm_ur5_{idx}"
          robo_dir = get_robot_morphs_dir() / "arm_ur5" / robo_name
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
        
        vals = {k: np.random.uniform(v[0], v[1]) for k, v in val_ranges.items()}  
        my_urdf = get_urdf(**vals)
        
        robo_name = f"r{idx}"
        robo_dir = get_robot_morphs_dir() / "arm_ur5" / robo_name
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
    # write urdf to file

    vals1 = {
        "v1": 0.05,
        "v2": 0.1,
        "v3": 0.3,
        "v4": 0.1,
        "v5": 0.15,
        "v6": 0.1,
        "v7": 0.05,
        "rad": 0.035,
    }
    # vals2 = {'v1': 0.05, 'v2': 0.1, 'v3': 0.2, 'v4': 0.1, 'v5': 0.1, 'v6': 0.1, 'v7': 0.05, 'rad':0.025}
    # vals3 = {'v1': 0.03, 'v2': 0.07, 'v3': 0.2, 'v4': 0.07, 'v5': 0.1, 'v6': 0.1, 'v7': 0.03, 'rad':0.025}

    my_urdf = get_urdf(**vals1)

    urdf_path = get_tmp_mesh_storage() / "tongs.urdf"
    with open(urdf_path, "w") as f:
        f.write(my_urdf)

    urdfpy_robot = urdfpy.URDF.load(urdf_path)
    vis = VizServer()

    joint_names = [j.name for j in urdfpy_robot.joints if j.joint_type != "fixed"]

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
