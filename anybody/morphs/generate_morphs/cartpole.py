import numpy as np
from anybody.utils.path_utils import get_robot_morphs_dir

urdf_template = \
"""<?xml version="1.0"?>

<robot xmlns="http://drake.mit.edu"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
 name="CartPole">

  <link name="world">
  </link>

  <link name="cart">
    <inertial>
      <origin xyz="0 0 HALFLENGTH" rpy="0 0 0" />
      <mass value="10" />
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0" />
    </inertial>
    <visual>
      <origin xyz="0 0 HALFLENGTH" rpy="0 0 0" />
      <geometry>
        <box size=".6 .3 .3" />
      </geometry>
      <material>
        <color rgba="0 1 0 1" />
      </material>
    </visual>
  </link>

  <link name="pole">
    <inertial>
      <origin xyz="0 0 LENGTH" rpy="0 0 0" />
      <mass value="MASS" />
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0" />
    </inertial>
    <visual>
      <origin xyz="0 0 HALFLENGTH" rpy="0 0 0" />
      <geometry>
         <cylinder length="LENGTH" radius=".01" />
      </geometry>
      <material>
        <color rgba="1 0 0 1" />
      </material>
    </visual>
    <visual>
      <origin xyz="0 0 LENGTH" rpy="0 0 0" />
      <geometry>
         <sphere radius=".05" />
      </geometry>
      <material>
        <color rgba="0 0 1 1" />
      </material>
    </visual>
  </link>
  
  <joint name="x" type="prismatic">
    <parent link="world" />
    <child link="cart" />
    <origin xyz="0 0 0" />
    <axis xyz="1 0 0" />
  </joint>

  <joint name="theta" type="continuous">
    <parent link="cart" />
    <child link="pole" />
    <origin xyz="0 0 HALFLENGTH" />
    <axis xyz="0 -1 0" />
  </joint>

  <transmission type="SimpleTransmission" name="cart_force">
    <actuator name="force" />
    <joint name="x" />
    <mechanicalReduction>1</mechanicalReduction>
  </transmission>
</robot>
"""

def get_urdf(length=0.5, mass=1.0):
    # replace halflength first to avoid replacing length
    return urdf_template.replace("HALFLENGTH", str(length/2)).replace("LENGTH", str(length)).replace("MASS", str(mass))

def save_urdfs(to_usd=False, randomized=True):
    lengths = np.linspace(0.5, 1.5, 5)
    masses = np.linspace(0.5, 1.5, 5)
    
    robo_count = 0
    for l, m in zip(lengths, masses):
        my_urdf = get_urdf(length=l, mass=m)
        robo_name = f"r{robo_count}"
        robo_dir = get_robot_morphs_dir() / "cartpole" / robo_name  
        robo_dir.mkdir(parents=True, exist_ok=True)
        
        urdf_path = robo_dir / f"{robo_name}.urdf"
        
        with open(urdf_path, "w") as f:
            f.write(my_urdf)
            
        robo_count += 1
    
        if to_usd:
            usd_path = robo_dir / f"{robo_name}.usd"
            from anybody.utils.to_usd import ArgsCli, main
            args_cli = ArgsCli(input=str(urdf_path), output=str(usd_path), headless=True)
            main(args_cli)
                            