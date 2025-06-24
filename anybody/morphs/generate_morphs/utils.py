import numpy as np
import urdfpy
import xml.etree.ElementTree as ET


def get_fixed_base_link(base_link_name):
    fixed_joint = f"""
    <joint name="joint_{base_link_name}" type="fixed">
        <parent link="base_link"/>
        <child link="{base_link_name}"/>
        <origin xyz="0 0 0" rpy="0 0 0"/>
    </joint>
    """
    return fixed_joint


def get_3dof_base_link(base_link_name):
    
    # the three dofs are prismatic in x, prismatic in y, and revolute in z
    
    base_link = f"""
    <link name="{base_link_name}_px"/> 
    <link name="{base_link_name}_py"/>
    
    <joint name="joint_{base_link_name}_px" type="prismatic">
        <parent link="base_link"/>
        <child link="{base_link_name}_px"/>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <axis xyz="1 0 0"/>
        <limit lower="-0.5" upper="0.5" effort="1" velocity="1"/>
    </joint>
    
    <joint name="joint_{base_link_name}_py" type="prismatic">
        <parent link="{base_link_name}_px"/>
        <child link="{base_link_name}_py"/>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <axis xyz="0 1 0"/>
        <limit lower="-0.5" upper="0.5" effort="1" velocity="1"/>
    </joint>
    
    <joint name="joint_{base_link_name}_rz" type="revolute">
        <parent link="{base_link_name}_py"/>
        <child link="{base_link_name}"/>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-{2 * np.pi}" upper="{2 * np.pi}" effort="1" velocity="1"/>
    </joint>
    """
    
    return base_link
    

def get_5dof_base_link(base_link_name, pos_only=False):
    """
    returns the urdf snippet
    assumes that 'base_link' is defined outside
    """

    base_link = f"""    
    <link name="{base_link_name}_px"/>
    <link name="{base_link_name}_py"/>

    <joint name="joint_{base_link_name}_px" type="prismatic">
        <parent link="base_link"/>
        <child link="{base_link_name}_px"/>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <axis xyz="1 0 0"/>
        <limit lower="-0.7" upper="0.7" effort="1" velocity="1"/>
    </joint>
    
    <joint name="joint_{base_link_name}_py" type="prismatic">
        <parent link="{base_link_name}_px"/>
        <child link="{base_link_name}_py"/>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <axis xyz="0 1 0"/>
        <limit lower="-0.7" upper="0.7" effort="1" velocity="1"/>
    </joint>
    
"""

    if pos_only:
        base_link += f"""
    <joint name="joint_{base_link_name}_pz" type="prismatic">
        <parent link="{base_link_name}_py"/>
        <child link="{base_link_name}"/>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="0.0" upper="0.5" effort="1" velocity="1"/>
    </joint>
"""

    else:
        base_link += f"""
    <link name="{base_link_name}_pz"/>
    <link name="{base_link_name}_rz"/>

    <joint name="joint_{base_link_name}_pz" type="prismatic">
        <parent link="{base_link_name}_py"/>
        <child link="{base_link_name}_pz"/>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="0.0" upper="0.5" effort="1" velocity="1"/>
    </joint>
    
    <joint name="joint_{base_link_name}_rz" type="revolute">
        <parent link="{base_link_name}_pz"/>
        <child link="{base_link_name}_rz"/>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-{2 * np.pi}" upper="{2 * np.pi}" effort="1" velocity="1"/>
    </joint>

    <joint name="joint_{base_link_name}_ry" type="revolute">
        <parent link="{base_link_name}_rz"/>
        <child link="{base_link_name}"/>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <axis xyz="0 1 0"/>
        <limit lower="-{2 * np.pi}" upper="{2 * np.pi}" effort="1" velocity="1"/>
    </joint>

"""
    return base_link
        



def update_urdf(urdf_str, joint_updates):
    """
    Update the URDF string with new joint limits, efforts, and velocities.

    Parameters:
    urdf_str (str): The original URDF string.
    joint_updates (list of dict): A list of dictionaries, each containing:
                                  - 'name' (str): Joint name.
                                  - 'lower' (float): Lower joint limit.
                                  - 'upper' (float): Upper joint limit.
                                  - 'effort' (float): Joint effort.
                                  - 'velocity' (float): Joint velocity.

    Returns:
    str: The updated URDF string.
    """
    # Parse the URDF string into an XML tree
    root = ET.fromstring(urdf_str)

    # Iterate over each joint update
    for update in joint_updates:
        joint_name = update["name"]

        # Find the corresponding joint element in the URDF
        joint_elem = root.find(f".//joint[@name='{joint_name}']")
        if joint_elem is not None:
            # Find the limit element within the joint element
            limit_elem = joint_elem.find("limit")
            if limit_elem is not None:
                # Update the limit attributes
                
                if 'lower' in update:
                    limit_elem.set("lower", str(update["lower"]))
                if 'upper' in update:
                    limit_elem.set("upper", str(update["upper"]))
                if 'effort' in update:
                    limit_elem.set("effort", str(update["effort"]))
                if 'velocity' in update:
                    limit_elem.set("velocity", str(update["velocity"]))
                
                # lower = update["lower"]
                # upper = update["upper"]
                # effort = update["effort"]
                # velocity = update["velocity"]
                        
                # limit_elem.set("lower", str(lower))
                # limit_elem.set("upper", str(upper))
                # limit_elem.set("effort", str(effort))
                # limit_elem.set("velocity", str(velocity))

    # Convert the updated XML tree back to a string
    updated_urdf_str = ET.tostring(root, encoding="unicode")
    
    updated_urdf_str = f"""<?xml version="1.0"?>\n{updated_urdf_str}"""

    return updated_urdf_str


def update_joint_params(urdf_path):
    robo_urdfpy = urdfpy.URDF.load(urdf_path)  
    joint_names = [j.name for j in robo_urdfpy.joints if j.joint_type != "fixed"]
    joint_effort = [100.0] * len(joint_names)
    joint_velocity = [10.0] * len(joint_names)

    joint_updates = [
        {"name": j, "effort": e, "velocity": v}
        for j, e, v in zip(
            joint_names, joint_effort, joint_velocity
        )
    ]
    
    with open(urdf_path, "r") as f:
        my_urdf = f.read()
    
    new_urdf = update_urdf(my_urdf, joint_updates)
    
    with open(urdf_path, "w") as f:
        f.write(new_urdf)

    return new_urdf



# Function to build the kinematic tree
def build_kinematic_tree(robot):
    tree = {}

    def add_link_to_tree(link, parent_name=None):
        tree[link.name] = {
            "parent": parent_name,
            "children": [child.name for child in link.child_links],
        }
        for child_link in link.child_links:
            add_link_to_tree(child_link, link.name)

    # Start from the base link (usually the first link in the list)
    base_link = robot.links[0]
    add_link_to_tree(base_link)

    return tree
