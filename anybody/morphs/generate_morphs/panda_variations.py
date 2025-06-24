import numpy as np
import urdfpy as upy
import trimesh
from anybody.utils.path_utils import get_robot_morphs_dir
import os
from urdfpy import URDF


def change_link_and_joint(link, joint, factor=0.8, tmp_files_dir=None):
    
    # the link is the parent of the joint
    assert joint.parent == link.name, f"Joint {joint.name} parent is not {link.name}"
    
    if joint.joint_type != "revolute":
        return 
    
    position = joint.origin[:3, 3].copy()
    
    # only one of the position values should be non-zero
    non_zero = [i for i in range(3) if np.abs(position[i]) > 1e-6]
    if len(non_zero) != 1:
        return
    
    if np.linalg.norm(position) < 1e-6:
        return
    
    direction = position / np.linalg.norm(position)
    
    # mesh = link.visuals[0].geometry.meshes[0]
    mesh = link.visuals[0].geometry.mesh
    
    # also change the collision mesh
    collision_mesh = link.collisions[0].geometry.mesh   
    
    # import pdb; pdb.set_trace()
    scale = []
    for i in range(3):
        val = direction[i]
        if np.abs(val) > 1e-6:
            scale.append(factor)
        else:
            scale.append(1.0)    

    mesh.scale = scale
    collision_mesh.scale = scale
    # get the vertices of the mesh
    
    new_joint_position = position * np.sqrt(factor)
    new_joint_pose = joint.origin.copy()
    new_joint_pose[:3, 3] = new_joint_position
    
    joint.origin = new_joint_pose
    
    # update the joint position
    
def get_link(robo_urdfpy, linkname):
    for link in robo_urdfpy.links:
        if link.name == linkname:
            return link
    return None


def get_joint(robo_urdfpy, jointname):
    for joint in robo_urdfpy.joints:
        if joint.name == jointname:
            return joint
    return None

def generate_new_urdf(robot_urdf, new_urdf_path):
    robo_urdfpy = upy.URDF.load(robot_urdf)
    
    for joint in robo_urdfpy.joints:
        parent_name = joint.parent
        parent_link = get_link(robo_urdfpy, parent_name)
        factor = np.random.uniform(0.5, 1.5)
        change_link_and_joint(parent_link, joint, factor=factor, tmp_files_dir=None)
        
    return robo_urdfpy

def create_panda_variations_urdfs(n=10):
    urdf_path = get_robot_morphs_dir() / "real" / "panda" / "panda_new.urdf"    
    new_urdf_path = get_robot_morphs_dir() / "panda_variations"
    
    new_urdf_path.mkdir(parents=True, exist_ok=True)
    
    for idx in range(10):
        print(f"Generating new urdf {idx}")
        new_urdf = new_urdf_path / f"panda_{idx}"
        new_urdf.mkdir(parents=True, exist_ok=True) 
        new_robo_urdfpy = generate_new_urdf(str(urdf_path), new_urdf)
        
        # view the new urdf
        # new_robo_urdfpy.animate()
        # new_robo_urdfpy.show()
        new_robo_urdfpy.save(str(new_urdf / f"panda_{idx}.urdf"))
        
        # load the urdf as string and replace all occurrences of 
        # absolute paths with ROBOT_BASE_DIR
        
        with open(str(new_urdf / f"panda_{idx}.urdf"), "r") as f:
            urdf_string = f.read()
        urdf_string = urdf_string.replace(
            str(get_robot_morphs_dir() / "real" / "panda"),
            "ROBOT_BASE_DIR",
        )
        with open(str(new_urdf / f"panda_{idx}.urdf"), "w") as f:
            f.write(urdf_string)
            