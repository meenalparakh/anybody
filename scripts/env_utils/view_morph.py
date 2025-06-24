import numpy as np
import urdfpy as ud
from urdfpy import URDF
from anybody.utils.path_utils import get_robot_morphs_dir
from anybody.utils.vis_server import VizServer
from anybody.morphs.generate_morphs import create_real_robot_usd
from anybody.utils.collision_utils import get_best_fit

import torch


import sys


def get_robo_link_info(robo_urdfpy: ud.URDF):
    # info is a tuple of (obj_type, shape_params, origin)
    link_info = {}
    for idx, link in enumerate(robo_urdfpy.links):
        
        # if idx == 1:
            
            
        link: ud.Link = link
        if link.visuals is None or (len(link.visuals) == 0):
            # return the default
            info = ("null", {}, np.eye(4), idx)

        elif len(link.visuals) > 1:
            
            all_meshes = [link.collision_mesh]
            origin_list = [np.eye(4)]
            
            info = get_mesh_info(all_meshes, origin_list, idx)

        elif len(link.visuals) == 1:
            link_geom = link.visuals[0].geometry
            origin = link.visuals[0].origin
            origin = torch.tensor(origin)

            if link_geom.box is not None:
                info = ("box", {"dims": link_geom.box.size}, origin, idx)
            elif link_geom.sphere is not None:
                info = ("sphere", {"radius": link_geom.sphere.radius}, origin, idx)
            elif link_geom.cylinder is not None:
                info = (
                    "cylinder",
                    {
                        "radius": link_geom.cylinder.radius,
                        "height": link_geom.cylinder.length,
                    },
                    origin,
                    idx,
                )

            elif link_geom.mesh is not None:
                
                info = get_mesh_info([link.collision_mesh], [np.eye(4)], idx)

            else:
                raise ValueError(f"{link.name}: Unknown geometry type")

        else:
            import pdb; pdb.set_trace()
            raise ValueError("Unknown geometry type")

        link_info[link.name] = dict(
            zip(["obj_type", "shape_params", "origin", "idx"], info)
        )

    return link_info



def get_mesh_info(meshes_list, origin_list, link_idx):
    # find combined mesh
    combined_mesh = meshes_list[0].apply_transform(origin_list[0])
    for idx, m in enumerate(meshes_list[1:], start=1):
        combined_mesh = combined_mesh + m.apply_transform(origin_list[idx])

    shape, shape_pose, shape_dims = get_best_fit(combined_mesh)
    if shape == "box":
        info = ("box", {"dims": shape_dims}, shape_pose, link_idx)
    else:
        assert shape == "cylinder", "Only box and cylinder shapes are supported"
        info = (
            "cylinder",
            {"radius": shape_dims[0], "height": shape_dims[1]},
            shape_pose,
            link_idx,
        )

    return info



def visualize_robot(robo_cat, robo_name):
    # urdf_path = 
    urdf_path = get_robot_morphs_dir() / str(robo_cat) / f"{robo_name}" / f"{robo_name}.urdf"
    
    if robo_cat in ['real', "panda_variations"]:
        try:
            create_real_robot_usd(robo_name)        
        except ModuleNotFoundError:
            print("Skipping USD creation for real robot, requires starting the simulation app")   
        
        urdf_path = urdf_path.with_name(f"{robo_name}_new.urdf")
    
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF file not found at {urdf_path}")
    
    urdfpy_robot = URDF.load(urdf_path)
    vis = VizServer()

    joint_names = [j.name for j in urdfpy_robot.joints if j.joint_type != "fixed"]

    jlimits = {j.name: j.limit for j in urdfpy_robot.joints}
    joint_lb = [jlimits[jname].lower for jname in joint_names]
    joint_ub = [jlimits[jname].upper for jname in joint_names]

    link_info = get_robo_link_info(urdfpy_robot)

    for _ in range(20):
        # joint_vals = np.zeros(len(joint_names))
        joint_vals = np.random.uniform(joint_lb, joint_ub)
        
        vis.view_robot_fitted(urdfpy_robot, None, joint_names, joint_vals) 

        # vis.view_robot_fitted_isaac(urdfpy_robot, None, joint_names, joint_vals, link_info) 
        
        input("Press Enter to continue...")


if __name__ == "__main__":
    
    robo_cat = sys.argv[1]
    robo_name = sys.argv[2]
    
    visualize_robot(robo_cat, robo_name)