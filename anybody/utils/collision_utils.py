import trimesh
import numpy as np


def get_collision_spheres_stick(length, dist, radius):
    # length is the length of stick (assuming stick is along x-axis)
    # dist is the distance between the spheres
    # radius is the radius of the spheres

    result = []
    start_pt = np.array([0.0, 0.0, 0.0])
    cur_pt = start_pt.copy()
    while cur_pt[0] < length:
        next_pt = cur_pt + np.array([dist, 0.0, 0.0])
        result.append((cur_pt, radius))
        cur_pt = next_pt

    return result


def get_sphere_string(pt, r):
    pt_lst = np.round(pt, decimals=4).tolist()
    pt_lst_str = [str(x) for x in pt_lst]
    pt_lst_str = ", ".join(pt_lst_str)

    return "- {'center': " + pt_lst_str + ", 'radius': " + str(r) + "}"


def convert_sphere_lst_to_string(sphere_lst, n_tabs=4):
    # assuming sphere lst is [(pt1, r1), (pt2, r2), ...]
    # returns a string of the form """
    # - {"center": [x1, y1, z1], "radius": r1}
    # - {"center": [x2, y2, z2], "radius": r2}
    # ...
    # """
    result = "\n".join(
        ["    " * n_tabs + get_sphere_string(pt, r) for pt, r in sphere_lst]
    )
    return result


def convert_collisions_info_to_str(collision_spheres, n_tabs=1):
    # the input is {'link_name': collision_lst, ...}
    # the output

    result = "collision_spheres:\n"
    for link_name in collision_spheres:
        result += "    " * n_tabs + f"{link_name}:\n"
        result += convert_sphere_lst_to_string(collision_spheres[link_name], n_tabs + 1)
        result += "\n"
    return result


def get_box_fit(mesh: trimesh.Trimesh):
    # returns the pose, and dimensions of the box that fits the mesh
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    return np.linalg.inv(to_origin), extents
    
    # mXw @ wXm = mXm

def get_cylinder_fit(mesh: trimesh.Trimesh):
    result = trimesh.bounds.minimum_cylinder(mesh, sample_count=20)
    return result['transform'], (result['radius'], result['height'])


def get_best_fit(mesh: trimesh.Trimesh):
    box_pose, box_dims = get_box_fit(mesh)
    cyl_pose, cyl_dims = get_cylinder_fit(mesh)
    
    box_vol = np.prod(box_dims)
    cyl_vol = np.pi * cyl_dims[0] ** 2 * cyl_dims[1]
    
    print(f"Box volume: {box_vol}, Cylinder volume: {cyl_vol}")
    
    if box_vol < cyl_vol:
        return 'box', box_pose, box_dims
    else:
        return 'cylinder', cyl_pose, cyl_dims
    
    
    