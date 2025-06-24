import numpy as np
import trimesh
import urdfpy as ud
import pickle
import shutil   
import random
import torch
from scipy.spatial.transform import Rotation as R
from gym.utils import seeding
import os
from collections import defaultdict


def nan_check(x, name):
    assert not torch.isnan(x).any(), f"NaN detected in {name}"

class tensordict(dict):
    
    def float(self):
        for k, v in self.items():
            self[k] = v.float()
        return self

    def copy_(self, other):
        for k in self.keys():
            self[k].copy_(other[k])
        return self

    
    @property
    def shape(self):
        return 


class mydequedict(dict):
    def clear(self):
        for k in self.keys():
            self[k].clear()
        return self


class mydefaultdict(defaultdict):
    
    def __init__(self, *args, default_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_fn = default_fn
    
    def __missing__(self, key):
        if self.default_fn:
            return self.default_fn(key)
        else:
            return super().__missing__(key)


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def load_pickle(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data


def strong_seed(seed):
    """Get a strong uncorrelated seed from naive seeding."""
    seed = seeding.hash_seed(seed)
    _, seed = divmod(seed, 2**32)
    return seed


def set_seed(seed, idx=0, use_strong_seeding=False):
    seed = seed + idx
    if use_strong_seeding:
        seed = strong_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def format_str(s, replace_dict):
    for k, v in replace_dict.items():
        s = s.replace(k, str(v))
    return s


def get_quat(euler):
    quat = R.from_euler("xyz", euler).as_quat()
    return (quat[3], *(quat[:3]))


def get_posquat(pose):
    pos = pose[:3, 3]
    # [w, x, y, z]
    quat = R.from_matrix(pose[:3, :3]).as_quat()
    posquat = np.array([*pos, quat[3], quat[0], quat[1], quat[2]])
    return posquat


def transform_pcd(pcd, pose):
    assert pcd.shape[1] == 3
    assert pose.shape == (4, 4)

    pcd_concatenated = np.concatenate([pcd, np.ones((pcd.shape[0], 1))], axis=1)

    pcd_transformed = pose @ pcd_concatenated.T
    return pcd_transformed[:3].T


def transform_normal(normal, pose):
    assert normal.shape[1] == 3
    assert pose.shape == (4, 4)
    normal_transformed = pose[:3, :3] @ normal.T
    return normal_transformed.T


def load_mesh(obj_type, obj_shape):
    # for box, shape is [x, y, z]
    # for cylinder, shape is [r, h]
    # for sphere, shape is [r]

    if obj_type == "box":
        return trimesh.creation.box(obj_shape)
    elif obj_type == "sphere":
        return trimesh.creation.icosphere(radius=obj_shape[0])
    elif obj_type == "cylinder":
        return trimesh.creation.cylinder(radius=obj_shape[0], height=obj_shape[1])


# [ R | t ]  @ [I | t1] = [R | t + R @ t1]
# [ 0 | 1 ]    [0 |  1]   [0 |          1]


def _rotation_matrices(angles, axis):
    """Compute rotation matrices from angle/axis representations.

    Parameters
    ----------
    angles : (n,) float
        The angles.
    axis : (3,) float
        The axis.

    Returns
    -------
    rots : (n,4,4)
        The rotation matrices
    """
    axis = axis / np.linalg.norm(axis)
    sina = np.sin(angles)
    cosa = np.cos(angles)
    M = np.tile(np.eye(4), (len(angles), 1, 1))
    M[:, 0, 0] = cosa
    M[:, 1, 1] = cosa
    M[:, 2, 2] = cosa
    M[:, :3, :3] += (
        np.tile(np.outer(axis, axis), (len(angles), 1, 1))
        * (1.0 - cosa)[:, np.newaxis, np.newaxis]
    )
    M[:, :3, :3] += (
        np.tile(
            np.array(
                [
                    [0.0, -axis[2], axis[1]],
                    [axis[2], 0.0, -axis[0]],
                    [-axis[1], axis[0], 0.0],
                ]
            ),
            (len(angles), 1, 1),
        )
        * sina[:, np.newaxis, np.newaxis]
    )
    return M


def get_parent_child_mapping(robo_urdfpy: ud.URDF):
    child_links = {}  # key is the parent, value is the list of children
    parent_links = {}  # key is the child, value is the parent

    for joint in robo_urdfpy.joints:
        joint: ud.Joint = joint
        if joint.parent not in child_links:
            child_links[joint.parent] = [joint.child]
        else:
            child_links[joint.parent].append(joint.child)

        parent_links[joint.child] = joint.parent

    return child_links, parent_links


def get_visible_parent_mapping(robo_urdfpy: ud.URDF):
    _, parent_links = get_parent_child_mapping(robo_urdfpy)
    link_parents = {"base_link": {"parent": "world"}}
    visibility_mapping = {"base_link": True}

    for link in robo_urdfpy.links:
        # check if visual element exists
        if link.name == "base_link":
            continue

        if link.visuals is None or (len(link.visuals) == 0):
            visibility_mapping[link.name] = False
        else:
            visibility_mapping[link.name] = True

    # find the visible parent of each link
    for link in robo_urdfpy.links:
        if link.name == "base_link":
            continue

        if visibility_mapping[link.name]:
            # now we want to find the visible parent of this link
            visible_parent = None
            query_link = link.name
            while visible_parent is None:
                possible_parent = parent_links[query_link]
                if visibility_mapping[possible_parent]:
                    visible_parent = possible_parent
                else:
                    query_link = possible_parent

            link_parents[link.name] = {
                "parent": visible_parent,
                "link_origin": link.visuals[0].origin,
                # 'joint_origin': None
            }

    return link_parents


# def get_relative_pose_vector(robo_urdfpy: ud.URDF):
#     link_parents = get_visible_parent_mapping(robo_urdfpy)

#     link_info = {'base_link': (np.eye(4), 'world')}
#     for link in robo_urdfpy.links:

#         link: ud.Link = link
#         # we only consider links with visual information
#         # we compute relative pose to the last visual element
#         if link.visuals is None or (len(link.visuals) == 0):
#             continue

#         assert len(link.visuals) == 1, "Only one visual per link is supported"
#         visual: ud.Visual = link.visuals[0]
#         visual_pose = visual.origin
#         link_info[link.name] = (visual_pose, link.parent)


def get_robo_link_vector(robo_link_info):
    return {
        k: get_obj_geometry_vector(*v, is_torch=True) for k, v in robo_link_info.items()
    }

def is_none(val):
    # check if the value is None or "None" or ""
    return (val is None) or (val == "None") or (val == "")


# for cuboid
def get_vec_bounds():
    return np.array([[0, -np.pi / 2], [2 * np.pi, np.pi / 2]])


def vec_to_pt_cuboid(v, cuboid_size, cube_pose):
    x = np.sin(v[1])
    y = np.cos(v[0]) * np.cos(v[1])
    z = np.sin(v[0]) * np.cos(v[1])

    m = np.max(np.abs([x, y, z]))
    m_idx = np.argmax(np.abs([x, y, z]))
    normal = np.zeros(3)
    normal[m_idx] = np.sign([x, y, z])[m_idx]

    pt = np.array([x, y, z]) / m * (np.array(cuboid_size) / 2.0)

    transformed_pt = transform_pcd(pt.reshape((1, 3)), cube_pose)
    transformed_normal = transform_normal(normal.reshape((1, 3)), cube_pose)

    return transformed_pt[0], transformed_normal[0]


def vec_to_pt(obj_shape, v, cube_pose):
    return vec_to_pt_cuboid(v, obj_shape, cube_pose)


def check_collision(
    robot_urdfpy: ud.URDF, base_pose, jdict, object_mesh_dict, object_pose_dict
):
    collision_manager = trimesh.collision.CollisionManager()

    for obj_id in object_mesh_dict:
        mesh = object_mesh_dict[obj_id]
        pose = object_pose_dict[obj_id]
        collision_manager.add_object(f"object_{obj_id}", mesh, transform=pose)

    link_fk = robot_urdfpy.collision_trimesh_fk(jdict)
    for idx, (mesh, pose) in enumerate(link_fk.items()):
        dist, name = collision_manager.min_distance_single(
            mesh, transform=base_pose @ pose, return_name=True
        )
        is_collision = dist < 0.00
        if is_collision:
            print("Collision detected with object: ", name, "distance: ", dist)
            return True
        else:
            collision_manager.add_object(
                f"robot_{idx}", mesh, transform=base_pose @ pose
            )
    return False


def calculate_center_offset(square_dims, grid_size, grid_spacing):
    """
    Calculate the offset of the grid such that it is centered at the origin.

    Parameters:
        square_dims (tuple): Dimensions (width, height) of each square.
        grid_size (tuple): Number of squares along (rows, cols).
        grid_spacing (tuple): Spacing between the squares (dx, dy).

    Returns:
        np.array: Offset to center the grid at the origin.
    """
    grid_width = (grid_size[1] - 1) * (square_dims[0] + grid_spacing[0])
    grid_height = (grid_size[0] - 1) * (square_dims[1] + grid_spacing[1])

    offset_x = -grid_width / 2.0
    offset_y = -grid_height / 2.0

    return np.array([offset_x, offset_y])


def arrange_squares_in_centered_grid(square_dims, num_squares, grid_spacing):
    """
    Arrange squares in a grid centered at the origin.

    Parameters:
        square_dims (tuple): Dimensions (width, height) of each square.
        num_squares (int): Total number of squares to arrange.
        grid_spacing (tuple): Spacing between the squares (dx, dy).

    Returns:
        list: A list of center coordinates for each square in the grid.
    """
    # Determine grid size that can hold the given number of squares
    grid_rows = int(np.ceil(np.sqrt(num_squares)))
    grid_cols = int(np.ceil(num_squares / grid_rows))
    grid_size = (grid_rows, grid_cols)

    # Calculate offset to center the grid at the origin
    grid_offset = calculate_center_offset(square_dims, grid_size, grid_spacing)

    centers = []

    for row in range(grid_rows):
        for col in range(grid_cols):
            if len(centers) >= num_squares:
                break
            center_x = col * (square_dims[0] + grid_spacing[0]) + grid_offset[0]
            center_y = row * (square_dims[1] + grid_spacing[1]) + grid_offset[1]
            centers.append(np.array([center_x, center_y, 0]))

    return centers


def sample_jvals(joint_types, joint_lb, joint_ub):
    
    # for jtype revolute and prismatic, sample from uniform distribution between lb and ub
    # for continuous joints, sample from uniform distribution between -pi and pi
    
    # replace None with np.inf and -np.inf
    
    jvals = []
    for i, jtype in enumerate(joint_types):
        if jtype in ['revolute', 'prismatic']:
            jvals.append(np.random.uniform(joint_lb[i], joint_ub[i]))
        else:
            jvals.append(np.random.uniform(-np.pi, np.pi))
            
    return jvals




def select_env(obs, env_idx, device=None):
    new_obs = {}
    for key, value in obs.items():
        if isinstance(value, dict):
            new_obs[key] = select_env(value, env_idx, device)
        else:
            if device:
                new_obs[key] = value[env_idx].to(device)
            else:   
                new_obs[key] = value[env_idx]

    return new_obs


def make_episodes(data_name, remove_original_files=False):
    # load the data
    # and divide it into separate episodes
    # for each episode, if truncated, discard the episode
    # if terminated, save the episode
    
    new_data_dir = get_bc_data_dir() / data_name / "episodes"
    
    obs_dir = get_bc_data_dir() / data_name / "obs"
    rew_dir = get_bc_data_dir() / data_name / "rewards"
    act_dir = get_bc_data_dir() / data_name / "actions"
    term_dir = get_bc_data_dir() / data_name / "terminated"
    trunc_dir = get_bc_data_dir() / data_name / "truncated"
    
    fnames = os.listdir(term_dir) 
    # fnames.sort()
    
    # fnames = term_dir.glob("*.pt")
    n_steps = len(fnames)
    
    n_envs = torch.load(str(term_dir / fnames[0]), map_location=torch.device('cpu')).shape[0]
    print(f"Number of environments: {n_envs}")
    
    for env_idx in range(n_envs):
        print(f"Processing environment: {env_idx}/{n_envs}")
        term_history = []
        trunc_history = []
        
        # for fn in fnames:
        for n in range(n_steps):
            file_name = str(n).zfill(6) + ".pt"
            
            term = torch.load(str(term_dir / file_name), map_location=torch.device('cpu'))
            trunc = torch.load(str(trunc_dir / file_name), map_location=torch.device('cpu'))
            
            # get the values for corresponding env
            term_history.append(term[env_idx])
            trunc_history.append(trunc[env_idx])
            
        # divide episodes based on term and trunc
        # if term, save the episode
        # if trunc, discard the episode
        term = torch.tensor(term_history)
        trunc = torch.tensor(trunc_history)
        
        ep_end = (term + trunc) > 0
        
        # get the indices of the episode ends
        ep_end_idx = torch.where(ep_end)[0]
        
        start_idx = 0
        for episode_idx, end_idx in enumerate(ep_end_idx):
            # load the obs and act data for given env_idx
                
            # check if the episode is truncated
            if term[end_idx] and ((end_idx - start_idx) > 1) and ((end_idx - start_idx) < 100):
                episode_dir = new_data_dir / f"env_{env_idx}" / f"episode_{episode_idx}"
                episode_dir.mkdir(parents=True, exist_ok=True)
                    
                for step_idx, idx in enumerate(range(start_idx, end_idx+1)):
                    fname = str(idx).zfill(6) + ".pt"
                    
                    obs = torch.load(str(obs_dir / fname))
                    
                    obs_to_save = select_env(obs, env_idx, device='cpu')
                    # obs = torch.load(obs_dir / fname)[env_idx]
                    act = torch.load(str(act_dir / fname), map_location=torch.device('cpu'))[env_idx].to("cpu")
                    
                    torch.save(obs_to_save, episode_dir / f"obs_{step_idx}.pt")
                    torch.save(act, episode_dir / f"act_{step_idx}.pt")
                
            start_idx = end_idx + 1         
            
            
    if remove_original_files:
        # remove the action, observation, reward, terminated and truncated files
        shutil.rmtree(obs_dir)
        shutil.rmtree(rew_dir)
        shutil.rmtree(act_dir)
        shutil.rmtree(term_dir)
        shutil.rmtree(trunc_dir)
            
            
def remove_original_dirs(data_name):
    
    obs_dir = get_bc_data_dir() / data_name / "obs"
    rew_dir = get_bc_data_dir() / data_name / "rewards"
    act_dir = get_bc_data_dir() / data_name / "actions"
    term_dir = get_bc_data_dir() / data_name / "terminated"
    trunc_dir = get_bc_data_dir() / data_name / "truncated"
    
    # cont = input(f"Remove directiories within {data_name}? [y/n]: ") 
    
    # if cont.lower() != 'y':
    #     return
    
    
    if obs_dir.exists():
        shutil.rmtree(obs_dir)
        
    if rew_dir.exists():
        shutil.rmtree(rew_dir)
        
    if act_dir.exists():
        shutil.rmtree(act_dir)
        
    if term_dir.exists():
        shutil.rmtree(term_dir)
        
    if trunc_dir.exists():
        shutil.rmtree(trunc_dir)
        
    print(f"Removed original directories within {data_name}")
    
    
def get_category_data_names(cat):
    # get all the dirs in the data dir
    data_dir = get_bc_data_dir()
    all_datanames = os.listdir(data_dir)
    
    cat_datanames = []
    for dataname in all_datanames:
        if cat in dataname:
            cat_datanames.append(dataname)
            
    return cat_datanames
    
