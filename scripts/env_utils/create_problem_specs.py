from anybody.utils.start_sim import args_cli

from anybody.utils.path_utils import get_robot_morphs_dir, get_benchmark_cfgs_dir

import yaml
from anybody.envs.tasks import generate_problem_spec
import json
from anybody.cfg import cfg
from pathlib import Path
from urdfpy import URDF
import anybody.envs.sim.utils as iu
from anybody.utils.utils import load_pickle, save_pickle  
from anybody.morphs.generate_morphs import create_real_robot_usd
import numpy as np
from scipy.spatial.transform import Rotation as R


def get_links_info(prob):

    robo_link_info = {}
    robo_joint_link_idx_dict = {}

    for robo_id, robo in prob.robot_dict.items():
        
        
        # get robo_info pickled file
        robo_info_pickle_fname = robo.robot_urdf.replace(".urdf", "_info.pkl")
        if not Path(robo_info_pickle_fname).exists() or cfg.FORCE_RECOMPUTE_LINK_INFO:
            robo_urdfpy = URDF.load(robo.robot_urdf)
            links_info = iu.get_robo_link_info(robo_urdfpy)
            joints_info, child_dict = iu.get_robo_joint_cfg(
                robo_urdfpy, return_child_dict=True
            )
            save_pickle((links_info, joints_info, child_dict), robo_info_pickle_fname)
        else:
            links_info, joints_info, child_dict = load_pickle(robo_info_pickle_fname)

        continue

        link_vec_dict, joint_link_idx_dict = iu.get_robo_link_vec(
            links_info=links_info,
            joints_info=joints_info,
            child_dict=child_dict,
            goal_state=prob.goal_robo_state_dict.get(robo_id, None),
            ee_link_name=robo.ee_link,
            link_idx_encode_dim=cfg.OBSERVATION.ROBO_LINK_IDX_ENCODE_DIM,
        )

        robo_link_info[robo_id] = link_vec_dict
        robo_joint_link_idx_dict[robo_id] = joint_link_idx_dict


    return
    to_iterate_indices = list(prob.robot_dict.keys())
    assert len(to_iterate_indices) == 1
    
    robo_id = to_iterate_indices[0]
    to_iterate_link_indices = list(robo_link_info[robo_id].keys())
    return len(to_iterate_link_indices)

# convert 4x4 array to 6x1 array
def X_to_euler(X):  
    pos = X[:3, 3]
    rot = X[:3, :3]
    euler = R.from_matrix(rot).as_euler('xyz')
    return np.concatenate([pos, euler])


def get_base_pose_info(prob):
    for robo_id, robo in prob.robot_dict.items():
        return X_to_euler(robo.pose)


def get_robo_names(robo_cat):
    robot_morphs_dir = get_robot_morphs_dir()
    robo_cat_dir = robot_morphs_dir / robo_cat
    robo_cat_dirs = [d for d in robo_cat_dir.iterdir() if d.is_dir()]
    robo_cat_dirs = [d.name for d in robo_cat_dirs]

    valid_robo_names = []        
    for robo_name in robo_cat_dirs:
        if "_" in robo_name:
            # typically "_" was in all prior generated robo names. the new names do not have these
            continue

        potential_robo_urdf = (
            robot_morphs_dir / robo_cat / robo_name / f"{robo_name}.urdf"
        )
        if potential_robo_urdf.exists():
            valid_robo_names.append(robo_name)
            
    return valid_robo_names


def generate_problem_specs_for_cfgs(cfg_name):
    
    cfg_file = get_benchmark_cfgs_dir() / f"{cfg_name}.yaml"
    # read the yaml file
    with open(cfg_file, "r") as f:
        cfg_dict = yaml.safe_load(f)
        
    # get the robot names
    robo_cats = cfg_dict['MULTIENV']['ROBOTS']
    robo_names = cfg_dict['MULTIENV']['VARIATIONS']
    
    # also append the test envs
    test_robo_cats = cfg_dict['TEST_ENVS']['ROBOTS']
    test_robo_names = cfg_dict['TEST_ENVS']['VARIATIONS']
    if isinstance(test_robo_cats, str):
        test_robo_cats = [test_robo_cats]
        test_robo_names = [test_robo_names]
        
    robo_cats += test_robo_cats
    robo_names += test_robo_names
        
    # get the task names
    task_names = cfg_dict['MULTIENV']['TASKS']
    if isinstance(task_names, list):
        task_name = task_names[0]
    else:
        task_name = task_names
                
    robo_count = 0

    if "panda_variations" in robo_cats:
        create_real_robot_usd("panda")


    for robo_cat, robo_name in zip(robo_cats, robo_names):

        if robo_cat in ["real", "panda_variations"]:
            create_real_robot_usd(robo_name[:-3])

        print(f"Generating problem spec for {robo_name} | {robo_cat} | {task_name} | s{robo_count}")
                
        prob = generate_problem_spec(
            benchmark_task=cfg_name,
            robo_task=task_name, 
            robo_cat=robo_cat,
            variation=robo_name,
            seed=robo_count,
            save_if_not_exist=True
        )
        get_links_info(prob)        
        robo_count += 1
    
    
    
if __name__ == "__main__":

    cfg.merge_from_list(args_cli.opts)
    cfg.ALLOW_PROBLEM_SPEC_GEN = True
    
    cfg.FORCE_RECOMPUTE_PROBLEM_SPEC = False
    cfg.FORCE_ROBO_LINK_USD_CONVERSION = True
    cfg.FORCE_RECOMPUTE_LINK_INFO = True

    generate_cfgs = True

    if generate_cfgs:
        cfg.NUM_GOAL_RANDOMIZATIONS = 100

    benchmark_tasks = get_benchmark_cfgs_dir() / "all_tasks.json"
    
    import json
    with open(benchmark_tasks, "r") as f:
        cfg_names = json.load(f)
                
    # cfg_names = ["inter_arms_reach_v2", "inter_arms_push_simple_v2"]
                
    for cfg_name in cfg_names:
        print(f"Generating problem specs for {cfg_name} ############################")
        generate_problem_specs_for_cfgs(cfg_name)
 
