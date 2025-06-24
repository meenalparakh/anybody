from .reach.base import envs as reach_base_envs

# from .push.cubes import envs as push_cube_envs
# from .push.stick import envs as push_stick_envs
# from .push.n_link import envs as push_nlink_envs
# from .push.simple_bot import envs as push_bot_envs
# from .push.prims import envs as push_prim_envs
# from .push.base import envs as push_base_envs

from .push_simple.stick import envs as push_simple_stick_envs
from .push_simple.n_link import envs as push_simple_nlink_envs
from .push_simple.base import envs as push_simple_base_envs

# from .grasp.simple_bot import envs as grasp_bot_envs
# from .grasp.prims import envs as grasp_prim_envs
# from .rotate.stick import envs as rotate_stick_envs
# from .rearrange.prims import envs as rearrange_prim_envs
# from .dexterous.scene import envs as dexterous_envs

# from anybody.robot_morphologies import robo_cfg_fns, get_default_robo_cfg
from anybody.utils.utils import save_pickle, load_pickle
from anybody.utils.path_utils import get_robot_morphs_dir, get_problem_spec_dir
from anybody.cfg import cfg
import os
from pathlib import Path

all_robo_cats = [
    "arm_ed",
    "arm_ur5",
    "chain",
    "cubes",
    "multi_finger_arm",
    "multi_finger_v1",
    "multi_finger_v2",
    "nlink",
    "planar_arm",
    "prims",
    "simple_bot",
    "stick",
    "tongs_v1",
    "tongs_v2",
    "real",
    "panda_variations"
]

def additional_envs(task_base_envs, task_cur_envs_list, task_name):
    env_dict = {}

    for robo_cat in all_robo_cats:
        # find all directories that exist in robo_morphs_dir / robo_cat

        if robo_cat in task_cur_envs_list:
            continue


        robo_cat_dir = get_robot_morphs_dir() / robo_cat
        
        if not robo_cat_dir.exists():
            print(f"[WARNING]: Skipping {robo_cat} as it does not exist in robot morphs directory.")
            continue
        
        robo_cat_dirs = [d for d in robo_cat_dir.iterdir() if d.is_dir()]
        robo_cat_dirs = [d.name for d in robo_cat_dirs]

        

        env_dict[robo_cat] = {"joint": {}, "pose": {}}

            
        for robo_name in robo_cat_dirs:
            potential_robo_usd = (
                get_robot_morphs_dir() / robo_cat / robo_name / f"{robo_name}.usd"
            )

            if not potential_robo_usd.exists():
                continue

            # print(f"Fetching {task_name} envs for {robo_cat} {robo_name}")

            for targ in ['pose', 'joint']:
                for version, version_fn in task_base_envs[targ].items():
                    env_dict[robo_cat][targ][f"{robo_name}_{version}"] = (
                        version_fn,
                        robo_cat,
                        robo_name,
                    )

                # lambda seed: version_fn(robo_cat, robo_name, seed)

    return env_dict


reach_envs = {
    # "stick": reach_stick_envs,
    # "nlink": reach_nlink_envs,
    # "cubes": reach_cube_envs,
    # "simple_bot": reach_bot_envs,
    # "ur5": reach_ur5_envs,
    # "prims": reach_prim_envs,
    # "base": reach_base_envs,
}

reach_envs.update(
    additional_envs(reach_base_envs, list(reach_envs.keys()), task_name="reach")
)


push_simple_envs = {
    "stick": push_simple_stick_envs,
    "nlink": push_simple_nlink_envs,
    # "cubes": push_simple_cube_envs,
    # "simple_bot": push_simple_bot_envs,
    # "prims": push_simple_prim_envs,
}
push_simple_envs.update(
    additional_envs(push_simple_base_envs, list(push_simple_envs.keys()), task_name="push_simple")
)


# push_envs = {
#     "stick": push_stick_envs,
#     "nlink": push_nlink_envs,
#     "cubes": push_cube_envs,
#     "simple_bot": push_bot_envs,
#     "prims": push_prim_envs,
# }
# push_envs.update(
#     additional_envs(push_base_envs, list(push_envs.keys()), task_name="push")
# )

# rotate_envs = {
#     "stick": rotate_stick_envs,
# }

# grasp_envs = {
#     "simple_bot": grasp_bot_envs,
#     "prims": grasp_prim_envs,
# }

# rearrange_envs = {
#     "prims": rearrange_prim_envs,
# }

# dexterous_envs = {
#     "kinova": dexterous_envs,
# }


task_env_dicts = {
    "reach": reach_envs,
    # "push": push_envs,
    "push_simple": push_simple_envs,
    # "rotate": rotate_envs,
    # "grasp": grasp_envs,
    # "rearrange": rearrange_envs,
    # "dexterous": dexterous_envs,
}

# task_env_fns = {
#     'reach': get_reach_env,
#     'push': get_push_envs,
#     'articulation': get_articulation_envs,
#     'grasp': get_grasp_envs,
#     'rearrange': get_rearrange_envs,
#     'dexterous': get_dexterous_envs,
# }


def generate_problem_spec(benchmark_task, robo_cat, robo_task, variation, seed=0, save_if_not_exist=False):

    if benchmark_task is None:
        benchmark_task = "other"

    problem_spec_dir = Path(benchmark_task) / f"{robo_cat}_{robo_task}_{variation}"
    # mydir = get_problem_spec_dir() / benchmark_task / 
    abs_dir_path = get_problem_spec_dir() / problem_spec_dir
    problem_spec_path = abs_dir_path / "problem_spec.pkl"
    
    if problem_spec_path.exists() and (not cfg.FORCE_PROBLEM_SPEC_GEN):
        print("*" * 10, "Problem spec already exists", "*" * 10)
        print(f"Loading problem spec from {problem_spec_path}")
        prob = load_pickle(problem_spec_path)
        return prob
    
    else:
        assert cfg.ALLOW_PROBLEM_SPEC_GEN, "Problem spec does not exist. Set ALLOW_PROBLEM_SPEC_GEN to True to generate it."
        target = "pose" if robo_task == "reach" else "joint"
        env_fn = task_env_dicts[robo_task][robo_cat][target][variation]
        # print(env_type, variation, env_fn)
        
        prob = env_fn[0](env_fn[1], env_fn[2], seed=seed, save_dirname=problem_spec_dir, save_fname="event_cfg.pkl")
        
        if save_if_not_exist:
            # save the problem spec
            abs_dir_path.mkdir(parents=True, exist_ok=True)
            save_pickle(prob, problem_spec_path)
    
    return prob