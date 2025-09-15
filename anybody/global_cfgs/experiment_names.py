from anybody.utils.path_utils import get_benchmark_cfgs_dir
from copy import deepcopy
import yaml

REACH_BENCHMARKS = {
    "Arm3": "intra_simple_bot_reach",
    # "Arm3 (Ob)": "intra_simple_bot_reach_v2",
    "Arm3 (Ob, Hd)": "intra_simple_bot_reach_hd_v2",
    "Panda": "intra_panda_reach",
    "EE-Arm": "inter_ee_arm_reach",
    # "Prims": "inter_prims_reach",
    "Arms": "inter_arms_reach_v2",
}

REACH_FINETUNE_BENCHMARKS = {
    "Panda": "intra_panda_reach",
    "EE-Arm": "inter_ee_arm_reach",
    # "Prims": "inter_prims_reach",
    "Arms": "inter_arms_reach_v2",
}


DEFAULT_REACH_EXP_NAMES = {
    "mt": [
        "Tr1-c1-s1-t-nt-h0_mt-eval",
        "Tr0-c0-s0-nm-h0_mt-eval",
        "IndTf_mt-eval",
        "IndMlp_mt-eval",
        "random",
        "diff-ik"
    ],
    "mt_names": [
        "Tf", "Mlp", "Se-Tf", "Se-Mlp", "Rand", "Diff-IK"
    ],
    "zs": [
        "Tr1-c1-s1-t-nt-h0_zs_eval",
        "Tr0-c0-s0-nm-h0_zs_eval",
        "IndTf_mt-eval",
        "IndMlp_mt-eval",
        "random",
        "diff-ik"
    ],
    "zs_names": [
        "Tf", "Mlp", "Se-Tf", "Se-Mlp", "Rand", "Diff-IK"
    ],
    "zs_metrics": ["R99"]     # need to override this for different reach tasks
}

PUSH_BENCHMARKS = {
    "Arm3": "intra_simple_bot_push_simple",
    "Arm3 (Ob)": "intra_simple_bot_push_simple_v2",
    "Arm3 (Ob, Hd)": "intra_simple_bot_push_simple_hd_v2",
    "Panda": "intra_panda_push_simple",
    "EE-Arm": "inter_ee_arm_push_simple",
    "EE-Task": "inter_task_ur5",
    "Prims": "inter_prims_push_simple",
    "Arms": "inter_arms_push_simple_v2",    
}

PUSH_FINETUNE_BENCHMARKS = {
    "Panda": "intra_panda_push_simple",
    "EE-Arm": "inter_ee_arm_push_simple",
    "EE-Task": "inter_task_ur5",
    "Arms": "inter_arms_push_simple_v2",
}

DEFAULT_PUSH_EXP_NAMES = {
    "mt": [
        "Tr1-c1-s1-t-nt-h0_mt-eval",
        "Tr0-c0-s0-nm-h0_mt-eval",
        "IndMlp_mt-eval",
        'IndTf_mt-eval',
        'diff-ik',
    ],
    "mt_names": [
        "Tf", "Mlp", "Se-Mlp", "Se-Tf", "Diff-IK"
    ],
    "zs": [
        "Tr1-c1-s1-t-nt-h0_zs_eval",
        "Tr0-c0-s0-nm-h0_zs_eval",
        "IndMlp_mt-eval",
        'IndTf_mt-eval',
        'diff-ik',
    ],
    "zs_names": [
        "Tf", "Mlp", "Se-Mlp", "Se-Tf", "Diff-IK"
    ],
    "zs_metrics": ["R99"]     # need to override this for different reach tasks
}

ABLATION_EXP_NAMES = {
    "mt": [
        "Tr1-c1-s1-t-nt-h0_mt-eval",
        "Tr0-c1-s1-t-nt-h0_mt-eval",
        "Tr1-c0-s1-t-nt-h0_mt-eval",
        "Tr1-c1-s0-t-nt-h0_mt-eval",
        
        "Tr0-c0-s0-nm-h0_mt-eval",    
        "Tr0-c1-s1-nm-h0_mt-eval",    
        
        "random"        
        "diff-ik"
    ],
    "mt_names": [
        "Tf", "Tf-cont", "Tf-noCE", "Tf-noSL", "Mlp", "Mlp+SlCe", "Rand", "Diff-IK"
    ],
    "zs": [
        "Tr1-c1-s1-t-nt-h0_zs_eval",
        "Tr0-c1-s1-t-nt-h0_zs_eval",
        "Tr1-c0-s1-t-nt-h0_zs_eval",
        "Tr1-c1-s0-t-nt-h0_zs_eval",
        
        "Tr0-c0-s0-nm-h0_zs_eval",    
        "Tr0-c1-s1-nm-h0_zs_eval",    
        
        "random" ,        
        "diff-ik"
    ],
    "zs_names": [
        "Tf", "Tf-cont", "Tf-noCE", "Tf-noSL", "Mlp", "Mlp+SlCe", "Rand", "Diff-IK"
    ],
    "zs_metrics": None
}


def get_exp_names(benchmark):
    benchmark_cfg_file = get_benchmark_cfgs_dir() / f"{benchmark}.yaml"
    assert benchmark_cfg_file.exists(), f"Benchmark config file {benchmark_cfg_file} does not exist"

    with open(benchmark_cfg_file, 'r') as f:
        benchmark_cfg = yaml.safe_load(f)
    
    if "reach" in benchmark:
        exp_names = deepcopy(DEFAULT_REACH_EXP_NAMES)
    else:
        exp_names = deepcopy(DEFAULT_PUSH_EXP_NAMES)
        if benchmark in ["intra_simple_bot_push_simple", "inter_prims_push_simple"]:
            exp_names['mt'][0] = "Tr0-c0-s0-t-nt-h0_mt-eval"
            exp_names['zs'][0] = "Tr0-c0-s0-t-nt-h0_zs_eval"
        
        
    # zs metric will be robo_cat_variation_task
    test_robo_cat = benchmark_cfg['TEST_ENVS']['ROBOTS']
    test_variations = benchmark_cfg['TEST_ENVS']['VARIATIONS']
    test_task = benchmark_cfg['TEST_ENVS']['TASKS']
    if isinstance(test_task, str):
        test_task = [test_task] * len(test_robo_cat)
    assert len(test_robo_cat) == len(test_variations) == len(test_task)

    exp_names['zs_metrics'] = [f"{cat}_{var}_{task}" for cat, var, task in zip(test_robo_cat, test_variations, test_task)]
    # exp_names['zs_metrics'] = benchmark_cfg['TEST_ENVS']['VARIATIONS']

    train_robo_cat = benchmark_cfg['MULTIENV']['ROBOTS']
    train_variations = benchmark_cfg['MULTIENV']['VARIATIONS']
    train_task = benchmark_cfg['MULTIENV']['TASKS']
    if isinstance(train_task, str):
        train_task = [train_task] * len(train_robo_cat)
    assert len(train_robo_cat) == len(train_variations) == len(train_task)

    exp_names['mt_metrics'] = [f"{cat}_{var}_{task}" for cat, var, task in zip(train_robo_cat, train_variations, train_task)]

    return exp_names


def get_ft_exp_names(exp_names):
    zs_agent_names = exp_names['zs'][:2]
    ft10, ft10_names = [], []
    ft30, ft30_names = [], []
    ft50, ft50_names = [], []

    for idx, name in enumerate(zs_agent_names):
        base_name = name.replace("_zs_eval", "")
        ft10.append(f"{base_name}_ft_zs_eval_10000")
        ft30.append(f"{base_name}_ft_zs_eval_30000")
        ft50.append(f"{base_name}_ft_zs_eval_50000")
        good_name = exp_names['zs_names'][idx]
        ft10_names.append(good_name + "-ft10")
        ft30_names.append(good_name + "-ft30")
        ft50_names.append(good_name + "-ft50")

    exp_names['zs'] = exp_names['zs'] + ft10 + ft30 + ft50
    exp_names['zs_names'] = exp_names['zs_names'] + ft10_names + ft30_names + ft50_names
    
    return exp_names


REACH_TASKS_DICT = {k: (v, get_exp_names(v)) for k, v in REACH_BENCHMARKS.items()}
PUSH_TASKS_DICT = {k: (v, get_exp_names(v)) for k, v in PUSH_BENCHMARKS.items()}

REACH_FINETUNE_DICT = {k: (v, get_ft_exp_names(get_exp_names(v))) for k, v in REACH_FINETUNE_BENCHMARKS.items()}
PUSH_FINETUNE_DICT = {k: (v, get_ft_exp_names(get_exp_names(v))) for k, v in PUSH_FINETUNE_BENCHMARKS.items()}

ABLATION_EXP_NAMES['zs_metrics'] = REACH_TASKS_DICT['Arm3'][1]['zs_metrics']  # use the same metrics as reach task
ABLATION_EXP_NAMES['mt_metrics'] = REACH_TASKS_DICT['Arm3'][1]['mt_metrics']
ABLATIONS_TASKS_DICT = {
    "Arm3 (ablation)": ("intra_simple_bot_reach", ABLATION_EXP_NAMES),
}