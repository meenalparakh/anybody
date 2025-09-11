from anybody.utils.path_utils import get_benchmark_cfgs_dir
import yaml

REACH_BENCHMARKS = {
    "Arm3": "intra_simple_bot_reach",
    "Arm3 (Ob)": "intra_simple_bot_reach_v2",
    "Arm3 (Ob, Hd)": "intra_simple_bot_reach_hd_v2",
    "Panda": "intra_panda_reach",
    "EE-Arm": "inter_ee_arm_reach",
    "Prims": "inter_prims_reach",
    "Arms": "inter_arms_reach_v2",
}

DEFAULT_REACH_EXP_NAMES = {
    "mt": [
        "Tr1-c1-s1-t-nt-h0_mt-eval",
        "Tr0-c0-s0-nm-h0_mt-eval",
        "IndTf_mt-eval",
        "IndMlp_mt-eval",
        "Rand"        
    ],
    "mt_names": [
        "Tf", "Mlp", "Se-Tf", "Se-Mlp", "Rand"
    ],
    "zs": [
        "Tr1-c1-s1-t-nt-h0_eval",
        "Tr0-c0-s0-nm-h0_eval",
        "IndTf_mt-eval",
        "IndMlp_mt-eval",
        "Rand"
    ],
    "zs_names": [
        "Tf", "Mlp", "Se-Tf", "Se-Mlp", "Rand"
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

DEFAULT_PUSH_EXP_NAMES = {
    "mt": [
        "Tr1-c1-s1-t-nt-h0_mt-eval",
        "Tr0-c0-s0-nm-h0_mt-eval",
        "IndMlp_mt-eval",
    ],
    "mt_names": [
        "Tf", "Mlp", "Se-Mlp"
    ],
    "zs": [
        "Tr1-c1-s1-t-nt-h0_eval",
        "Tr0-c0-s0-nm-h0_eval",
        "IndMlp_mt-eval",
    ],
    "zs_names": [
        "Tf", "Mlp", "Se-Mlp"
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
        
        "Rand"        
    ],
    "mt_names": [
        "Tf", "Tf-cont", "Tf-noCE", "Tf-noSL", "Mlp", "Mlp+SlCe", "Rand"
    ],
    "zs": [
        "Tr1-c1-s1-t-nt-h0_mt-eval",
        "Tr0-c1-s1-t-nt-h0_mt-eval",
        "Tr1-c0-s1-t-nt-h0_mt-eval",
        "Tr1-c1-s0-t-nt-h0_mt-eval",
        
        "Tr0-c0-s0-nm-h0_mt-eval",    
        "Tr0-c1-s1-nm-h0_mt-eval",    
        
        "Rand"         
    ],
    "zs_names": [
        "Tf", "Tf-cont", "Tf-noCE", "Tf-noSL", "Mlp", "Mlp+SlCe", "Rand"
    ],
    "zs_metrics": None
}

def get_exp_names(benchmark):
    benchmark_cfg_file = get_benchmark_cfgs_dir() / f"{benchmark}.yaml"
    assert benchmark_cfg_file.exists(), f"Benchmark config file {benchmark_cfg_file} does not exist"

    with open(benchmark_cfg_file, 'r') as f:
        benchmark_cfg = yaml.safe_load(f)
    
    if "reach" in benchmark:
        exp_names = DEFAULT_REACH_EXP_NAMES.copy()
    else:
        exp_names = DEFAULT_PUSH_EXP_NAMES.copy()
        
    exp_names['zs_metrics'] = benchmark_cfg['TEST_ENVS']['VARIATIONS']

    return exp_names


reach_tasks = {k: (v, get_exp_names(v)) for k, v in REACH_BENCHMARKS.items()}
push_tasks = {k: (v, get_exp_names(v)) for k, v in PUSH_BENCHMARKS.items()}

ABLATION_EXP_NAMES['zs_metrics'] = reach_tasks['Arm3'][1]['zs_metrics']  # use the same metrics as reach task
ablations_tasks = {
    "Arm3 (ablation)": ("intra_simple_bot_reach", ABLATION_EXP_NAMES),
}