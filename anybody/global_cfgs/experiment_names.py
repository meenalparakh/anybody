from anybody.utils.path_utils import get_benchmark_cfgs_dir
import yaml

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

def get_exp_names(benchmark, hd=False):
    benchmark_cfg_file = get_benchmark_cfgs_dir() / f"{benchmark}.yaml"
    assert benchmark_cfg_file.exists(), f"Benchmark config file {benchmark_cfg_file} does not exist"

    with open(benchmark_cfg_file, 'r') as f:
        benchmark_cfg = yaml.safe_load(f)
    
    if "reach" in benchmark:
        exp_names = DEFAULT_REACH_EXP_NAMES.copy()
        exp_names['zs_metrics'] = benchmark_cfg['TEST_ENVS']['VARIATIONS']

    return exp_names


reach_tasks = {
    "Arm3": ("intra_simple_bot_reach", intra_simple_bot_reach),
    "Arm3 (Ob)": ("intra_simple_bot_reach_v2", intra_simple_bot_reach_v2),
    "Arm3 (Ob, Hd)": ("intra_simple_bot_reach_v2", intra_simple_bot_reach_v2_pcd),
    "Panda": ("intra_panda_reach", intra_panda_reach),
    "Panda_supp": ("inter_arms_reach_v2", intra_panda_reach_supp),
    "EE-Arm": ("inter_ee_arm_reach", inter_ee_arm_reach),
    "Prims": ("inter_prims_reach", inter_prims_reach),
    "Arms": ("inter_arms_reach", inter_arms_reach),
}




intra_panda_reach = {
    "mt": [
        "Tr1-c1-s1-t-nt-h0_mt-eval",
        "Tr0-c0-s0-nm-h0_mt-eval",
        "TInd_mt-eval",
        "Trandom"        
    ],
    "mt_names": [
        "Tf", "Mlp", "Ind", "Rand"
    ],
    "zs": [
        "Tr1-c1-s1-t-nt-h0_eval",
        "Tr0-c0-s0-nm-h0_eval",
        None,
        "Trandom_test"
    ],
    "zs_names": [
        "Tf", "Mlp", "Ind", "Rand"
    ],
    "zs_metrics": ["Panda"]
}

intra_panda_reach_supp = {
    "zs": [
        None,
        None,
        "TInd_mt-eval",
        None
    ]
}

intra_simple_bot_reach_v2 = {
    "mt": [
        "Tr1-c1-s1-t-nt-h0_mt-eval",     # checked
        "Tr0-c0-s0-nm-h0_mt-eval",    # checked
        "TInd_mt-eval", # checked
        "Trandom"         # checked
    ],
    "mt_names": [
        "Tf", "Mlp", "Ind", "Rand"
    ],
    "zs": [
        "Tr1-c1-s1-t-nt-h0_eval",   # checked
        "Tr0-c0-s0-nm-h0_eval",    # checked
        "lInd_test_mt-eval",     # checked
        "Trandom_test" # checked
    ],
    "zs_names": [
        "Tf", "Mlp", "Ind", "Rand"
    ],
    "zs_metrics": ["R53"]
}


intra_simple_bot_reach_v2_pcd = {
    "mt": [
        "Tr1-c1-s1-t-nt-h1_mt-eval",   # checked
        "Tr0-c0-s0-nm-h1_mt-eval",  # checked
        "TInd_hd_mt-eval",     ############## missing  - running
        "Trandom"         # checked
    ],
    "mt_names": [
        "Tf", "Mlp", "Ind", "Rand"
    ],
    "zs": [
        "Tr1-c1-s1-t-nt-h1_eval",   # checked
        "Tr0-c0-s0-nm-h1_eval",     # checked
        "TInd_test_hd_mt-eval",      # checked
        "Trandom_test"      # checked
    ],
    "zs_names": [
        "Tf", "Mlp", "Ind", "Rand"
    ],
    "zs_metrics": ["R53"]
}

###### for now use the old results
inter_arms_reach = {
    "mt": [
        "Tr1-c1-s1-t-nt-h0_mt-eval",   # checked
        "Tr0-c0-s0-nm-h0_mt-eval",  # missing  - running
        "lInd_mt-eval",    # checked
        "Trandom"         # checked
    ],
    "mt_names": [
        "Tf", "Mlp", "Ind", "Rand"
    ],
    "zs": [
        "Tr1-c1-s1-t-nt-h0_eval",   # checked
        "Tr0-c0-s0-nm-h0_eval",     # checked
        "lInd_test_mt-eval",      # checked
        "Trandom_test"      # checked
    ],
    "zs_names": [
        "Tf", "Mlp", "Ind", "Rand"
    ],
    "zs_metrics": ["Widowx"]  
}

inter_prims_reach = {
    "mt": [
        "Tr1-c1-s1-t-nt-h0_mt-eval",
        "Tr0-c0-s0-nm-h0_mt-eval",
        "TInd_mt-eval",    ############# missing  - running
        "Trandom"        
    ],
    "mt_names": [
        "Tf", "Mlp", "Ind", "Rand"
    ],
    "zs": [
        "Tr1-c1-s1-t-nt-h0_eval",
        "Tr0-c0-s0-nm-h0_eval",
        "TInd_test_mt-eval",
        "Trandom_test"
    ],
    "zs_names": [
        "Tf", "Mlp", "Ind", "Rand"
    ],
    "zs_metrics": ["Chain_3b_2", "Chain_2b_0"]
}

inter_ee_arm_reach = {
    "mt": [
        "Tr1-c1-s1-t-nt-h0_mt-eval",    # checked
        "Tr0-c0-s0-nm-h0_mt-eval",   # checked
        "TInd_mt-eval",    ############# missing
        "Trandom"        # checked
    ],
    "mt_names": [
        "Tf", "Mlp", "Ind", "Rand"
    ],
    "zs": [
        "Tr1-c1-s1-t-nt-h0_eval",     # checked
        "Tr0-c0-s0-nm-h0_eval",     # checked
        "TInd_test_mt-eval",   # checked 
        "Trandom_test"   # checked
    ],
    "zs_names": [
        "Tf", "Mlp", "Ind", "Rand"
    ],
    "zs_metrics": ["Ur5_stick"]
}





intra_simple_bot_push = {
    "mt": [
        "Dr0-c0-s0-t-nt-h0_mt-eval",    # ready (close to 800k)
        "Tr0-c0-s0-nm-h0_mt-eval",   # use checkpoint before 400k
        "TInd_mt-eval",     # ready
    ],
    "mt_names": [
        "Tf", "Mlp", "Ind"
    ],
    "zs": [
        "Dr0-c0-s0-t-nt-h0_eval",     # ready (close to 800k)
        "Tr0-c0-s0-nm-h0_eval",     # use checkpoint before 400k
        "TInd_test_mt-eval",   #  ready
    ],
    "zs_names": [
        "Tf", "Mlp", "Ind"
    ],
    "zs_metrics": ["R17"]
}
        
intra_simple_bot_push_v2 = {
    "mt": [
        "Dr0-c0-s0-t-nt-h0_mt-eval",    # ready (close to 1M)
        "Tr0-c0-s0-nm-h0_mt-eval",   # ready
        "lInd_mt-eval",     # ready   
    ],
    "mt_names": [
        "Tf", "Mlp", "Ind"
    ],
    "zs": [
        "Dr0-c0-s0-t-nt-h0_eval",     # ready (close to 1M)
        "Tr0-c0-s0-nm-h0_eval",     # ready
        "lInd_test_mt-eval",   #  ready
    ],
    "zs_names": [
        "Tf", "Mlp", "Ind"
    ],
    "zs_metrics": ["R16"]
}

        
intra_simple_bot_push_v2_pcd = {
    "mt": [
        "Dr0-c0-s0-t-nt-h1_mt-eval",    # notready (close to 400k)
        "Tr0-c0-s0-nm-h1_mt-eval",   # ready 
        "TInd_hd_mt-eval",     # ready   
    ],
    "mt_names": [
        "Tf", "Mlp", "Ind"
    ],
    "zs": [
        "Dr0-c0-s0-t-nt-h1_eval",     # notready (close to 400k)
        "Tr0-c0-s0-nm-h1_eval",     # ready
        "TInd_test_hd_mt-eval",   #  ready
    ],
    "zs_names": [
        "Tf", "Mlp", "Ind"
    ],
    "zs_metrics": ["R16"]
}

intra_panda_push_simple = {
    "mt": [
        "Dr0-c0-s0-t-nt-h0_mt-eval",    # ready (close to 600k)
        "Tr0-c0-s0-nm-h0_mt-eval",   # ready 
        "TInd_mt-eval",     # ready   
    ],
    "mt_names": [
        "Tf", "Mlp", "Ind"
    ],
    "zs": [
        "Dr0-c0-s0-t-nt-h0_eval",     # ready (close to 600k)
        "Tr0-c0-s0-nm-h0_eval",     # ready
        "TInd_test_mt-eval",   #  ready
    ],
    "zs_names": [
        "Tf", "Mlp", "Ind"
    ],
    "zs_metrics": ["Panda"]
}
intra_panda_push_simple_supp = {
    "zs": [
        None,
        None,
        "DInd_mt-eval",
    ]
}


inter_arms_push_main = {
    "mt": [
        "Dr0-c0-s0-t-nt-h0_mt-eval",    # ready (close to 800k)
        "Dr0-c0-s0-nm-h0_mt-eval",   # ready 
        "DInd_mt-eval",     # ready   
    ],
    "mt_names": [
        "Tf", "Mlp", "Ind"
    ],
    "zs": [
        "Dr0-c0-s0-t-nt-h0_eval",     # ready (close to 600k)
        "Dr0-c0-s0-nm-h0_eval",     # ready
        None,   #  ready
        # "TInd_test_mt-eval",   #  ready
    ],
    "zs_names": [
        "Tf", "Mlp", "Ind"
    ],
    "zs_metrics": ["Widowx"]
}

inter_arms_push_supp = {
    "mt": [
        None,    # ready (close to 600k)
        None,
        "lInd_mt-eval",     # ready   
    ],
    "zs": [
        None,
        None,
        "lInd_test_mt-eval",   #  ready
    ]
}

inter_prims_push = {
    "mt": [
        "Dr0-c0-s0-t-nt-h0_mt-eval",    # notready (close to 250k)
        "Dr0-c0-s0-nm-h0_mt-eval",   # ready 
        "TInd_mt-eval",     # ready   (Nlink missing)
    ],
    "mt_names": [
        "Tf", "Mlp", "Ind"
    ],
    "zs": [
        "Dr0-c0-s0-t-nt-h0_eval",     # notready (close to 250k)
        "Dr0-c0-s0-nm-h0_eval",     # ready
        "TInd_test_mt-eval",   #  ready
    ],
    "zs_names": [
        "Tf", "Mlp", "Ind"
    ],
    "zs_metrics": ["Chain_2b_4", "Chain_3b_1"]
}

inter_ee_arm_push = {
    "mt": [
        "Dr0-c0-s0-t-nt-h0_mt-eval",    # ready (close to 600k)
        "Tr0-c0-s0-nm-h0_mt-eval",   # ready 
        "TInd_mt-eval",     # ready   (Nlink missing)
    ],
    "mt_names": [
        "Tf", "Mlp", "Ind"
    ],
    "zs": [
        "Dr0-c0-s0-t-nt-h0_eval",     # ready (close to 600k)
        "Tr0-c0-s0-nm-h0_eval",     # exit
        "TInd_test_mt-eval",   #  ready
    ],
    "zs_names": [
        "Tf", "Mlp", "Ind"
    ],
    "zs_metrics": ["Ur5_stick"]
}

inter_task_ur5 = {
    "mt": [
        "Dr0-c0-s0-t-nt-h0_mt-eval",    # ready (close to 800k)
        "Tr0-c0-s0-nm-h0_mt-eval",   # ready 
        "lInd_mt-eval",     # ready   
    ],
    "mt_names": [
        "Tf", "Mlp", "Ind"
    ],
    "zs": [
        "Dr0-c0-s0-t-nt-h0_eval",     # ready (close to 600k)
        "Tr0-c0-s0-nm-h0_eval",     # ready
        "lInd_test_mt-eval",   #  ready
    ],
    "zs_names": [
        "Tf", "Mlp", "Ind"
    ],
    "zs_metrics": ["Ur5_planar_push"]
}

# reach_tasks = {
#     "Arm3": ("intra_simple_bot_reach_15", intra_simple_bot_reach),
#     "Arm3 (Ob)": ("intra_simple_bot_reach_v2_15", intra_simple_bot_reach_v2),
#     "Arm3 (Ob, Hd)": ("intra_simple_bot_reach_v2_15", intra_simple_bot_reach_v2_pcd),
#     "Panda": ("intra_panda_reach_15", intra_panda_reach),
#     "Panda_supp": ("inter_arms_reach_v2_15", intra_panda_reach_supp),
#     "EE+Arm": ("inter_ee_arm_reach_15", inter_ee_arm_reach),
#     "Prims": ("inter_prims_reach_15", inter_prims_reach),
#     "Arms": ("inter_arms_reach_15", inter_arms_reach),
# }


intra_simple_bot_ablation = {
    "mt": [
        "TInd_mt-eval",     # exist
        "Tr0-c0-s0-nm-h0_mt-eval",    # exist
        "Tr0-c1-s1-nm-h0_mt-eval",   # exist
        "Tr0-c0-s0-nt-h0_mt-eval",   # exist
        "Tr0-c1-s1-nt-h0_mt-eval",   # exist
        "Tr1-c1-s1-nt-h0_mt-eval",   # exist
        "Tr1-c1-s1-t-nt-h0_mt-eval",   # exist
        "Trandom"         # exist
    ],
    "mt_names": [
        "Ind", "Mlp", "Mlp+Sl", "Tf", "Tf+Sl", "Tf+Sl+Dis", "Tf+Sl+Dis+T", "Rand"     
    ],
    "zs": [
        "TInd_test_mt-eval",     # exist
        "Tr0-c0-s0-nm-h0_eval",    # exist
        "Tr0-c1-s1-nm-h0_eval",   
        "Tr0-c0-s0-nt-h0_eval",   
        "Tr0-c1-s1-nt-h0_eval",   
        "Tr1-c1-s1-nt-h0_eval",   
        "Tr1-c1-s1-t-nt-h0_eval",   # exist
        "Trandom_test"         # exist
    ],
    "zs_names": [
        "Ind", "Mlp", "Mlp+Sl", "Tf", "Tf+Sl", "Tf+Sl+Dis", "Tf+Sl+Dis+T", "Rand"
    ],
    "zs_metrics": ["R99"]
}




push_tasks = {
    "Arm3": ("intra_simple_bot_push_simple", intra_simple_bot_push),
    "Arm3 (Ob)": ("intra_simple_bot_push_simple_v2", intra_simple_bot_push_v2),
    "Arm3 (Ob, Hd)": ("intra_simple_bot_push_simple_v2", intra_simple_bot_push_v2_pcd),
    "Panda": ("intra_panda_push_simple", intra_panda_push_simple),
    "Panda_supp": ("inter_arms_push_simple_v2", intra_panda_push_simple_supp),
    "EE-Arm": ("inter_ee_arm_push_simple", inter_ee_arm_push),
    "EE-Task": ("inter_task_ur5", inter_task_ur5),
    "Prims": ("inter_prims_push_simple", inter_prims_push),
    "Arms": ("inter_arms_push_simple_v2", inter_arms_push_main),
    "Arms_supp": ("inter_arms_push_simple", inter_arms_push_supp),
}

ablations_tasks = {
    "Arm3 (ablation)": ("intra_simple_bot_reach", intra_simple_bot_ablation),
}