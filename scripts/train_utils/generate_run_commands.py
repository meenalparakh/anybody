import json
import argparse
from anybody.utils.path_utils import get_experiment_scripts_dir, get_benchmark_cfgs_dir
import yaml
from pathlib import Path

short_names = {
    "intra_simple_bot_reach": "1", 
    "intra_simple_bot_push_simple": "2", 
    "intra_panda_reach": "3", 
    "intra_panda_push_simple": "4", 
    "inter_arms_reach": "5", 
    "inter_arms_push_simple": "6", 
    "inter_ee_arm_reach": "7", 
    "inter_ee_arm_push_simple": "8", 
    "inter_prims_reach": "9", 
    "inter_prims_push_simple": "10", 
    "inter_task_ur5": "11", 
    "intra_simple_bot_reach_v2": "12", 
    "intra_simple_bot_push_simple_v2": "13", 
    "inter_arms_reach_v2": "14", 
    "inter_arms_push_simple_v2": "15"
}


all_benchmark_names = get_benchmark_cfgs_dir() / "all_tasks.json"

with open(all_benchmark_names, "r") as f:
    all_benchmark_names = json.load(f)


def get_cfg_name(benchmark_name, model_type, se=False):
    if se:
        if model_type == "mlp":
            return "experiment_cfgs/se_mlp.yaml"
        elif model_type == "tf":
            task = "reach" if "reach" in benchmark_name else "push"
            return f"experiment_cfgs/se_tf_{task}.yaml"
    else:
        task = "reach" if "reach" in benchmark_name else "push"
        return f"experiment_cfgs/mt_{model_type}_{task}.yaml"

def get_task_info(benchmark):

    benchmark_cfg_file = get_benchmark_cfgs_dir() / f"{benchmark}.yaml"
    assert benchmark_cfg_file.exists(), f"Benchmark config file {benchmark_cfg_file} does not exist"

    with open(benchmark_cfg_file, 'r') as f:
        benchmark_cfg = yaml.safe_load(f)


    n_train_subtasks = len(benchmark_cfg['MULTIENV']['ROBOTS'])
    n_test_subtasks = len(benchmark_cfg['TEST_ENVS']['ROBOTS'])
    n_subtasks = n_train_subtasks + n_test_subtasks

    robots = benchmark_cfg['MULTIENV']['ROBOTS'] + benchmark_cfg['TEST_ENVS']['ROBOTS']
    variations = benchmark_cfg['MULTIENV']['VARIATIONS'] + benchmark_cfg['TEST_ENVS']['VARIATIONS']
    train_tasks = benchmark_cfg['MULTIENV']['TASKS'] 
    test_tasks = benchmark_cfg['TEST_ENVS']['TASKS']

    if isinstance(train_tasks, str):
        train_tasks = [train_tasks] * n_train_subtasks
    if isinstance(test_tasks, str):
        test_tasks = [test_tasks] * n_test_subtasks
        
    tasks = train_tasks + test_tasks
    assert len(robots) == len(variations) == len(tasks), "Robots, variations and tasks must have the same length in the benchmark config"
    
    project_name = benchmark_cfg['PROJECT_NAME']


    return {
        'robots': robots,
        'variations': variations,
        'tasks': tasks,
        'n_subtasks': n_subtasks,
        'n_train_subtasks': n_train_subtasks,
        'project_name': project_name,
    }


if __name__ == "__main__":
    # take as input the task, the benchmark cfg
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark config to run")
    parser.add_argument("--output_file", type=str, default=None, help="Output file to write commands to")
    parser.add_argument("--neuronic", action='store_true', help="If set, use neuronic cluster script")
    parser.add_argument("--project_dir", type=str, default="/n/fs/pvl-procur/anybody", help="Project directory on the cluster")

    args = parser.parse_args()

    is_neuronic = args.neuronic
    slurm_script = "./docker/cluster/submit_job_neuronic.sh"
    if not is_neuronic:
        slurm_script = "./docker/cluster/submit_job_ionic.sh"
    project_dir = args.project_dir

    seeds = [42, 23, 34]
    
    args = parser.parse_args()
    
    
    if args.benchmark == 'all':
        benchmarks = all_benchmark_names 
    else:
        benchmarks = [args.benchmark]

    total_runs = 0

    for benchmark in benchmarks:
        
        args.benchmark = benchmark

        if (not args.output_file) or (len(benchmarks) > 1):
            args.output_file = "run_" + args.benchmark
            
        output_path = get_experiment_scripts_dir() / (args.output_file + ".sh")
            
        commands = []
        
        # MT runs
        # example_command: python scripts/run.py --headless BENCHMARK_TASK intra_simple_bot_reach OVERRIDE_CFGNAME experiment_cfgs/mt_mlp_reach.yaml

        short_name = short_names[args.benchmark]

        RUN_TEMPLATE = f"{slurm_script} {short_name}_1 {project_dir} COMMAND"
        # mt tf run
        for seed in seeds:
            cfg_name = get_cfg_name(args.benchmark, "tf", se=False)
            cmd = f"scripts/run.py --headless BENCHMARK_TASK {args.benchmark} OVERRIDE_CFGNAME {cfg_name} RUN_SEED {seed}"
            commands.append(RUN_TEMPLATE.replace("COMMAND", cmd))

        RUN_TEMPLATE = f"{slurm_script} {short_name}_2 {project_dir} COMMAND"        
        # mt mlp run
        for seed in seeds:
            cfg_name = get_cfg_name(args.benchmark, "mlp", se=False)
            cmd = f"scripts/run.py --headless BENCHMARK_TASK {args.benchmark} OVERRIDE_CFGNAME {cfg_name} RUN_SEED {seed}"
            commands.append(RUN_TEMPLATE.replace("COMMAND", cmd))

        # SE run
        # example command: python scripts/run.py --headless OVERRIDE_CFGNAME experiment_cfgs/se.yaml SE_TASK simple_bot/r40_v1/reach RUN_SEED 42 PROJECT_NAME isbrv EXPERIMENT_NAME simple_bot-r40_v1-reach-42 
        task_info = get_task_info(args.benchmark)    
        total_num_envs = task_info['n_train_subtasks'] * 128            # MT runs have 128 envs per task
        
        # for reach, we use diff-ik baselines.
        for seed in seeds[:1]:
            for robot, var, task in zip(task_info['robots'], task_info['variations'], task_info['tasks']):
                
                ts = 200000 if "reach" in args.benchmark else 1000000

                base_cmd = f"scripts/run.py --headless SE_TASK {robot}/{var}/{task} RUN_SEED {seed} PROJECT_NAME {task_info['project_name']} TRAIN.NUM_ENVS_PER_TASK {total_num_envs} TRAINER.TIMESTEPS {ts}"
                # EXPERIMENT_NAME {robot}-{var}-{task}-{seed} OVERRIDE_CFGNAME experiment_cfgs/se.yaml
                
                if ("reach" not in args.benchmark) or (args.benchmark in ['intra_simple_bot_reach', 'intra_panda_reach']):
                    # se mlp run
                    
                    RUN_TEMPLATE = f"{slurm_script} {short_name}_3 {project_dir} COMMAND"
                    cfg_name = get_cfg_name(args.benchmark, "mlp", se=True)
                    cmd = f"{base_cmd} EXPERIMENT_NAME {robot}-{var}-{task}-{seed}-mlp OVERRIDE_CFGNAME {cfg_name}"
                    commands.append(RUN_TEMPLATE.replace("COMMAND", cmd))

                if (args.benchmark in ['intra_simple_bot_reach', 'intra_panda_reach']):
                    # se tf run       (only for reach task)
                    RUN_TEMPLATE = f"{slurm_script} {short_name}_4 {project_dir} COMMAND"
                    cfg_name = get_cfg_name(args.benchmark, "tf", se=True)
                    cmd = f"{base_cmd} EXPERIMENT_NAME {robot}-{var}-{task}-{seed}-tf OVERRIDE_CFGNAME {cfg_name}"
                    commands.append(RUN_TEMPLATE.replace("COMMAND", cmd))
                
                
        print(f"Writing {len(commands)} commands to {output_path}")
        with open(output_path, 'w') as f:
            f.write("#!/bin/bash\n\n")
            for cmd in commands:
                f.write(cmd + "\n")
                
        total_runs += len(commands)

    print(f"Total runs: {total_runs}")