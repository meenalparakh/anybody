import argparse
from anybody.utils.path_utils import get_experiment_scripts_dir, get_benchmark_cfgs_dir
import yaml
from pathlib import Path


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
        'project_name': project_name
    }


if __name__ == "__main__":
    # take as input the task, the benchmark cfg
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark config to run")
    parser.add_argument("--output_file", type=str, default=None, help="Output file to write commands to")
    parser.add_argument("--neuronic", action='store_true', help="If set, use neuronic cluster script")
    parser.add_argument("--project_dir", type=str, default="/n/fs/pvl-exptrack/anybody", help="Project directory on the cluster")
    parser.add_argument("--job_name", type=str, default="test_run", help="Job name for the cluster job")

    args = parser.parse_args()

    is_neuronic = args.neuronic
    slurm_script = "./docker/cluster/submit_job_neuronic.sh"
    if not is_neuronic:
        slurm_script = "./docker/cluster/submit_job_ionic.sh"
    project_dir = args.project_dir
    job_name = args.job_name


    # seeds = [23, 34, 42]
    seeds = [42]
    
    args = parser.parse_args()
    
    if not args.output_file:
        args.output_file = "run_" + args.benchmark
        
    output_path = get_experiment_scripts_dir() / (args.output_file + ".sh")
        
    commands = []

    # MT runs
    # example_command: python scripts/run.py --headless BENCHMARK_TASK intra_simple_bot_reach OVERRIDE_CFGNAME experiment_cfgs/mt_mlp_reach.yaml

    RUN_TEMPLATE = f"{slurm_script} test1 {project_dir} COMMAND"
    # mt tf run
    for seed in seeds:
        cmd = f"scripts/run.py --headless BENCHMARK_TASK {args.benchmark} OVERRIDE_CFGNAME experiment_cfgs/mt_tf_reach.yaml RUN_SEED {seed}"
        commands.append(RUN_TEMPLATE.replace("COMMAND", cmd))

    RUN_TEMPLATE = f"{slurm_script} test2 {project_dir} COMMAND"        
    # mt mlp run
    for seed in seeds:
        cmd = f"scripts/run.py --headless BENCHMARK_TASK {args.benchmark} OVERRIDE_CFGNAME experiment_cfgs/mt_mlp_reach.yaml RUN_SEED {seed}"
        commands.append(RUN_TEMPLATE.replace("COMMAND", cmd))
        
    RUN_TEMPLATE = f"{slurm_script} test3 {project_dir} COMMAND"        
    # SE run
    # example command: python scripts/run.py --headless OVERRIDE_CFGNAME experiment_cfgs/se.yaml SE_TASK simple_bot/r40_v1/reach RUN_SEED 42 PROJECT_NAME isbrv EXPERIMENT_NAME simple_bot-r40_v1-reach-42 
    task_info = get_task_info(args.benchmark)    
    for seed in seeds:
        for robot, var, task in zip(task_info['robots'], task_info['variations'], task_info['tasks']):
            base_cmd = f"scripts/run.py --headless SE_TASK {robot}/{var}/{task} RUN_SEED {seed} PROJECT_NAME {task_info['project_name']}"
            # EXPERIMENT_NAME {robot}-{var}-{task}-{seed} OVERRIDE_CFGNAME experiment_cfgs/se.yaml
            
            # se mlp run
            cmd = f"{base_cmd} EXPERIMENT_NAME {robot}-{var}-{task}-{seed}-mlp OVERRIDE_CFGNAME experiment_cfgs/se_mlp.yaml"
            commands.append(RUN_TEMPLATE.replace("COMMAND", cmd))
            
            # se tf run
            cmd = f"{base_cmd} EXPERIMENT_NAME {robot}-{var}-{task}-{seed}-tf OVERRIDE_CFGNAME experiment_cfgs/se_tf_{task}.yaml"
            commands.append(RUN_TEMPLATE.replace("COMMAND", cmd))
            
            
    with open(output_path, 'w') as f:
        f.write("#!/bin/bash\n\n")
        for cmd in commands:
            f.write(cmd + "\n")