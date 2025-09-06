import argparse
from anybody.utils.path_utils import get_experiment_scripts_dir, get_benchmark_cfgs_dir
import yaml
from pathlib import Path


def get_agent_name(task, backbone):
    if (task, backbone) == ("reach", "Tf"):
        return "Tr1-c1-s1-t-nt-h0"
    elif (task, backbone) == ("reach", "Mlp"):
        return "Tr0-c0-s0-nm-h0"
    raise NotImplementedError(f"Agent name for task {task} and backbone {backbone} not implemented")
        




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
    
    args = parser.parse_args()
    
    if not args.output_file:
        args.output_file = "eval_" + args.benchmark
    output_path = get_experiment_scripts_dir() / (args.output_file + ".sh")

    # for evaluating mt model, example command:
    # evaluation on MT means: evaluating on the train tasks. This applies even for single-task models.
    commands = []    

    # seeds = [23, 34, 42]
    seeds = [42]

    RUN_TEMPLATE = f"{slurm_script} test4 {project_dir} COMMAND"    

    timestep = 100000


    
    for seed in seeds:
        for backbone in ['Tf', 'Mlp']:
            agent_name = get_agent_name("reach", backbone)
            cmd = f"scripts/run.py --headless --enable_cameras OVERRIDE_CFGNAME experiment_cfgs/eval_mt.yaml EVAL_CHECKPOINT LOGS_PATH/{args.benchmark}/{agent_name}_{seed}/checkpoints/agent_{timestep}.pt"
            commands.append(RUN_TEMPLATE.replace("COMMAND", cmd))


    with open(output_path, 'w') as f:
        f.write("#!/bin/bash\n\n")
        for cmd in commands:
            f.write(cmd + "\n")