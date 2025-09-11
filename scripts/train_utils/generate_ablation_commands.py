import json
import argparse
from anybody.utils.path_utils import get_experiment_scripts_dir, get_benchmark_cfgs_dir
import yaml
from pathlib import Path

short_names = {
    "ablation": "18"
}

cfg_names = {
    # "mt_tf": "ablation_cfgs/mt_tf.yaml",    # already run in main experiments
    "mt_tf_no_ce": "ablation_cfgs/mt_tf_no_ce.yaml",
    "mt_tf_no_sl": "ablation_cfgs/mt_tf_no_sl.yaml",
    "mt_tf_no_disc": "ablation_cfgs/mt_tf_no_disc.yaml",
    # "mt_mlp": "ablation_cfgs/mt_mlp.yaml",  # already run in main experiments
    "mt_mlp_sl_ce": "ablation_cfgs/mt_mlp_sl_ce.yaml",
}


if __name__ == "__main__":
    # take as input the task, the benchmark cfg
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_file", type=str, default=None, help="Output file to write commands to")
    parser.add_argument("--neuronic", action='store_true', help="If set, use neuronic cluster script")
    parser.add_argument("--project_dir", type=str, default="/n/fs/pvl-procur/anybody", help="Project directory on the cluster")
    parser.add_argument("--run_types", type=str, default="all", help="Type of runs to generate commands for.")

    args = parser.parse_args()

    is_neuronic = args.neuronic
    slurm_script = "./docker/cluster/submit_job_neuronic.sh"
    run_dir = "neuronic"
    if not is_neuronic:
        slurm_script = "./docker/cluster/submit_job_ionic.sh"
        run_dir = "ionic"
    project_dir = args.project_dir

    if args.run_types == 'all':
        run_types = list(cfg_names.keys())
    else:
        run_types = [args.run_types]      # e.g. mt-tf_no_ce_mt_mlp_sl_ce

    print(f"Generating commands for run types: {run_types}")
    seeds = [42, 23, 34]
    benchmark_task = "intra_simple_bot_reach"

    total_runs = 0

    if (not args.output_file):
        args.output_file = run_dir + "/ablation"
            
        output_path = get_experiment_scripts_dir() / (args.output_file + ".sh")
        output_path.parent.mkdir(parents=True, exist_ok=True)
            
        commands = []
        
        # MT runs
        # example_command: python scripts/run.py --headless BENCHMARK_TASK intra_simple_bot_reach OVERRIDE_CFGNAME experiment_cfgs/mt_mlp_reach.yaml

        short_name = short_names["ablation"]

        for idx, run_name in enumerate(run_types):
            if run_name not in cfg_names:
                print(f"Run type {run_name} not recognized. Skipping.")
                continue

            RUN_TEMPLATE = f"{slurm_script} {short_name}_{idx} {project_dir} COMMAND"
            # mt tf run
            for seed in seeds:
                cfg_name = cfg_names[run_name]
                cmd = f"scripts/run.py --headless BENCHMARK_TASK {benchmark_task} OVERRIDE_CFGNAME {cfg_name} RUN_SEED {seed}"
                commands.append(RUN_TEMPLATE.replace("COMMAND", cmd))

                
        print(f"Writing {len(commands)} commands to {output_path}")
        with open(output_path, 'w') as f:
            f.write("#!/bin/bash\n\n")
            for cmd in commands:
                f.write(cmd + "\n")
                
        total_runs += len(commands)

    print(f"Total runs: {total_runs}")