import argparse
from anybody.utils.path_utils import get_experiment_scripts_dir, get_wandb_csv_dir, get_logs_dir, get_synced_slurm_logs_dir
from anybody.utils.wandb_utils_v2 import collect_runs, construct_train_df
from anybody.utils.utils import load_pickle
import yaml
from pathlib import Path
import os

def load_checkpoints(project_name, force=False, local_checkpoints=False):
    runs_info = collect_runs(project_name, force=force)
    
    # remove certain groups
    if "random" in runs_info:
        del runs_info["random"]
    
    checkpoint_dict = {}
    for group in runs_info:
        
        # skip certain groups
        if "eval" in group.lower():
            continue
        
        for run_id, (run_name, run_directory) in runs_info[group].items():
            
            logs_dir = get_synced_slurm_logs_dir() if local_checkpoints else get_logs_dir()
            ckpt_dir = logs_dir / project_name / run_name / "checkpoints"
            if not ckpt_dir.exists():
                print(f"Checkpoint dir {ckpt_dir} does not exist, skipping")
                continue

            saved_ckpts = os.listdir(ckpt_dir)
            timesteps = [int(ckpt.split("_")[-1].split(".")[0]) for ckpt in saved_ckpts]
            # collect the latest checkpoint
            if len(timesteps) == 0:
                print(f"No checkpoints found in {ckpt_dir}, skipping")
                continue
            latest_timestep = max(timesteps)

            if latest_timestep < 1000000:
                continue
            
            checkpoint_directory = ckpt_dir / f"agent_{latest_timestep}.pt"
            # print(f"Run ID: {run_id}/{run_name} - Latest checkpoint at timestep {latest_timestep}: {checkpoint_directory}")
            checkpoint_dict[run_id] = (run_name,checkpoint_directory)

    return checkpoint_dict



if __name__ == "__main__":
    # take as input the task, the benchmark cfg
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark config to run")
    parser.add_argument("--output_file", type=str, default=None, help="Output file to write commands to")
    parser.add_argument("--neuronic", action='store_true', help="If set, use neuronic cluster script")
    parser.add_argument("--project_dir", type=str, default="/n/fs/pvl-exptrack/anybody", help="Project directory on the cluster")
    parser.add_argument("--job_name", type=str, default="test_run", help="Job name for the cluster job")
    parser.add_argument("--cluster_local_eval", action='store_true', help="If set, evaluate cluster runs locally.")
    parser.add_argument("--force", action='store_true', help="If set, force re-collection of runs from wandb")
    parser.add_argument("--view_runs", action='store_true', help="If set, just view the runs collected from wandb and exit")

    args = parser.parse_args()

    if args.view_runs:
        df = construct_train_df(args.benchmark)
        exit()


    is_neuronic = args.neuronic
    slurm_script = "./docker/cluster/submit_job_neuronic.sh"
    run_dir = "neuronic"
    if not is_neuronic:
        slurm_script = "./docker/cluster/submit_job_ionic.sh"
        run_dir = "ionic"
    if args.cluster_local_eval:
        slurm_script = ""  # no slurm script needed for local evalation
        run_dir = "local"
    project_dir = args.project_dir
    job_name = args.job_name
    
    args = parser.parse_args()
    
    if not args.output_file:
        args.output_file = f"{run_dir}/eval_" + args.benchmark
    output_path = get_experiment_scripts_dir() / (args.output_file + ".sh")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # for evaluating mt model, example command:
    # evaluation on MT means: evaluating on the train tasks. This applies even for single-task models.
    commands = []    

    # seeds = [23, 34, 42]
    seeds = [42]

    RUN_TEMPLATE = f"{slurm_script} test4 {project_dir} COMMAND"    
    if args.cluster_local_eval:
        RUN_TEMPLATE = "python COMMAND"
    
    ckpt_dict = load_checkpoints(args.benchmark, force=args.force, local_checkpoints=args.cluster_local_eval)
    print(f"Found {len(ckpt_dict)} checkpoints for benchmark {args.benchmark}")

    for run_id, (run_name, ckpt_path) in ckpt_dict.items():
        print(f"Run ID: {run_id} - Run Name: {run_name} - Checkpoint Path: {ckpt_path}")
        cmd = f"scripts/run.py --headless --enable_cameras OVERRIDE_CFGNAME experiment_cfgs/eval_mt.yaml EVAL_CHECKPOINT {ckpt_path} EVAL_CLUSTER_ON_LOCAL {args.cluster_local_eval}"
        commands.append(RUN_TEMPLATE.replace("COMMAND", cmd))        

    with open(output_path, 'w') as f:
        f.write("#!/bin/bash\n\n")
        for cmd in commands:
            f.write(cmd + "\n")