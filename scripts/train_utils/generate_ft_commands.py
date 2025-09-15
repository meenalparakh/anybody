import argparse
from anybody.utils.path_utils import get_experiment_scripts_dir, get_logs_dir, get_synced_slurm_logs_dir, get_benchmark_cfgs_dir
from anybody.utils.wandb_utils_v2 import collect_runs, construct_train_df, is_eval_run
import yaml
import json
import os

short_names = {
    "intra_simple_bot_reach": "1",        # done
    "intra_simple_bot_push_simple": "2",          # done
    "intra_panda_reach": "3",        # done
    "intra_panda_push_simple": "4",      # revisit -- till here fired on ionic
    # "inter_arms_reach": "5", 
    # "inter_arms_push_simple": "6",
    "inter_ee_arm_reach": "7",              # done      
    "inter_ee_arm_push_simple": "8",        # queued
    "inter_prims_reach": "9",              # done
    "inter_prims_push_simple": "10",           # done
    "inter_task_ur5": "11",                 # queued
    "intra_simple_bot_reach_v2": "12",       # revisit 
    "intra_simple_bot_push_simple_v2": "13",      # revisit
    "intra_simple_bot_reach_hd_v2": "14",          # done
    "intra_simple_bot_push_simple_hd_v2": "15",     # revisit
    "inter_arms_reach_v2": "16",          # remaining   - firing locally
    "inter_arms_push_simple_v2": "17",      # remaining
    "ablation": "18"
}

all_benchmark_names = get_benchmark_cfgs_dir() / "all_tasks.json"

with open(all_benchmark_names, "r") as f:
    all_benchmark_names = json.load(f)


def is_mt_run(ckpt_dir, run_name):
    if run_name.startswith("Tr"):
        return True
    else:
        return False
    
def load_checkpoints(project_name, force=False, local_checkpoints=False):
    runs_info = collect_runs(project_name, force=force)
    
    checkpoint_dict = {}
    
    # eval_runs = []
    # for group in runs_info:
    #     if "eval" in group.lower():
    #         runs = runs_info[group]
    #         for run_id, (run_name, _) in runs.items():
    #             eval_runs.append(run_name)

    for group in runs_info:
        
        # skip certain groups
        if is_eval_run(group):
            continue
        
        for run_id, (run_name, run_directory) in runs_info[group].items():
            
            
            # if run-name is already there, continue
            if run_name in [v[0] for v in checkpoint_dict.values()]:
                continue
            
            # logs_dir = get_synced_slurm_logs_dir() if local_checkpoints else get_logs_dir()
            logs_dir = get_synced_slurm_logs_dir()
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

            if latest_timestep < 300000:
                # print(f"Latest checkpoint timestep {project_name}: {group} {latest_timestep} is less than 500k, skipping")
                continue
            
            if latest_timestep < 1000000:
                with open("remaining_runs.txt", 'a') as fs:
                    fs.write(f"{short_name} {project_name}: {run_name} has {latest_timestep} steps.\n")
                
            
            checkpoint_directory = ckpt_dir / f"agent_{latest_timestep}.pt"
            # print(f"Run ID: {run_id}/{run_name} - Latest checkpoint at timestep {latest_timestep}: {checkpoint_directory}")

            cluster_ckpt_path = str(checkpoint_directory).replace(str(logs_dir), "LOGS_PATH")

            checkpoint_dict[run_id] = (run_name, cluster_ckpt_path)

    return checkpoint_dict



if __name__ == "__main__":
    # take as input the task, the benchmark cfg
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, required=True, help="Benchmark config to run")
    parser.add_argument("--output_file", type=str, default=None, help="Output file to write commands to")
    parser.add_argument("--neuronic", action='store_true', help="If set, use neuronic cluster script")
    parser.add_argument("--project_dir", type=str, default="/n/fs/pvl-procur/anybody", help="Project directory on the cluster")
    parser.add_argument("--job_name", type=str, default="test_run", help="Job name for the cluster job")
    parser.add_argument("--cluster_local_eval", action='store_true', help="If set, evaluate cluster runs locally.")
    parser.add_argument("--force", action='store_true', help="If set, force re-collection of runs from wandb")
    parser.add_argument("--view_runs", action='store_true', help="If set, just view the runs collected from wandb and exit")
    parser.add_argument("--single_script", action='store_true', help="If set, write all commands to a single script")

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

    
    if args.benchmark == 'all':
        benchmarks = all_benchmark_names 
    else:
        benchmarks = [args.benchmark]
        
    total_runs = 0
    
    if args.single_script:
        all_commands = []
        all_commands_output_path = get_experiment_scripts_dir() / f"{run_dir}/all.sh"

    for benchmark in benchmarks:
    
        args.benchmark = benchmark
        short_name = short_names[benchmark] if benchmark in short_names else "X"
    
        if (not args.output_file) or (len(benchmarks) > 1):
            args.output_file = f"{run_dir}/ft_" + args.benchmark
        output_path = get_experiment_scripts_dir() / (args.output_file + ".sh")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # for evaluating mt model, example command:
        # evaluation on MT means: evaluating on the train tasks. This applies even for single-task models.
        commands = []    

        RUN_TEMPLATE = f"{slurm_script} test_{short_name} {project_dir} COMMAND"    
        if args.cluster_local_eval:
            RUN_TEMPLATE = "python COMMAND"
        
        ckpt_dict = load_checkpoints(args.benchmark, force=args.force, local_checkpoints=args.cluster_local_eval)
        print(f"Found {len(ckpt_dict)} checkpoints for benchmark {args.benchmark}")

        for run_id, (run_name, ckpt_path) in ckpt_dict.items():

            print(f"Run ID: {run_id} - Run Name: {run_name} - Checkpoint Path: {ckpt_path}")
            
            # finetuning is only for MT runs
            if is_mt_run(ckpt_path, run_name):
                cmd = f"scripts/run.py --headless OVERRIDE_CFGNAME experiment_cfgs/ft_mt.yaml EVAL_CHECKPOINT {ckpt_path} EVAL_CLUSTER_ON_LOCAL {args.cluster_local_eval}"
                commands.append(RUN_TEMPLATE.replace("COMMAND", cmd))


        if not args.single_script:
            print(f"Writing {len(commands)} commands to {output_path}")
            with open(output_path, 'w') as f:
                f.write("#!/bin/bash\n\n")
                for cmd in commands:
                    f.write(cmd + "\n")
        else:
            all_commands.extend(commands)
                    
        total_runs += len(commands)

    print(f"Total runs: {total_runs}")
    if args.single_script:
        print(f"Writing all {len(all_commands)} commands to {all_commands_output_path}")
        with open(all_commands_output_path, 'w') as f:
            f.write("#!/bin/bash\n\n")
            for cmd in all_commands:
                f.write(cmd + "\n")