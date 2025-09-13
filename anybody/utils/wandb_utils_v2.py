import os
from anybody.utils.path_utils import get_wandb_csv_dir, get_experiment_scripts_dir
from anybody.utils.utils import load_pickle, save_pickle
import wandb
import pandas as pd

# given a project name,
# collect a list of all runs in that project, and their corresponding group, and the config 

def is_metric_column(project: str, col: str):
    if "reach" in project.lower():
        return ("robo_0_ee" in col)
    if "push" in project.lower():
        return ("Success rate" in col)
    if "task" in project.lower():
        return ("Success rate" in col) or ("robo_0_ee" in col)

    return False

def collect_runs(project_name, force=False):
    save_path = get_wandb_csv_dir() / f"{project_name}_runs.pkl"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if save_path.exists() and not force:
        print(f"Loading previously saved runs info from {save_path}")
        return load_pickle(save_path)

    entity = "meenalp_project"
    api = wandb.Api()

    # Fetch all runs in the project
    runs = api.runs(f"{entity}/{project_name}")
    runs_info = {}

    for run in runs:
        # Extract relevant information from each run
        run_id = run.id
        run_group = run.group

        # Fetch the history (time-series data) for the run
        history = run.history()
        metrics = [col for col in history.columns if is_metric_column(project_name, col)]
        metrics.sort()
        
        if len(metrics) == 0:
            print(f"Run ID: {run_id}/{run.name}/{run_group} - No relevant metrics found. Skipping.")
            continue

        run_directory = run.config["AGENT"]["EXPERIMENT"]["DIRECTORY"]


        if run_group not in runs_info:
            runs_info[run_group] = {}

        runs_info[run_group][run_id] = (run.name, run_directory)
        
    # dump into a pickle file
    save_pickle(runs_info, save_path)
    print(f"Saved runs info to {save_path}")
    return runs_info

def is_eval_run(run_group: str):
    if "eval" in run_group.lower():
        return True
    if "random" in run_group.lower():
        return True
    if "diff-ik" in run_group.lower():
        return True
    return False

def get_renamed_cols(cols):
    renamed_cols = []
    
    reach_metric = " / EpisodeInfo / Episode_Reward/robo_0_ee"
    push_metric = "_push_simple / Episode / Success rate"
    
    for col in cols:
        if reach_metric in col:
            renamed_cols.append(col.replace(reach_metric, ""))
        elif push_metric in col:
            renamed_cols.append(col.replace(push_metric, ""))
        else:
            renamed_cols.append(col)
    return renamed_cols


def construct_train_df(project_name):
    entity = "meenalp_project"
    api = wandb.Api()

    # Fetch all runs in the project
    runs = api.runs(f"{entity}/{project_name}")

    df = []

    for run in runs:
        # Extract relevant information from each run
        run_id = run.id
        run_group = run.group
        
        if is_eval_run(run_group):    
            continue
        
        # for the run, record the last timestep, and the final value of each metric logged
        history = run.history()
        metrics = [col for col in history.columns if is_metric_column(project_name, col)]
        metrics.sort()
        
        if len(metrics) == 0:
            print(f"Run ID: {run_id}/{run.name}/{run_group} - No relevant metrics found. Skipping.")
            continue
        last_timestep = history['global_step'].max()
        # obtain value of each metric at the last timestep
        last_values = history[history['global_step'] == last_timestep]
        
        row = {
            "run_name": run.name, "run_id": run_id,
            "group": run_group,
            "last_timestep": last_timestep,
        }
        for metric in metrics:
            row[metric] = last_values[metric].values[0]
            
        df.append(row)
        
    # Construct main DataFrame
    df = pd.DataFrame(df)
    renamed_cols = get_renamed_cols(df.columns)
    df.columns = renamed_cols
    
    save_path = get_wandb_csv_dir() / f"{project_name}_train_df.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Saved training DataFrame to {save_path}")
    
    return df


def construct_eval_df(project_name, force=False):
    save_path = get_wandb_csv_dir() / f"{project_name}_df.csv"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    grouped_save_path = get_wandb_csv_dir() / f"{project_name}_grouped_df.csv"

    if save_path.exists() and not force:
        print(f"Loading previously saved runs info from {save_path}, {grouped_save_path}")
        return pd.read_csv(save_path, index_col=0), pd.read_csv(grouped_save_path, index_col=0, header=[0,1])

    entity = "meenalp_project"
    api = wandb.Api()

    # Fetch all runs in the project
    runs = api.runs(f"{entity}/{project_name}")
    runs_info = {}

    df = []

    for run in runs:
        # Extract relevant information from each run
        run_id = run.id
        run_group = run.group
        
        if not is_eval_run(run_group):
            continue            

        # Fetch the history (time-series data) for the run
        history = run.history()
        metrics = [col for col in history.columns if is_metric_column(project_name, col)]
        metrics.sort()
        
        if len(metrics) == 0:
            print(f"Run ID: {run_id}/{run.name}/{run_group} - No relevant metrics found. Skipping.")
            continue

        # each run is a row, with columns: group, and the average value over the time steps for each metric that
        # is available for that run.
        row = {"run_name": run.name, "run_id": run_id,
            "group": run_group}
        for metric in metrics:
            row[metric] = history[metric][1:].mean()   # skip the first value

        df.append(row)
        
        
    # Construct main DataFrame
    df = pd.DataFrame(df)
    renamed_cols = get_renamed_cols(df.columns)

    df.columns = renamed_cols
    df.to_csv(save_path, index=True)
    print(f"Saved run-level DataFrame to {save_path}")

    
    # Grouped summary: mean and std for each metric per group
    metric_cols = [col for col in df.columns if col not in ["group", "run_name", "run_id"]]
    
    # remove the columns that are non-metric, keeping only the group and metric columns
    _df = df[["group"] + metric_cols]

    df_grouped = _df.groupby("group")[metric_cols].agg(['mean', 'std', 'count'])
    df_grouped.to_csv(grouped_save_path, index=True)
    print(f"Saved grouped summary DataFrame to {grouped_save_path}")

    return df, df_grouped
