import pandas as pd
import numpy as np
import os
from anybody.utils.path_utils import get_metrics_log_dir, get_wandb_fname
from anybody.utils.utils import save_pickle, load_pickle
import wandb

with open(get_wandb_fname(), "r") as f:
    key = f.read().strip()
wandb.login(key=key)


def read_from_wandb(project_name, select_run_fn=lambda x: "eval" in x, select_metric_fn=lambda x: "robo_0_ee" in x):
    # Authenticate and initialize wandb API
    api = wandb.Api()

    # Fetch all runs from the project
    runs = api.runs(project_name)

    # Initialize a dictionary to store results
    results = {}

    for run in runs:
        # Filter runs with "eval" in their name
        if select_run_fn(run.name):
            # Initialize a dictionary to store metrics for this run
            # run_metrics = {}
            if run.name not in results:
                results[run.name] = {}

            # Fetch the history (time-series data) for the run
            history = run.history()

            # Filter metrics containing "robo_0_ee"
            metrics = [col for col in history.columns if select_metric_fn(col)]
            
            if "Reach_Real_Pose_Ur5_stick_v1 / Episode / Success rate" in metrics:
                # remove it
                metrics.remove("Reach_Real_Pose_Ur5_stick_v1 / Episode / Success rate")
            
            metrics.sort()
            
            if len(metrics) == 0:
                print(f"No metrics found for run {run.name}.")
                continue

            for metric in metrics:
                # Average the values of the metric across all time steps
                avg_value = history[metric].mean()
                if metric in results[run.name]:
                    results[run.name][metric].append(avg_value)
                else:
                    results[run.name][metric] = [avg_value]


    # for each run_name, compute the average of the values across all its metrics, and find the best idx
    for method_name in results:
        metrics = results[method_name].keys()
        n_metric_vals = [len(results[method_name][metric]) for metric in metrics]
        # check if all the metrics have the same number of values
        
        if ("Ind" in method_name) or ("random" in method_name.lower()):
            # choose the max
            for metric in metrics:
                results[method_name][metric] = np.max(results[method_name][metric])
                            
        
        elif len(set(n_metric_vals)) != 1:
            print(f"{method_name} - Different number of values for metrics: {n_metric_vals}.")
            print(f"The metric list is: {metrics}.")
            assert False
        
        elif n_metric_vals[0] == 1:
            for metric in metrics:
                results[method_name][metric] = results[method_name][metric][0]
        else:            
            results_per_method = np.zeros((len(metrics), n_metric_vals[0]))
            for i, metric in enumerate(metrics):
                results_per_method[i] = results[method_name][metric]
            
            # compute the average of the values across all its metrics
            avg_per_run = np.mean(results_per_method, axis=0)
            best_idx = np.argmax(avg_per_run)
            # assign the best idx to all the metrics
            for i, metric in enumerate(metrics):
                results[method_name][metric] = results_per_method[i][best_idx]
                
    # # average over overlapping values
    # for run_name in results:
    #     for metric, values in results[run_name].items():
    #         if len(values) > 1:
    #             print(f"Multiple values found for {run_name} - {metric}: {values}. Taking the max.")
                
    #             # for each run_name, first compute the average of the values across all its metrics, and then choose the idx that has the max value, assign that to all the metrics
                
    #             results[run_name][metric] = np.max(values)
    #         else:
    #             results[run_name][metric] = values[0]

    # Convert the results dictionary to a DataFrame
    df = pd.DataFrame.from_dict(results, orient="index")

    # rename the columns to be concise
    # a typical column has name:  Reach_Simple_bot_Pose_R63_v1 / EpisodeInfo / Episode Reward/robo_0_ee
    
    if "reach" in project_name:
        new_column_names = {name: name.split("Pose_")[-1].split(" /")[0][:-3] for name in df.columns}    
    else:
        new_column_names = {}
        for name in df.columns:
            if "robo_0_ee" in name:
                new_column_names[name] = name.split("Pose_")[-1].split(" /")[0][:-3]
                if "task" in project_name:
                    new_column_names[name] = new_column_names[name] + "_reach"
            elif "Success rate" in name:    
                new_column_names[name] = name.split("Joint_")[-1].split(" /")[0][:-3]
                if "task" in project_name:
                    new_column_names[name] = new_column_names[name] + "_push"
            else:
                raise ValueError(f"Unknown column name: {name}.")   
        
    # if project_name == "inter_task_ur5_15":
    #     import pdb; pdb.set_trace()
        
        # new_column_names = {name: name.split("Joint_")[-1].split(" /")[0][:-3] for name in df.columns}
    df = df.rename(columns=new_column_names)
    
    # for each row (run), create an average column, that is the average of all the columns and ignores the NaN values
    df["Average"] = df.mean(axis=1, skipna=True)

    # Return the DataFrame
    return df

def get_best_wandb_run(
    entity,
    project_name,
    metrics,
    t_min=5000,
    t_max=int(1e7),
    save=True,
    skip_values=False,
    force_rerun=False,
    use_last_available=False,
    look_for=None,
):
    """
    metrics: list of regex strings to match the metrics
    t_min: int, minimum global step to consider
    t_max: int, maximum global step to consider
        the t_min and t_max refers to agent_{t}.pt files saved at these global steps
        as currently the step and global_step are confusing - what do they mean

    Returns:
        run_dicts: dict of each metric.
        for each metric, the dict value is another dict with keys as the run names and values list
               the list is of different seeds that share the same name.
    """

    # Initialize wandb API
    api = wandb.Api(timeout=200)

    # Replace with your project and run details
    project = project_name

    # Fetch all runs in the project
    runs = api.runs(f"{entity}/{project}")

    save_path = get_metrics_log_dir() / f"{project_name}.pkl"

    if save and save_path.exists() and (not force_rerun):
        run_dicts, run_dir_dicts = load_pickle(save_path)

    else:
        run_dicts = {}
        run_dir_dicts = {}
        get_metrics_log_dir().mkdir(parents=True, exist_ok=True)

    # Loop over each run and find the global_step with the highest reward
    for run in runs:
        run_id = run.id
        
        if look_for is not None:
            if run.name not in look_for:
                continue

        if run_id in run_dir_dicts:
            print(f"Run ID: {run_id}/{run.name} - Already found. Skipping.")
            continue

        try:
            # for each metric, get the value and store it in the run_dicts
            run_directory = run.config["AGENT"]["EXPERIMENT"]["BASE_DIRECTORY"]
            run_dir_dicts[run_id] = (run.name, run_directory)
        except KeyError:
            print(
                f"Run ID: {run_id}/{run.name} - Key 'AGENT.EXPERIMENT.BASE_DIRECTORY' not found. Skipping."
            )
            continue

        if not skip_values:
            # Fetch the history of the run
            history = run.history()
            if isinstance(history, list):
                # organize with keys
                hist_dict = {}
                for h in history:
                    for k, v in h.items():
                        if k not in hist_dict:
                            hist_dict[k] = []
                        hist_dict[k].append(v)
                history = hist_dict
                # convert to dataframe
                history = pd.DataFrame(history)

            # Extract all keys from the history
            all_keys = history.keys()
            selected_keys = [k for k in all_keys if any([m in k for m in metrics])]
            history = run.history(keys=["global_step", *selected_keys])

            if "global_step" not in history:
                print(
                    f"Run ID: {run_id}/{run.name} - Key 'global_step' not found. Skipping."
                )
                continue

            # select rows where the global_step is between t_min and t_max
            _history = history[
                (history["global_step"] >= t_min) & (history["global_step"] <= t_max)
            ]

            if len(_history) == 0:
                if not use_last_available:
                    print(f"Run ID: {run_id}/{run.name} - No data found. Skipping.")
                    continue
                else:
                    _history = history

            history = _history

            # get the last timestep
            last_timestep = history["global_step"].max()
            print(f"Run ID: {run_id}/{run.name} - Last timestep: {last_timestep}.")

            # get the metric values for the last timestep
            entry_idx = history["global_step"].idxmax()
            entry = history.loc[entry_idx]

            # for metric in metrics:
            for key in selected_keys:
                # check if the metric is present in the entry
                if key not in entry:
                    print(
                        f"Run ID: {run_id}/{run.name} - Key {key} not found. Skipping."
                    )
                    continue

                value = entry[key]
                # also append the run id to the metric value
                if key in run_dicts:
                    if run.name in run_dicts[key]:
                        run_dicts[key][run.name].append((value, run_id))
                    else:
                        run_dicts[key][run.name] = [(value, run_id)]
                else:
                    run_dicts[key] = {run.name: [(value, run_id)]}

        if save:
            save_pickle((run_dicts, run_dir_dicts), save_path)

    return run_dicts, run_dir_dicts, save_path


def get_last_checkpoints(
    project_name,
    true_logs_path,
    t_min=5000,
    t_max=int(1e7),
    save=True,
    force_read=False,
    filter_fn=None,
):
    if force_read:
        save_path = get_metrics_log_dir() / f"{project_name}.pkl"
        if not save_path.exists():
            return {}               # empty dict as there are no saved runs
        run_dicts, run_dir_dicts = load_pickle(save_path)
    else:
        run_dicts, run_dir_dicts, save_path = get_best_wandb_run(
            project_name, ["Total reward (mean)"], t_min, t_max, save
        )

    last_checkpoints_dict = {}
    for run_id, (run_name, run_dir) in run_dir_dicts.items():
        
        # print(f"Considering run: {run_id}/{run_name}. before filtering.")
        
        if filter_fn is not None and filter_fn(run_name):
            continue
        
        # print(f"Considering run: {run_id}/{run_name}. after filtering.")

        # if run_name == "Ind_test":
        #     _run_name = "Ind"
        # else:
        #     _run_name = run_name
        
        _run_name = run_name
        checkpoints_dir = (
            run_dir.replace("/workspace/logs", "LOGS_PATH")
            + f"/{_run_name}/checkpoints/"
        )
        LOGS_PATH = true_logs_path
        true_ckpts_dir = checkpoints_dir.replace("LOGS_PATH", LOGS_PATH)
        if true_ckpts_dir.startswith("/home/" + os.environ["USER"]):
            continue           # this is a local run for code testing
        
        if not os.path.exists(true_ckpts_dir):
            print(
                f"Run ID: {run_id}/{run_name} - Checkpoints directory does not exist: {checkpoints_dir}. Skipping."
            )
            continue
        
        all_agent_fnames = os.listdir(true_ckpts_dir)
        agent_timesteps = [
            int(fname.split("_")[1].split(".")[0]) for fname in all_agent_fnames
        ]

        if len(agent_timesteps) == 0:
            print(f"Run ID: {run_id}/{run_name} - No checkpoints found. Skipping.")
            continue

        agent_timesteps.sort()

        # filter the agent timesteps between t_min and t_max
        _agent_timesteps = [t for t in agent_timesteps if t >= t_min and t <= t_max]

        if len(_agent_timesteps) == 0:
            continue
            print(
                f"Run ID: {run_id}/{run_name} - No checkpoints found between t_min and t_max. Choosing the last checkpoint available: {agent_timesteps[-1]}."
            )
            _agent_timesteps = [agent_timesteps[-1]]

        agent_timesteps = _agent_timesteps

        # get the last checkpoint
        last_ckpt_step = agent_timesteps[-1]
        last_checkpoint = f"{true_ckpts_dir}/agent_{last_ckpt_step}.pt"
        print(
            f"Run ID: {run_id}/{run_name} - Run directory: {run_dir}. Last checkpoint step: {last_ckpt_step}."
        )

        last_checkpoints_dict[run_id] = (run_name, last_checkpoint)

    return last_checkpoints_dict


def regroup_wandb_runs(entity, project_name):

    # Initialize the API
    api = wandb.Api()

    # Fetch all runs in the project
    runs = api.runs(f"{entity}/{project_name}")

    # Dictionary to map eval runs to their base runs
    group_mapping = {}

    for run in runs:
        name = run.name  # Get the run name
        if name.endswith("_eval"):
            base_name = name.replace("_eval", "")
            group_mapping[name] = base_name  # Store mapping
        else:
            group_mapping[name] = name  # Keep original name

    # Apply the grouping
    for run in runs:
        new_group = group_mapping.get(run.name, run.name)

        if run.group != new_group:
            print(f"Updating run {run.name}: setting group to {new_group}")
            run.group = new_group
            run.update()
            # run.update({"group": new_group})  # Update the group name

    print("Grouping update completed!")


def wandb_rename_runs(entity, project_name):

    # Initialize the API
    api = wandb.Api()

    # Fetch all runs in the project
    runs = api.runs(f"{entity}/{project_name}")

    for run in runs:
        original_name = run.name
        
        if "Ind" in original_name:
            # check the config file for the run to see if OBSERVATION.HIGH_DIM is True
            # if it is, replace Ind with Ind_hd
            if "Ind_hd" in original_name:
                continue        # already renamed

            if run.config["OBSERVATION"]["HIGH_DIM"]:
                print("Renaming run:", original_name)
                new_name = original_name.replace("Ind", "Ind_hd")
                run.name = new_name
                run.update()
            
                
        
