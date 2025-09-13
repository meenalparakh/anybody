import argparse
from anybody.utils.path_utils import get_figures_dir
from anybody.global_cfgs.experiment_names import reach_tasks, push_tasks, ablations_tasks
from anybody.utils.wandb_utils_v2 import construct_eval_df
from anybody.utils.plot_utils_v2 import modify_method_names, summarize_df, SubplotVisualizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", type=str, default="reach", help="Task name from benchmark_cfgs to generate plots for",
        choices=['reach', 'push']        # for ablation, we have just a table.
    )
    parser.add_argument(
        "--force", action='store_true', help="If set, force re-collection of runs from wandb"
    )
    
    args = parser.parse_args()
    
    if args.task in ['reach', 'arm3']:
        tasks_list = reach_tasks
        def metric_fn(x): return "robo_0_ee" in x
    elif args.task == "push":
        tasks_list = push_tasks
        def metric_fn(x): return ("Success rate" in x) or ("robo_0_ee" in x)
    elif args.task == "ablation":
        tasks_list = ablations_tasks
        def metric_fn(x): return "robo_0_ee" in x   
        
        
    # for task_name, (project_name, run_dict) in tasks_list.items():
        # df, df_grouped = construct_df(project_name, force=args.force)


    benchmark_task = "Arm3"
    project_name, run_dict = reach_tasks[benchmark_task]    
    df, df_grouped = construct_eval_df(project_name, force=args.force)

    renamed_df = modify_method_names(df=df_grouped, run_dict=run_dict)
    df_summary = summarize_df(renamed_df, run_dict=run_dict, normalize=(args.task == 'reach'))

    print("." * 50 + f" {benchmark_task} " + "." * 50)
    print(renamed_df)
    print(df_summary)   
    print("." * 100)        
    
    
    # /////////////////////////////////// plot check, using the same df for all ////////////////////////////////////
    all_dfs = [df_summary.copy() for _ in range(len(reach_tasks))]
    all_names = list(reach_tasks.keys())

    visualizer = SubplotVisualizer(
        dataframes=all_dfs, names=all_names
    )
    
    visualizer.create_subplots_bar2(
        metric='mt', legend=False, group_indices=[0, 4, 6], task="reach", big=False, remove_group=None)