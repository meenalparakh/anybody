import argparse
from anybody.global_cfgs.experiment_names import (
    REACH_TASKS_DICT,
    REACH_FINETUNE_DICT,
    PUSH_TASKS_DICT,
    PUSH_FINETUNE_DICT,
    ABLATIONS_TASKS_DICT,
)
from anybody.utils.wandb_utils_v2 import construct_eval_df
from anybody.utils.plot_utils_v2 import summarize_df, SubplotVisualizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="reach",
        help="Task name from benchmark_cfgs to generate plots for",
        choices=["reach", "push", "ablation", "reach-ft", "push-ft"],  # for ablation, we have just a table.
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="If set, force re-collection of runs from wandb",
    )
    parser.add_argument(
        "--legend",
        action="store_true",
        help="If set, show legend in plots",
    )

    args = parser.parse_args()

    max_timestep = 1000000
    min_timestep = 400000
    if args.task in ["reach"]:
        tasks_dict = REACH_TASKS_DICT
        group_indices = [0, 3, 4]
        method_names = ["Mlp", "Tf"]
    elif args.task == 'reach-ft':
        tasks_dict = REACH_FINETUNE_DICT
        group_indices = [0, 1, 2]
        method_names = ['Mlp', 'Mlp-ft10', 'Mlp-ft30', 'Mlp-ft50', 
                        'Tf', 'Tf-ft10', 'Tf-ft30', 'Tf-ft50']
    elif args.task == "push":
        tasks_dict = PUSH_TASKS_DICT
        group_indices = [0, 4, 6]
        # method_names = ["Se-Mlp", "Mlp", "Tf"]
        method_names = ['Mlp', 'Tf']
    elif args.task == 'push-ft':
        tasks_dict = PUSH_FINETUNE_DICT
        group_indices = [0, 1, 3]
        method_names = ['Mlp', 'Mlp-ft50', 
                        'Tf', 'Tf-ft50']

    elif args.task == "ablation":
        tasks_dict = ABLATIONS_TASKS_DICT
        method_names = ["Tf", "Tf-noSL", "Tf-cont", "Tf-noCE", "Mlp", "Mlp+SlCe"]
        group_indices = [0]
        max_timestep = min_timestep = 300000  # only consider runs at 300k for ablation
    else:
        raise ValueError(f"Unknown task {args.task}")

    all_dfs = []
    all_names = []

    for benchmark_task, (project_name, run_dict) in tasks_dict.items():
        df, df_grouped = construct_eval_df(
            project_name,
            force=args.force,
            min_timestep=min_timestep,
            max_timestep=max_timestep,
            ablation=(args.task == "ablation"),
        )
        df_summary = summarize_df(df_grouped, run_dict=run_dict, normalize=True, only_zs=("ft" in args.task))

        print("." * 50 + f" {benchmark_task} " + "." * 50)
        print(df_summary)
        print("." * 100)

        all_dfs.append(df_summary)
        all_names.append(benchmark_task)

    # /////////////////////////////////// plot check, using the same df for all ////////////////////////////////////
    visualizer = SubplotVisualizer(dataframes=all_dfs, names=all_names)

    if args.task == "ablation":
        visualizer.create_per_task_bargraphs(
            df=all_dfs[0], task_name=all_names[0], methods=method_names
        )
    else:
        if "ft" not in args.task:
            visualizer.create_subplots_bar(
                metric="mt",
                legend=args.legend,
                methods=method_names,
                group_indices=group_indices,
                task=args.task,
                big=False,
                remove_group=None,
            )
        visualizer.create_subplots_bar(
            metric="zs",
            legend=args.legend,
            methods=method_names,
            group_indices=group_indices,
            task=args.task,
            big=False,
            remove_group=None,
        )
