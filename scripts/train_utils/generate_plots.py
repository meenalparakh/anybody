import argparse
from anybody.utils.path_utils import get_figures_dir
from anybody.global_cfgs.experiment_names import reach_tasks, push_tasks, ablations_tasks

def metric_fn(x): return "robo_0_ee" in x   


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", type=str, default="reach", help="Task name from benchmark_cfgs to generate plots for",
        choices=['reach', 'push']        # for ablation, we have just a table.
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
        
        
        
    for task_name, (project_name, run_dict) in tasks_list.items():
        pass