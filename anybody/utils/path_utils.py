import os, sys
from pathlib import Path

# path to current file:
current_file_path = Path(__file__).resolve()

# path to the root of the project:
project_root = current_file_path.parents[2]

def get_src():
    return project_root

def get_problem_spec_dir():
    return get_src() / "anybody" / "envs" / "problem_specs"

def get_tmp_mesh_storage():
    # return Path("/tmp/")
    return get_src() / ".cache"

def get_robot_morphs_dir():
    return get_src() / "anybody" / "morphs" / "robots"

def get_metrics_log_dir():
    return get_src().parents[0] / "metrics_logs"

def get_benchmark_cfgs_dir():
    return get_src() / "anybody" / "envs" / "benchmark_cfgs"


def get_global_cfgs_dir():
    return get_src() / "anybody" / "global_cfgs"

def get_logs_dir():
    return get_src().parents[0] / "logs"


def get_figures_dir():
    return get_src() / "figures"


def overheat_script_dir():
    return "/workspace/check_overheat"

# check overheating on cluster. Specifically for Princeton Ionic/Neuronic cluster
def import_overheat_module():
    if 'ISAACLAB_PATH' in os.environ and 'workspace' in os.environ['ISAACLAB_PATH']:
        # running on the cluster
        # check if batch job
        # check if overheat script exists
        if os.path.exists(overheat_script_dir()):
            sys.path.append(overheat_script_dir())
            import check_overheat
            return check_overheat
        else:
            
            print("\n" + "#"*50)
            print(f"Overheat script not found at {overheat_script_dir()}")
            print("#"*50 + "\n")
            
            return None
        
        sys.path.append(overheat_script_dir())
        try:
            import check_overheat
            return check_overheat
        except ValueError as e:
            if 'Skipping' in str(e):
                print("\n#"*50)
                print(e)
                print("#"*50 + "\n")
                return None
            else:
                raise e
            
    
    return None
    


def get_wandb_fname():
    path1 = get_src().parents[0] / "wandb" / "wandb_key.txt"
    if path1.exists():
        return path1
    else:
        return Path(os.environ["ISAACLAB_PATH"]) / "wandb_key.txt"