import json
from anybody.utils.path_utils import get_override_cfgs_dir, get_robot_morphs_dir
import yaml
import random

def get_robo_names(robo_cat, name_prefix=""):
    robo_cat_dir = get_robot_morphs_dir() / robo_cat
    robo_cat_dirs = [d for d in robo_cat_dir.iterdir() if d.is_dir()]
    robo_cat_dirs = [d.name for d in robo_cat_dirs]

    valid_robo_names = []        
    for robo_name in robo_cat_dirs:
        if (robo_cat != "chain") and ("_" in robo_name):
            # typically "_" was in all prior generated robo names. the new names do not have these
            continue
        
        if len(name_prefix) > 0 and not robo_name.startswith(name_prefix):
            continue

        potential_robo_urdf = (
            get_robot_morphs_dir() / robo_cat / robo_name / f"{robo_name}.urdf"
        )
        if potential_robo_urdf.exists():
            valid_robo_names.append(robo_name)
    
    random.shuffle(valid_robo_names)
            
    return valid_robo_names


class BenchmarkTaskCfgs:
    def __init__(self, robo_name_fn=None):
        self.cfg_dir = get_override_cfgs_dir() / "benchmark_tasks"
        self.cfg_dir.mkdir(exist_ok=True, parents=True)
        self.n_train = 10
        self.n_test = 1
        self.robo_name_fn = robo_name_fn
                
        self.cfg_generators = {
            "intra_simple_bot_reach": (self.intra_category_cfg, ("reach",)),
            "intra_simple_bot_push_simple": (self.intra_category_cfg, ("push_simple",)),

            "intra_panda_reach": (self.intra_category_cfg_2, ("reach",)),
            "intra_panda_push_simple": (self.intra_category_cfg_2, ("push_simple",)),
            
            "inter_arms_reach": (self.arms_category_cfg, ("reach",)),
            "inter_arms_push_simple": (self.arms_category_cfg, ("push_simple",)),
            
            "inter_ee_arm_reach": (self.ee_arm_category_cfg, ("reach",)),
            "inter_ee_arm_push_simple": (self.ee_arm_category_cfg, ("push_simple",)),
            
            "inter_prims_reach": (self.prims_category_cfg, ("reach",)),
            "inter_prims_push_simple": (self.prims_category_cfg, ("push_simple",)),
            
            "inter_task_ur5": (self.inter_task_category_cfg, ("ur5",)),
            
            "intra_simple_bot_reach_v2": (self.intra_category_cfg, ("reach", "v2")),
            "intra_simple_bot_push_simple_v2": (self.intra_category_cfg, ("push_simple", "v2")),            
            
            # uses different set of robots
            "inter_arms_reach_v2": (self.arms_category_cfg_2, ("reach",)),
            "inter_arms_push_simple_v2": (self.arms_category_cfg_2, ("push_simple",)),
            
        }
        
        self.cfg_paths = {
            k: self.cfg_dir / f"{k}.yaml" for k in self.cfg_generators.keys()
        }

        # dump a json with the list of cfgs
        with open(self.cfg_dir / "cfgs.json", "w") as f:
            json.dump(list(self.cfg_generators.keys()), f)
                
    @staticmethod
    def base_cfg():
        with open(get_override_cfgs_dir() / "rl_general.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        return cfg
    
    
    def generate_new_cfgs(self):
        for cfg_name, cfg_path in self.cfg_paths.items():
            if cfg_path.exists():
                # print(f"Skipping {cfg_name} as it already exists.")
                continue
            cfg_gen, cfg_args = self.cfg_generators[cfg_name]
            cfg_gen(*cfg_args)
            print(f"Generated cfg for {cfg_name}")
    
    def generate_all_cfgs(self):
        # check if the cfgs already exist
        for cfg_name, cfg_path in self.cfg_paths.items():
            if cfg_path.exists():
                input(f"Cfg {cfg_name} already exists. Continuing will overwrite the cfg. Press Enter to continue.")
        
        for cfg_name, (cfg_gen_fn, args) in self.cfg_generators.items():
            cfg_gen_fn(*args)
            print(f"Generated cfg for {cfg_name}")
    
    
    def generate_cfg(self, train_robo_cats, train_robo_names, test_robo_cats, test_robo_names, task, cfg_name, version="v1"):
        
        base_cfg = BenchmarkTaskCfgs.base_cfg()
        
        base_cfg['MULTIENV']['ROBOTS'] = train_robo_cats
        base_cfg['MULTIENV']['VARIATIONS'] = [f"{robo_name}_{version}" for robo_name in train_robo_names]
        base_cfg['MULTIENV']['SEEDS'] = [0] * len(train_robo_names)
        base_cfg['MULTIENV']['TASKS'] = task
        base_cfg['MULTIENV']['TARGETS'] = "pose" if task == "reach" else "joint"
        
        base_cfg['TEST_ENVS']['ROBOTS'] = test_robo_cats
        base_cfg['TEST_ENVS']['VARIATIONS'] = [f"{robo_name}_{version}" for robo_name in test_robo_names]
        base_cfg['TEST_ENVS']['SEEDS'] = [0] * len(test_robo_names)
        base_cfg['TEST_ENVS']['TASKS'] = task
        base_cfg['TEST_ENVS']['TARGETS'] = "pose" if task == "reach" else "joint"
        
        with open(self.cfg_paths[cfg_name], "w") as f:
            yaml.dump(base_cfg, f)
            
    def intra_category_cfg(self, task, version="v1"):
        # simple bot
        cat = "simple_bot"
        all_robo_names = get_robo_names(cat)
        
        # sample train + test robo names
        train_robo_names = all_robo_names[:self.n_train]
        test_robo_names = all_robo_names[self.n_train:self.n_train+self.n_test]
        
        cfg_name = f"intra_{cat}_{task}"
        if version == "v2":
            cfg_name += "_v2"
        self.generate_cfg([cat] * self.n_train, train_robo_names, [cat] * self.n_test, test_robo_names, task, cfg_name, version)
            

    def intra_category_cfg_2(self, task, version="v1"):
        # simple bot
        cat = "panda_variations"
        all_robo_names = [f"panda_{i}" for i in range(10)]
        
        cfg_name = f"intra_panda_{task}"
        self.generate_cfg([cat] * self.n_train, all_robo_names, ["real"], ["panda"], task, cfg_name, version)
            
            
    def arms_category_cfg(self, task):
        
        unreal_cats = ["arm_ur5", "simple_bot"]
        # unreal_cats = ["arm_ed", "simple_bot"]
        real_robo_names = ["ur5_stick", "kinova_gen3", "jaco2", "xarm7", "lwr", "yumi"]
        real_cats = ["real"] * len(real_robo_names)
        
        test_cat = ["real"]
        test_robo_names = ["widowx"]
        
        n_unreal = 2
        unreal_robo_names = [get_robo_names(cat)[:n_unreal] for cat in unreal_cats]
        unreal_cats = [[cat] * n_unreal for cat in unreal_cats]
        
        all_robo_names = real_robo_names 
        for un_rnames in unreal_robo_names:
            all_robo_names.extend(un_rnames)
            
        all_cats = real_cats
        for un_cats in unreal_cats:
            all_cats.extend(un_cats)
            
        self.generate_cfg(all_cats, all_robo_names, test_cat, test_robo_names, task, f"inter_arms_{task}")
        


    def arms_category_cfg_2(self, task):
        
        unreal_cats = ["arm_ur5"]
        # unreal_cats = ["arm_ed", "simple_bot"]
        real_robo_names = ["ur5_stick", "kinova_gen3", "jaco2", "xarm7", "lwr", "yumi", "ur5_ez", "panda"]
        real_cats = ["real"] * len(real_robo_names)
        
        test_cat = ["real"]
        test_robo_names = ["widowx"]
        
        n_unreal = 2
        unreal_robo_names = [get_robo_names(cat)[:n_unreal] for cat in unreal_cats]
        unreal_cats = [[cat] * n_unreal for cat in unreal_cats]
        
        all_robo_names = real_robo_names 
        for un_rnames in unreal_robo_names:
            all_robo_names.extend(un_rnames)
            
        all_cats = real_cats
        for un_cats in unreal_cats:
            all_cats.extend(un_cats)
            
        self.generate_cfg(all_cats, all_robo_names, test_cat, test_robo_names, task, f"inter_arms_{task}_v2")
        

    def ee_arm_category_cfg(self, task):
        # cats = ["stick", "ur5_planar", "prims", "ur5_ez"]
        # stick_robo_names = get_robo_names("stick")

        planar_prim_robo_names = get_robo_names("prims", "pl")      # will output planar prims
        cylindrical_prim_robo_names = get_robo_names("prims", "cy") # will output cylindrical prims
        
        # all_robo_names = stick_robo_names[:2] + \
        all_robo_names = \
            planar_prim_robo_names[:3] + \
            cylindrical_prim_robo_names[:4] + \
            ["ur5_planar", "ur5_ez", "ur5_sawyer"]
            
        # all_cats = ["stick"] * 2 + \
        all_cats = \
            ["prims"] * 7 + \
            ["real"] * 3
            
        test_cat = ["real"]
        test_robo_names = ["ur5_stick"]

        self.generate_cfg(all_cats, all_robo_names, test_cat, test_robo_names, task, f"inter_ee_arm_{task}")

                    
    def prims_category_cfg(self, task):
        cats = ["stick", "nlink", "prims", "chain"]
        stick_robo_names = get_robo_names("stick")
        nlink_robo_names = get_robo_names("nlink")
        pl_prim_robo_names = get_robo_names("prims", "pl")      # will output planar prims
        cy_prim_robo_names = get_robo_names("prims", "cy") # will output cylindrical prims
        cu_prim_robo_names = get_robo_names("prims", "cu") # will output cuboid prims
        sp_prim_robo_names = get_robo_names("prims", "sp") # will output spherical prims
        
        all_robo_names =  stick_robo_names[:2] + \
            nlink_robo_names[:2] + \
            pl_prim_robo_names[:1] + \
            cy_prim_robo_names[:1] + \
            cu_prim_robo_names[:2] + \
            sp_prim_robo_names[:2] 
        all_cats = ["stick"] * 2 + ["nlink"] * 2 + ["prims"] * 6
        
        test_cat = ["chain", "chain"]
        test_robo_names = get_robo_names("chain", "chain_2b")[:1] + get_robo_names("chain", "chain_3b")[:1]
        
        self.generate_cfg(all_cats, all_robo_names, test_cat, test_robo_names, task, f"inter_prims_{task}")
        
        
    def inter_task_category_cfg(self, cat):
        
        def get_v1_info():
            cat = ["prims", "ur5_stick"]
            
            planar_robo_names = get_robo_names("prims", "pl")[:3]
            
            all_robo_names = planar_robo_names + ["ur5_stick"]
            all_robo_cats = ["prims"] * 3 + ["real"]
            train_tasks = ["push_simple"] * 3 + ["reach"]
            
            test_cat = ["real"]
            test_robo_names = ["ur5_planar"]            
            test_tasks = ["push_simple"]
            
            return all_robo_cats, all_robo_names, test_cat, test_robo_names, train_tasks, test_tasks

        def get_v2_info():
            pass
                    

        def _generate_cfg(train_cats, train_names, train_tasks, test_cats, test_names, test_tasks, cfg_name):
            
            base_cfg = BenchmarkTaskCfgs.base_cfg()
            
            base_cfg['MULTIENV']['ROBOTS'] = train_cats
            base_cfg['MULTIENV']['VARIATIONS'] = [f"{robo_name}_v1" for robo_name in train_names]
            base_cfg['MULTIENV']['SEEDS'] = [0] * len(train_names)
            base_cfg['MULTIENV']['TASKS'] = train_tasks
            base_cfg['MULTIENV']['TARGETS'] = ['pose' if task == "reach" else "joint" for task in train_tasks]
            
            base_cfg['TEST_ENVS']['ROBOTS'] = test_cats
            base_cfg['TEST_ENVS']['VARIATIONS'] = [f"{robo_name}_v1" for robo_name in test_names]
            base_cfg['TEST_ENVS']['SEEDS'] = [0] * len(test_names)
            base_cfg['TEST_ENVS']['TASKS'] = test_tasks
            base_cfg['TEST_ENVS']['TARGETS'] = ['pose' if task == "reach" else "joint" for task in test_tasks]
                    
            with open(self.cfg_paths[cfg_name], "w") as f:
                yaml.dump(base_cfg, f)
                
        if cat == "ur5":                
            all_robo_cats, all_robo_names, test_cat, test_robo_names, train_tasks, test_tasks = get_v1_info()
            _generate_cfg(all_robo_cats, all_robo_names, train_tasks, test_cat, test_robo_names, test_tasks, "inter_task_ur5")
        
        
        
if __name__ == "__main__":
    cfgs_generator = BenchmarkTaskCfgs()
    # cfgs_generator.generate_all_cfgs()
    cfgs_generator.generate_new_cfgs()