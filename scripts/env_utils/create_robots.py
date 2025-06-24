import os
import argparse

from anybody.morphs.generate_morphs import morph_generator_dict, real_robo_names, create_real_robot_usd

from anybody.utils.utils import set_seed
from anybody.utils.path_utils import get_robot_morphs_dir

set_seed(0)

def create_robots(args):    
    if args.real:
        for robo_name in real_robo_names:
            print(f"################### Converting to usd: {robo_name} ###################")
            create_real_robot_usd(robo_name)
    
    if args.artificial:
        for name, func in morph_generator_dict.items():
            print(f"################### Creating morph: {name} ###################")
            if args.randomized:
                try:
                    func(to_usd=to_usd, randomized=True)
                # exception would be that randomized argument is not supported by the function
                except TypeError:
                    print(f"Function for {name} does not support randomized argument")
                    # for now we just ignore those morphs
                    # func(to_usd=to_usd)
            else:
                func(to_usd=to_usd)
                
    elif args.morphs:        
        
        for name in args.morphs:
            
            print(f"################### Creating morph: {name} ###################")
            
            if name == "panda_variations":
                create_real_robot_usd("panda")   
                
                # uncomment the following to generate new or more variations of panda 
                #               (named panda_0, panda_1, ..., panda_9, and so on.)
                
                # morph_generator_dict[name](n=10)
                         
                for i in range(10):
                    print(f"################### Creating morph: panda_{i} ###################")
                    create_real_robot_usd(f"panda_{i}")

            else:
                if args.randomized:
                    try:
                        morph_generator_dict[name](to_usd=to_usd, randomized=True)
                    # exception would be that randomized argument is not supported by the function
                    except TypeError:
                        print(f"Function {morph_generator_dict[name]} does not support randomized argument")
                        # map_name_to_func[name](to_usd=to_usd)
                else:
                    morph_generator_dict[name](to_usd=to_usd, randomized=False)
            
def convert_to_usd(args):
    from anybody.utils.to_usd import ArgsCli, main

    def robo_cat_to_usd(robo_cat):
        robo_cat_dir = get_robot_morphs_dir() / name
        robo_names = os.listdir(robo_cat_dir)
        for robo_name in robo_names:
            potential_urdf = robo_cat_dir / robo_name / f"{robo_name}.urdf"
            if potential_urdf.exists():
                print(f"Converting {name}/{robo_name} to USD")
                args_cli = ArgsCli(
                    input=str(potential_urdf),
                    output=str(robo_cat_dir / robo_name / f"{robo_name}.usd"),
                    headless=True,
                )
                main(args_cli)                

    if args.real:
        # same as main, as real robots already use existing urdfs
        for robo_name in real_robo_names:
            print(f"################### Converting to usd: {robo_name} ###################")
            create_real_robot_usd(robo_name)

    if args.artificial:

        for name in morph_generator_dict.keys():
            # loop over all possible urdfs in the directory
            robo_cat_to_usd(name)
    elif args.morphs:        
        for name in args.morphs:
            if name == "panda_variations":
                create_real_robot_usd("panda")            
                for i in range(10):
                    print(f"################### Converting to usd: panda_{i} ###################")
                    create_real_robot_usd(f"panda_{i}")
            else:
                robo_cat_to_usd(name)                



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--artificial", default=False, action="store_true", help="Create all morphs"
    )
    
    parser.add_argument(
        "--real", default=False, action="store_true", help="Create real robot morphs"
    )
    
    parser.add_argument(
        "--remove_old", default=False, action="store_true", help="Remove old usds"
    )
    
    parser.add_argument(
        "--no_usd", default=False, action="store_true", help="Do not convert to USD"
    )
    parser.add_argument(
        "--only_usd", default=False, action="store_true", help="Convert the morphs to USD only (existing URDFs will be used)"
    )    
    
    parser.add_argument(
        "--randomized", default=False, action="store_true", help="Create randomized morphs"
    )
    # provide custom list of morphs to create
    parser.add_argument(
        "--morphs", nargs="+", help="List of morphs to create"
    )    

    args = parser.parse_args()
    to_usd = not args.no_usd

    if to_usd:
        from isaacsim import SimulationApp
        # launch omniverse app
        config = {"headless": True}
        simulation_app = SimulationApp(config)
        import traceback
        import carb        

    if args.remove_old:
        # remove the .usd files in the morphs directory
        morphs_dir = get_robot_morphs_dir()
        for usd_file in morphs_dir.glob("**/*.usd"):
            if "real" in str(usd_file):
                continue
            usd_file.unlink()
            print(f"Removed {usd_file}")

    if to_usd:
        try:
            if args.only_usd:
                convert_to_usd(args)
            else:            
                create_robots(args)
        except Exception as err:
            carb.log_error(err)
            carb.log_error(traceback.format_exc())
            raise
        finally:
            # close sim app
            simulation_app.close()
        
    else:
        create_robots(args)
        