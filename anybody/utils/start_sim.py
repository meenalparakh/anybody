import argparse


from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# parser.add_argument(
#     "--cfg", dest="cfg_file", help="Config file", required=True, type=str
# )

# all config options in bench/envs/cfg.py
parser.add_argument(
    "opts",
    help="See morphology/core/config.py for all options",
    default=None,
    nargs=argparse.REMAINDER,
)

# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
