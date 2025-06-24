import argparse


from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Base arguments for loading a benchmark environment."
)
parser.add_argument(
    "--num_envs", type=int, default=2, help="Number of environments to spawn."
)
# add following args: task_type, env_type, targ_type, seed, variation

parser.add_argument(
    "--env_type", type=str, default="stick", help="Type of environment."
)
parser.add_argument("--seed", type=int, default=0, help="Random seed.")
parser.add_argument(
    "--variation", type=str, default="stick_0_v1", help="Variation of the environment."
)


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
