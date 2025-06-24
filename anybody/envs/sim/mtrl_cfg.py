from urdfpy import URDF

from isaaclab.utils import configclass
from isaaclab.envs import ManagerBasedMTRLEnvCfg, TaskConfigs

from anybody.envs.sim.mdp.action_cfg import ActionsCfg
from anybody.envs.sim.mdp.obs_cfg import ObservationsCfg
from anybody.envs.sim.mdp.reward_cfg import (
    ReachRewardCfg,
    PushRewardCfg,
    TerminationsCfg,
    ReachCurriculumCfg,
    PushCurriculumCfg,
)
from anybody.envs.sim.mdp.scene_cfg import MySceneCfg
from anybody.envs.sim.mdp.event_cfg import EventCfg
from anybody.envs.sim.mdp.command_cfg import CommandsCfg

from anybody.envs.tasks import generate_problem_spec
from anybody.cfg import cfg

from anybody.envs.tasks.env_utils import ProblemSpec


def check_env_cfg(prob: ProblemSpec):
    for robo_id, robo in prob.robot_dict.items():
        robo_urdfpy = URDF.load(robo.robot_urdf)
        n_links = (
            len(robo_urdfpy.links) - 1
        )  # -1 for the base link - it is counted in robo base
        assert n_links <= cfg.BENCH.MAX_NUM_LINKS, (
            f"Too many links than MAX_NUM_LINKS, max supported: {cfg.BENCH.MAX_NUM_LINKS}, you have {n_links}"
        )

    assert len(prob.obj_dict) <= (cfg.BENCH.MAX_NUM_OBJECTS + 1), (
        f"Too many objects, max supported: {cfg.BENCH.MAX_NUM_OBJECTS} and 1 obstacle, you have {len(prob.obj_dict)}"
    )


@configclass
class BenchmarkRLCfg(ManagerBasedMTRLEnvCfg):
    def __init__(self):
        # actually there is no need to pass cfg here - it is global

        super().__init__()

        for i in range(len(cfg.MULTIENV.TASKS)):
            robo_task = cfg.MULTIENV.TASKS[i]
            robo_type = cfg.MULTIENV.ROBOTS[i]
            seed = cfg.MULTIENV.SEEDS[i]
            robot_env_variation = cfg.MULTIENV.VARIATIONS[i]

            task_name = f"Task_{i}"

            prob = generate_problem_spec(
                benchmark_task=cfg.BENCHMARK_TASK,
                robo_cat=robo_type,
                robo_task=robo_task,
                variation=robot_env_variation,
                seed=seed,
                save_if_not_exist=True,
            )

            check_env_cfg(prob)
            task_cfg = TaskConfigs(
                episode_length_s=cfg.TRAIN.EPISODE_LENGTH_S,
                scene=MySceneCfg(
                    prob,
                    robo_type,
                ),
                observations=ObservationsCfg(prob, robo_task=robo_task, task_idx=i),
                actions=ActionsCfg(prob),
                events=EventCfg(prob),
                terminations=TerminationsCfg(prob, robo_task),
                commands=CommandsCfg(prob, unique_name=task_name, task=robo_task),
            )
            if robo_task == "reach":
                task_cfg.rewards = ReachRewardCfg(prob)
                task_cfg.curriculum = ReachCurriculumCfg(prob)

            elif robo_task == "push_simple":
                task_cfg.rewards = PushRewardCfg(prob)
                task_cfg.curriculum = PushCurriculumCfg(prob)
            else:
                raise ValueError(f"Unknown task {robo_task} for robot {robo_type}")

            self.__setattr__(task_name, task_cfg)

        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = cfg.TRAIN.SIM_DT
        self.num_multi_task_envs = len(cfg.MULTIENV.TASKS)
        self.task_spacing = cfg.MULTIENV.TASK_SPACING
        self.num_envs_per_task = cfg.TRAIN.NUM_ENVS_PER_TASK
        self.envs_spacing = cfg.MULTIENV.ENV_SPACING
        self.decimation = cfg.TRAIN.DECIMATION
        self.sim.render_interval = cfg.TRAIN.DECIMATION
        self.seed = 0