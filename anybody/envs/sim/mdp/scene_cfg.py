import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg

from isaaclab.utils import configclass

from anybody.envs.sim.utils import (
    get_robo_cfg_dict,
    get_obst_cfg_dict,
    get_articulations_cfg_dict,
)
# from anybody.envs import robo_cfg_fns, get_default_robo_cfg
from anybody.morphs.robots import robo_cfg_fns, get_default_robo_cfg
from anybody.cfg import cfg as global_cfg

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    def __init__(self, prob, robo_type, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # self.filter_collision = False
        ground_ht = prob.ground if prob.ground > (global_cfg.THRESHOLD_GROUND_HT) else global_cfg.THRESHOLD_GROUND_HT
        prob.ground = ground_ht

        # ground plane
        ground_cfg = sim_utils.GroundPlaneCfg()
        self.world_ground = AssetBaseCfg(
            prim_path="/World/defaultGroundPlane",
            spawn=ground_cfg,
            init_state=AssetBaseCfg.InitialStateCfg(pos=(10.0, 0.0, 0.0)),
            collision_group=-1,
        )

        # lights
        self.world_dome_light = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Light",
            spawn=sim_utils.DomeLightCfg(
                intensity=3000.0, color=(0.75, 0.75, 0.75)
            ),
        )

        # robot_0 = robo_dict['robot_0']
        # obstacle_0 = obst_dict['obstacle_1']

        if robo_type in robo_cfg_fns:
            cfg_fn = robo_cfg_fns[robo_type]
        else:            
            def cfg_fn(fname, robo_id):
                
                return get_default_robo_cfg(robo_type, fname, robo_id)
            
            # raise NotImplementedError(f"Robot type {robo_type} not implemented")

        robot_configs_dict = get_robo_cfg_dict(
            prob, get_cfg_fn=cfg_fn, ground_ht=ground_ht
        )
        obstacle_configs_dict = get_obst_cfg_dict(prob, ground_ht=ground_ht)
        articulation_configs_dict = get_articulations_cfg_dict(
            prob, ground_ht=ground_ht
        )

        # add robot and obstacles
        for robo_name, robo in robot_configs_dict.items():
            self.__setattr__(robo_name, robo)

        for obst_name, obst in obstacle_configs_dict.items():
            self.__setattr__(obst_name, obst)

        for art_name, art in articulation_configs_dict.items():
            self.__setattr__(art_name, art)
