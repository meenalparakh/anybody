from .arm_ed import save_urdfs as create_arm_ed_urdfs
from .arm_ur5 import save_urdfs as create_arm_ur5_urdfs
from .chain import save_urdfs as create_chain_urdfs
from .cubes import save_urdfs as create_cubes_urdfs
from .multi_finger_arm import save_urdfs as create_mf_arm_urdfs
from .multi_finger_v2 import save_urdfs as create_mf_v2_urdfs
from .multi_finger_v1 import save_urdfs as create_mf_v1_urdfs
from .nlink import save_urdfs as create_nlink_urdfs
from .planar_arm import save_urdfs as create_planar_arm_urdfs
from .prims import save_urdfs as create_prims_urdfs
from .simple_bot import save_urdfs as create_simple_bot_urdfs
from .stick import save_urdfs as create_stick_urdfs
from .tongs_v1 import save_urdfs as create_tongs1_urdfs
from .tongs_v2 import save_urdfs as create_tongs2_urdfs
from .cartpole import save_urdfs as create_cartpole_urdfs
from .panda_variations import create_panda_variations_urdfs
from .real import create_real_robot_usd     # noqa: F401

morph_generator_dict = {
    "arm_ed": create_arm_ed_urdfs,
    "arm_ur5": create_arm_ur5_urdfs,
    "chain": create_chain_urdfs,
    "cubes": create_cubes_urdfs,
    "multi_finger_arm": create_mf_arm_urdfs,
    "multi_finger_v1": create_mf_v1_urdfs,
    "multi_finger_v2": create_mf_v2_urdfs,
    "nlink": create_nlink_urdfs,
    "planar_arm": create_planar_arm_urdfs,
    "prims": create_prims_urdfs,
    "simple_bot": create_simple_bot_urdfs,
    "stick": create_stick_urdfs,
    "tongs_v1": create_tongs1_urdfs,
    "tongs_v2": create_tongs2_urdfs,
    "cartpole": create_cartpole_urdfs,
    "panda_variations": create_panda_variations_urdfs,
}

real_robo_names = [
    "jaco2",
    "kinova_gen3",
    "widowx",
    "yumi",
    "panda",
    "xarm7",
    "lwr",
    "ur5_stick",
    "ur5_planar",
    "ur5_ez",
    "ur5_sawyer",
]
