
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip
from isaaclab_assets.robots import UR10_CFG

from anybody.utils.path_utils import get_robot_morphs_dir
import urdfpy
from anybody.cfg import cfg as global_cfg

robo_cfg_fns = {
    # "simple_bot": get_simple_bot_cfg,
    # "kinova": get_kinova_cfg,
}


def get_default_robo_cfg(robot_category, robot_name, robo_id):
    # load the urdf using urdfpy and get joint names, joint limits, and initial joint positions

    if robot_category in ["real", "panda_variations"]:
        urdf_path = (
            get_robot_morphs_dir()
            / str(robot_category)
            / f"{robot_name}"
            / f"{robot_name}_new.urdf"
        )        

    else:
        urdf_path = (
            get_robot_morphs_dir()
            / str(robot_category)
            / f"{robot_name}"
            / f"{robot_name}.urdf"
        )


    usd_path = get_robot_morphs_dir() / str(robot_category) / f"{robot_name}" / f"{robot_name}.usd"


    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF file not found at {urdf_path}")

    if robot_category == 'real' and robot_name == 'panda':
        cfg = FRANKA_PANDA_HIGH_PD_CFG
        cfg = cfg.replace(prim_path="{ENV_REGEX_NS}/" + str(robot_category) + "_" + str(robo_id))
        # cfg.spawn.usd_path = str(usd_path)
        return cfg    

    if robot_category == "panda_variations":
        cfg = FRANKA_PANDA_HIGH_PD_CFG
        cfg = cfg.replace(prim_path="{ENV_REGEX_NS}/" + str(robot_category) + "_" + str(robo_id))
        cfg.spawn.usd_path = str(usd_path)
        return cfg

    if robot_category == 'real' and (robot_name in ["ur5_stick", "ur5_planar"]):
        cfg = UR10_CFG
        cfg.prim_path = "{ENV_REGEX_NS}/" + str(robot_category) + "_" + str(robo_id)    
        cfg.spawn.usd_path = str(usd_path) 
        return cfg

    urdfpy_robot = urdfpy.URDF.load(urdf_path)
    joint_names = [j.name for j in urdfpy_robot.joints if j.joint_type != "fixed"]
    
    # print("Number of joints: ", len(joint_names))   
    # print(f"Loading URDF from {urdf_path}. File: robot_morphologies/__init__.py")
    # import pdb; pdb.set_trace()
    
    
    joint_limits = {j.name: j.limit for j in urdfpy_robot.joints}
    joint_types = {j.name: j.joint_type for j in urdfpy_robot.joints}
    joint_lb = [joint_limits[jname].lower for jname in joint_names]
    joint_ub = [joint_limits[jname].upper for jname in joint_names]
    joint_types = [joint_types[jname] for jname in joint_names]

    joint_efforts = {jname: joint_limits[jname].effort for jname in joint_names}
    joint_velocities = {jname: joint_limits[jname].velocity for jname in joint_names}

    # if robot_category != 'real':
    #     joint_efforts = 100.0
    #     joint_velocities = 20.0

    if robot_category != 'real':
        # joint_efforts = 100.0
        # joint_velocities = 10.0
        # stiffness = 400.0
        # damping = 80.0
        
        joint_efforts = 100.0
        joint_velocities = 10.0
        
        
    if robot_category == "prims":
        joint_efforts = 1000.0
        joint_velocities = 20.0
        

    stiffness = global_cfg.STIFFNESS
    damping = global_cfg.DAMPING

    _stiffness, _damping = get_actuator_values(robot_category, robot_name)

    if stiffness < 0.0:
        stiffness = _stiffness
    if damping < 0.0:
        damping = _damping

    init_vals = []
    init_jpos = {}
    for idx, jname in enumerate(joint_names):
        if joint_types[idx] == "continuous":
            init_jpos[jname] = 0.0
        else:
            init_jpos[jname] = (joint_lb[idx] + joint_ub[idx]) / 2.0


    vis_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=global_cfg.ROBOT_COLOR,
    )

    cfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/" + str(robot_category) + "_" + str(robo_id),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(
                get_robot_morphs_dir()
                / str(robot_category)
                / f"{robot_name}"
                / f"{robot_name}.usd"
            ),
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            # visual_material=vis_material,
            # articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            #     enabled_self_collisions=True,
            #     solver_position_iteration_count=8,
            #     solver_velocity_iteration_count=0,
            # ),
            # collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos=init_jpos,
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                velocity_limit=100.0,
                effort_limit=87.0,
                stiffness=800.0,
                damping=40.0,
            ),
            # "joint_acts": ImplicitActuatorCfg(
            # # "joint_acts": DCMotorCfg(
            # # "joint_acts": IdealPDActuatorCfg(
            #     joint_names_expr=joint_names,
            #     effort_limit_sim=87.0,
            #     velocity_limit_sim=2.175,
            #     # effort_limit_sim=joint_efforts,
            #     # velocity_limit_sim=joint_velocities,
            #     # saturation_effort=joint_efforts,
            #     stiffness=stiffness,
            #     damping=damping,
            # ),
        },
        soft_joint_pos_limit_factor=1.0,
    )
    
    return cfg

def get_actuator_values(robot_category, robot_name):
    # make case 

    if robot_category == 'arm_ed':
        stiffness = 400.0
        damping = 80.0
    elif robot_category == 'arm_ur5':
        stiffness = 400.0
        damping = 80.0
    elif robot_category == 'chain':
        stiffness = 400.0
        damping = 5.0
    elif robot_category == "cubes":
        stiffness = 100.0
        damping = 10.0
    elif robot_category in ['mf_v2', 'mf_v1']:
        stiffness = 100.0
        damping = 40.0
    elif robot_category == 'mf_arm':
        stiffness = 400.0
        damping = 80.0
    elif robot_category == 'nlink':
        # stiffness = 10.0
        # damping = 40.0
        stiffness = 400.0
        damping = 80.0
    elif robot_category == 'planar_arm':
        stiffness = 50.0
        damping = 40.0
    elif robot_category == 'prims':
        stiffness = 200.0
        damping = 5.0
    elif robot_category == 'simple_bot':
        stiffness = 400.0
        damping = 80.0
    elif robot_category == 'stick':
        # stiffness = 10.0
        # damping = 40.0
        stiffness = 400.0
        damping = 80.0
    elif robot_category in ['tongs_v2', 'tongs_v1']:
        stiffness = 200.0
        damping = 10.0
    elif robot_category == 'real':
        if robot_name == 'fetch':
            stiffness = 400.0
            damping = 1.0
        elif robot_name == 'jaco2':
            stiffness = 400.0
            damping = 10.0
        elif robot_name == 'kinova_gen3':
            stiffness = 200.0
            damping = 80.0
        elif robot_name == 'lwr':
            stiffness = 400.0
            damping = 80.0
        elif robot_name == 'panda':
            # didn't find - as the code overrides the cfg with FRANKA_PANDA_HIGH_PD_CFG
            stiffness = 400.0
            damping = 80.0
        elif robot_name[:4] == "ur5_":
            stiffness = 400.0
            damping = 1.0
        elif robot_name == 'widowx':
            # didn't observe any change wrt to the stiffness and damping values
            stiffness = 400.0
            damping = 80.0
        elif robot_name == 'xarm7':
            stiffness = 400.0
            damping = 80.0
        elif robot_name == 'yumi':
            stiffness = 100.0
            damping = 1.0
        else:
            raise ValueError(f"Unknown robot name {robot_name} in real robot category")
        
    else:
        raise ValueError(f"Unknown robot category {robot_category}")    
        
    return stiffness, damping