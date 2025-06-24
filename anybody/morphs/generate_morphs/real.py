import os
from .utils import update_urdf
from anybody.utils.path_utils import get_robot_morphs_dir
from anybody.utils.utils import format_str
import numpy as np
import urdfpy as ud


def create_real_robot_usd(robo_name):
    if robo_name.startswith("panda_"):
        create_real_robot_usd_helper(
            robo_name, robo_base="panda", force_usd_creation=True
        )

    elif robo_name in ["fetch", "jaco2", "kinova_gen3", "widowx", "yumi", "panda"]:
        create_real_robot_usd_helper(
            robo_name, robo_base=robo_name, force_usd_creation=True
        )

    elif robo_name == "xarm7":
        create_real_robot_usd_helper(
            robo_name,
            robo_base=robo_name,
            ee_dir="kinova_gen3",
            force_usd_creation=True,
        )

    elif robo_name == "lwr":
        create_real_robot_usd_helper(
            robo_name, robo_base="lwr", ee_dir="panda", force_usd_creation=True
        )

    elif robo_name.startswith("ur5_"):
        create_real_robot_usd_helper(
            robo_name, robo_base="ur5_base", ee_dir=robo_name, force_usd_creation=True
        )

    else:
        raise ValueError(f"Unknown robot name: {robo_name}")


def replace_urdf_paths(urdf_path, replace_dict):
    new_urdf_path = urdf_path.parent / f"{urdf_path.stem}_new.urdf"

    with open(urdf_path, "r") as f:
        urdf_text = f.read()

    replaced_path_urdf = format_str(urdf_text, replace_dict)

    with open(new_urdf_path, "w") as f:
        f.write(replaced_path_urdf)

    return new_urdf_path


def create_real_robot_usd_helper(
    robo_name, robo_base=None, ee_dir=None, force_usd_creation=False
):
    replace_dict = {}

    if robo_base is None:
        robo_base_dir = get_robot_morphs_dir() / "real" / robo_name

    else:
        robo_base_dir = get_robot_morphs_dir() / "real" / robo_base

    replace_dict = {"ROBOT_BASE_DIR": str(robo_base_dir)}

    if ee_dir is not None:
        ee_dir = get_robot_morphs_dir() / "real" / ee_dir
        replace_dict["EE_DIR"] = str(ee_dir)

    robo_dir = get_robot_morphs_dir() / "real" / robo_name
    if robo_name.startswith("panda_"):
        robo_dir = get_robot_morphs_dir() / "panda_variations" / robo_name

    urdf_path = robo_dir / f"{robo_name}.urdf"
    new_urdf_path = replace_urdf_paths(urdf_path, replace_dict)

    # reset the joint limits for real robots
    # for revolute joints, the limits are [-2*pi, 2*pi]
    # leave the prismatic joints as is, same for continuous joints

    robot_urdfpy = ud.URDF.load(new_urdf_path)

    joint_updates = []

    if not robo_name.startswith("panda_"):
        for joint in robot_urdfpy.joints:
            if joint.joint_type == "revolute":
                joint_updates.append(
                    {
                        "name": joint.name,
                        "lower": -2 * np.pi,
                        "upper": 2 * np.pi,
                    }
                )

    # load the urdf as a string
    with open(new_urdf_path, "r") as f:
        urdf_text = f.read()

    update_urdf(urdf_str=urdf_text, joint_updates=joint_updates)

    usd_path = robo_dir / f"{robo_name}.usd"
    if not os.path.exists(usd_path) or force_usd_creation:
        from anybody.utils.to_usd import ArgsCli, main

        args_cli = ArgsCli(
            input=str(new_urdf_path), output=str(usd_path), headless=True
        )
        main(args_cli)
