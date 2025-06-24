import os, sys

sys.path.append(os.environ["PROJECT_PATH"])

import numpy as np
import urdfpy
from collections import OrderedDict

from anybody.utils.vis_server import VizServer
from anybody.utils.path_utils import get_robot_morphs_dir, get_tmp_mesh_storage
from anybody.utils.utils import format_str, transform_pcd
from anybody.utils.collision_utils import (
    get_collision_spheres_stick,
    convert_collisions_info_to_str,
)
from .utils import update_urdf, update_joint_params
import sys
import yaml

from .arm_ur5 import get_ee_link

# Customizing the YAML dumper to output dictionaries and lists in a more compact form
# class CompactDumper(yaml.Dumper):
#     def increase_indent(self, flow=False, indentless=False):
#         return super(CompactDumper, self).increase_indent(flow, False)


base_link = """
    <link name="base_link">
    <inertial>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <mass value="1.0"/>
        <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
    </link>
"""

stick_link = """
    <!-- Link LINKNAME -->
    <link name="link_LINKNAME">
        <inertial>
            <origin xyz="HALFLENGTH 0 0" rpy="0 0 0"/>
            <mass value="1.0"/>
            <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
        </inertial>
        <visual>
            <origin xyz="HALFLENGTH 0 0" rpy="0 0 0"/>
            <geometry>
                <box size="LENGTH 0.02 0.02"/>
            </geometry>
            <material name="blue">
                <color rgba="0 0 1 1"/>
            </material>
        </visual>
        <collision>
            <origin xyz="HALFLENGTH 0 0" rpy="0 0 0"/>
            <geometry>
                <box size="COLLISIONLENGTH COLLISIONWIDTH COLLISIONWIDTH"/>
            </geometry>
        </collision>
    </link>
"""

ball_link = """
    <!-- Link ball -->
    <link name="link_LINKNAME">
        <inertial>
            <origin xyz="0.02 0 0" rpy="0 0 0"/>
            <mass value="0.1"/>
            <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
        </inertial>
        <visual>
            <origin xyz="0.02 0 0" rpy="0 0 0"/>
            <geometry>
                <sphere radius="0.02"/>
            </geometry>
            <material name="blue">
                <color rgba="0 0 1 1"/>
            </material>
        </visual>
        <collision>
            <origin xyz="0.02 0 0" rpy="0 0 0"/>
            <geometry>
                <sphere radius="0.02"/>
            </geometry>
        </collision>        
    </link>
"""


base_joint = f"""
    <!-- Joint 0 -->
    <joint name="joint_0" type="revolute">
        <parent link="base_link"/>
        <child link="link_0"/>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-{2 * np.pi}" upper="{2 * np.pi}" effort="20." velocity="5"/>
    </joint>
"""

stick_joint = """
    <!-- Joint JOINTNAME -->
    <joint name="joint_JOINTNAME" type="revolute">
        <parent link="link_PARENTLINK"/>
        <child link="link_CHILDLINK"/>
        <origin xyz="LENGTH 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-3.14" upper="3.14" effort="20." velocity="5"/>
    </joint>
"""

fixed_joint = """
    <!-- Joint JOINTNAME -->
    <joint name="joint_JOINTNAME" type="fixed">
        <parent link="link_PARENTLINK"/>
        <child link="link_CHILDLINK"/>
        <origin xyz="LENGTH 0 0" rpy="0 0 0"/>
    </joint>
"""


# n_link_urdf = f"""<?xml version="1.0"?>
# <robot name="n_link_robot">

# {base_link}
# {stick_link.replace("LINKNAME", "1")}
# {stick_link.replace("LINKNAME", "2")}
# {stick_link.replace("LINKNAME", "3")}
# {base_joint}
# {stick_joint.replace("JOINTNAME", "1").replace("PARENTLINK", "1").replace("CHILDLINK", "2")}
# {stick_joint.replace("JOINTNAME", "2").replace("PARENTLINK", "2").replace("CHILDLINK", "3")}


# </robot>
# """

# n_link_urdf = format_str(n_link_urdf, OrderedDict([
#     ("COLLISIONLENGTH", "0.8"),
#     ("COLLISIONWIDTH", "0.015"),
#     ("HALFLENGTH", "0.5"),
#     ("LENGTH", "1"),
# ]))


def get_parallel_ee_link(parent_link, origin, displacement=0.07, thickness=0.01, width=0.02, ee_length=0.05):
    
    # initialize a l length cuboid
    length = displacement    
    
    link = f"""
    <!-- Link parallel -->
    <link name="link_parallel">
        <inertial>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <mass value="0.1"/>
            <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>   
        </inertial>
        <visual>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <box size="{thickness} {length} {width}"/>
            </geometry>
            <material name="blue">
                <color rgba="0 0 1 1"/>
            </material>
        </visual>
        <collision>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <box size="{thickness} {length} {width}"/>  
            </geometry>
        </collision>
    </link>
    
    <!-- fixed joint with last link-->
    <joint name="joint_parallel" type="fixed">
        <parent link="{parent_link}"/>
        <child link="link_parallel"/>
        <origin xyz="{thickness/2 + origin[0]} {origin[1]} {origin[2]}" rpy="0 0 0"/>
    </joint>
    """
    
    link += get_ee_link("link_parallel", 
                        ee_length=ee_length, 
                        displacement=length/2, 
                        joint_origin=(thickness/2, 0, 0), 
                        joint_rpy=(np.pi/2, 0, np.pi/2),
                        ee_width=width,
                        ee_thickness=0.005)
                
    return link            


def get_claw_ee_link(parent_link, origin, ee_length=0.05, margin=0.005, thickness=0.01):
    
    ee_size = ee_length    
    def get_ee_link(collision=True):
    
        ee_link = f"""
        <link name="ee_link_SIDE">
            <inertial>
                <origin xyz="{ee_size/2} 0 0" rpy="0 0 0"/>
                <mass value="1.0"/>
                <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
            </inertial>
            <visual>
                <origin xyz="{ee_size/2} 0 0" rpy="0 0 0"/>
                <geometry>
                    <box size="{ee_size} {margin} {thickness}"/>
                </geometry>
                <material name="blue">
                    <color rgba="0 0 1 1"/>
                </material>
            </visual>
"""

        if collision:
            ee_link += f"""
            <collision>
                <origin xyz="{ee_size/2} 0 0" rpy="0 0 0"/>
                <geometry>
                    <box size="{ee_size - margin} {margin} {thickness}"/>
                </geometry>
            </collision>
"""

        ee_link += """
        </link>     
"""
        return ee_link


    ee_arm_joint = f"""
    <joint name="joint_ee_SIDE_rev" type="revolute">
        <parent link="{parent_link}"/>
        <child link="ee_link_SIDE"/>
        <origin xyz="{origin[0]} {origin[1]} {origin[2]}" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-1.57" upper="1.57" effort="1" velocity="1"/>
    </joint>
"""

    ee_ee_joint = f"""
    <joint name="joint_ee_SIDE_rev" type="revolute">
        <parent link="ee_link_PARENT"/>
        <child link="ee_link_SIDE"/>
        <origin xyz="{ee_size} 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-1.57" upper="1.57" effort="1" velocity="1"/>
    </joint>
"""

    urdf = f"""
    {get_ee_link(collision=False).replace("SIDE", "left1")}
    {get_ee_link(collision=False).replace("SIDE", "right1")}
    {ee_arm_joint.replace("SIDE", "left1")}    
    {ee_arm_joint.replace("SIDE", "right1")}    
    {get_ee_link(collision=True).replace('SIDE', "left2")}
    {get_ee_link(collision=True).replace('SIDE', "right2")}
    {ee_ee_joint.replace("PARENT", "left1").replace("SIDE", 'left2')}
    {ee_ee_joint.replace("PARENT", "right1").replace("SIDE", 'right2')}
"""
    return urdf



def create_n_link_urdf(n=2, length=0.5, ee_type=None, ee_kwargs={}):
    assert n >= 1, "n must be greater than or equal to 1"

    links = base_link

    for link_idx in range(n):
        links += stick_link.replace("LINKNAME", str(link_idx))
        links += "\n"


    joints = base_joint
    for joint_idx in range(1, n):
        j = format_str(
            stick_joint,
            OrderedDict(
                [
                    ("JOINTNAME", str(joint_idx)),
                    ("PARENTLINK", str(joint_idx - 1)),
                    ("CHILDLINK", str(joint_idx)),
                ]
            ),
        )
        joints += j + "\n"

    if ee_type == "ball":
        # this case is exactly same as the n-link robot case, so disabling
        raise NotImplementedError
        ee_link = f"""
        {ball_link.replace("LINKNAME", str(n))}
        {fixed_joint.replace("JOINTNAME", str(n)).replace("PARENTLINK", str(n - 1)).replace("CHILDLINK", str(n))}
        """

    elif ee_type == "parallel":
        ee_link = get_parallel_ee_link(f"link_{n-1}", origin=(length, 0, 0), **ee_kwargs)
    
    elif ee_type == "claw":
        ee_link = get_claw_ee_link(f"link_{n-1}", origin=(length, 0, 0), **ee_kwargs)
    

    n_link_urdf = f"""<?xml version="1.0"?>
<robot name="n_link_robot">
{links}
{joints}
{ee_link}
</robot>
"""

    collision_length = length * 0.8
    n_link_urdf = format_str(
        n_link_urdf,
        OrderedDict(
            [
                ("COLLISIONLENGTH", f"{collision_length}"),
                ("COLLISIONWIDTH", "0.01"),
                ("HALFLENGTH", f"{length / 2}"),
                ("LENGTH", f"{length}"),
            ]
        ),
    )

    # joint_names = [f"joint_{i}" for i in range(n)]
    # joint_ub = [np.pi] * n
    # joint_lb = [-np.pi] * n
    # joint_effort = [100.0] * n
    # joint_velocity = [5.0] * n

    # joint_updates = [
    #     {"name": j, "lower": lb, "upper": ub, "effort": e, "velocity": v}
    #     for j, lb, ub, e, v in zip(
    #         joint_names, joint_lb, joint_ub, joint_effort, joint_velocity
    #     )
    # ]

    # n_link_urdf = update_urdf(n_link_urdf, joint_updates)


    # collision_spheres = {f"link_{i}": [] for i in range(n + 1)}

    collision_spheres = {}

    for link_idx in range(n):
        collision_spheres[f"link_{link_idx}"] = get_collision_spheres_stick(
            length, dist=0.03, radius=0.02
        )
        # radius = 0.02
        # dist = 0.03

        # start_pt = np.array([0.0, 0.0, 0.0])
        # cur_pt = start_pt
        # while cur_pt[0] < length:
        #     next_pt = cur_pt + np.array([dist, 0.0, 0.0])
        #     collision_spheres[f"link_{link_idx}"].append((next_pt, radius))
        #     cur_pt = next_pt

    collision_spheres[f"link_{n}"] = [(np.array([0.0, 0.0, 0.0]), 0.02)]

    return n_link_urdf, collision_spheres


def collision_info_to_yaml(collision_spheres, config_path):
    info_dict = {}
    info_dict["collision_link_names"] = list(collision_spheres.keys())
    info_dict["collision_spheres"] = {}
    for link_name in info_dict["collision_link_names"]:
        info_dict["collision_spheres"][link_name] = []
        for center, radius in collision_spheres[link_name]:
            info_dict["collision_spheres"][link_name].append(
                {
                    "center": np.round(center, decimals=3).tolist(),
                    "radius": np.round(radius, decimals=3).item(),
                }
            )
    # write to yaml
    with open(config_path, "w+") as fn:
        yaml.dump(info_dict, fn)


# def create_urdf_files():

#     lengths = np.linspace(0.5, 0.2, 4)
#     n_links = [2, 3, 4, 5]
#     # names = ['l2', 'l3', 'l4', 'l5']

#     (get_robot_morphs_dir() / "n_link").mkdir(parents=True, exist_ok=True)

#     for n, l in zip(n_links, lengths):
#         urdf, _ = create_n_link_urdf(n, l)
#         urdf_path = get_robot_morphs_dir() / "n_link" / f"{n}_link.urdf"

#         with open(urdf_path, "w") as f:
#             f.write(urdf)


def save_urdfs(to_usd=False, randomized=False):
    
    if not randomized:
        n_links = [2, 3]
        lengths = [0.5, 0.3]
        ee_lengths = [0.06, 0.04]
    else:
        lengths = np.linspace(0.1, 0.5, 50)
        n_links = np.random.randint(2, 6, 50)
        ee_lengths = np.random.uniform(0.03, 0.06, 50)

    robo_names = []

    robo_count = 0

    for ee_type in ['parallel', 'claw']:
        for n, l, el in zip(n_links, lengths, ee_lengths):
            final_urdf, collision_spheres_dict = create_n_link_urdf(n, l, ee_type=ee_type, ee_kwargs={'ee_length': el})
            
            robo_name = f"{n}_link_{ee_type}" if not randomized else f"{ee_type[0]}{robo_count}"
            robo_count += 1
            
            robot_dir = get_robot_morphs_dir() / "planar_arm" / robo_name

            robo_names.append(str(robot_dir))
            robot_dir.mkdir(parents=True, exist_ok=True)
            urdf_path = robot_dir / f"{robo_name}.urdf"
            with open(urdf_path, "w") as f:
                f.write(final_urdf)

            final_urdf = update_joint_params(urdf_path)

            if to_usd:

                usd_path = robot_dir / f"{robo_name}.usd"
                # check if usd exists
                if not os.path.exists(usd_path):
                    from anybody.utils.to_usd import ArgsCli, main
                    args_cli = ArgsCli(input=str(urdf_path), output=str(usd_path), headless=True)
                    main(args_cli)


                # robo info

                # base_link_name = "base_link"
                # ee_link_name = f"link_{n}"
                # joint_names = [f"joint_{i}" for i in range(n)]

                # joint_default_vals = [0.0] * n
                # collision_ignore = {f"link_{i}": [f"link_{i-1}"] for i in range(1, n + 1)}

                # self_collision_buffer = {f"link_{i}": 0.0 for i in range(n + 1)}
                # null_space_weight = [1.0] * len(joint_names)
                # cspace_distance_weight = [1.0] * len(joint_names)

                # # create collision spheres yml
                # collision_spheres_yml = robot_dir / "collision_spheres.yml"
                # collision_spheres_str = convert_collisions_info_to_str(
                #     collision_spheres_dict
                # )

                # # write collision spheres yml
                # with open(collision_spheres_yml, "w") as f:
                #     f.write(collision_spheres_str)

                # # writing the robot config file
                # # load the template from robot_morphologies / 'template_cfg.yml'
                # with open(get_robot_morphs_dir() / "template_cfg.yml", "r") as f:
                #     template_cfg = yaml.safe_load(f)

                # # write the cfg file
                # template_cfg["robot_cfg"]["kinematics"]["usd_path"] = str(usd_path)
                # template_cfg["robot_cfg"]["kinematics"]["usd_robot_root"] = f"/{n}_link"
                # template_cfg["robot_cfg"]["kinematics"]["urdf_path"] = str(urdf_path)

                # template_cfg["robot_cfg"]["kinematics"]["asset_root_path"] = str(robot_dir)
                # template_cfg["robot_cfg"]["kinematics"]["base_link"] = base_link_name
                # template_cfg["robot_cfg"]["kinematics"]["ee_link"] = ee_link_name

                # template_cfg["robot_cfg"]["kinematics"]["collision_link_names"] = [
                #     "link_rev_z",
                #     "link_ball",
                # ]
                # template_cfg["robot_cfg"]["kinematics"]["collision_spheres"] = str(
                #     collision_spheres_yml
                # )

                # template_cfg["robot_cfg"]["kinematics"]["self_collision_ignore"] = (
                #     collision_ignore
                # )
                # template_cfg["robot_cfg"]["kinematics"]["self_collision_buffer"] = (
                #     self_collision_buffer
                # )

                # template_cfg["robot_cfg"]["kinematics"]["mesh_link_names"] = [
                #     "link_rev_z",
                #     "link_ball",
                # ]
                # template_cfg["robot_cfg"]["kinematics"]["cspace"]["joint_names"] = (
                #     joint_names
                # )
                # template_cfg["robot_cfg"]["kinematics"]["cspace"]["retract_config"] = (
                #     joint_default_vals
                # )
                # template_cfg["robot_cfg"]["kinematics"]["cspace"]["null_space_weight"] = (
                #     null_space_weight
                # )
                # template_cfg["robot_cfg"]["kinematics"]["cspace"][
                #     "cspace_distance_weight"
                # ] = cspace_distance_weight

                # curobo_cfg = robot_dir / "curobo_cfg.yml"
                # with open(curobo_cfg, "w") as f:
                #     yaml.safe_dump(template_cfg, f, sort_keys=False)

    return robo_names



if __name__ == "__main__":
    # write urdf to file

    urdf = create_n_link_urdf(3, 0.3, 'claw')[0]

    urdf_path = get_tmp_mesh_storage() / "planar_arm.urdf"
    with open(urdf_path, "w") as f:
        f.write(urdf)

    urdfpy_robot = urdfpy.URDF.load(urdf_path)
    vis = VizServer()

    joint_names = [j.name for j in urdfpy_robot.joints if j.joint_type != "fixed"]

    jlimits = {j.name: j.limit for j in urdfpy_robot.joints}
    joint_lb = [jlimits[jname].lower for jname in joint_names]
    joint_ub = [jlimits[jname].upper for jname in joint_names]

    for _ in range(20):
        # joint_vals = np.zeros(len(joint_names))
        joint_vals = np.random.uniform(joint_lb, joint_ub)
        # joint_vals[:3] = [0, 0.0, 0.0]

        vis.view_robot(
            urdfpy_robot,
            joint_names,
            joint_vals,
            "obj_robot",
            0x00FF00,
            mesh_type="visual",
        )
        vis.view_robot(
            urdfpy_robot,
            joint_names,
            joint_vals,
            "obj_robot_col",
            0x0000FF,
            mesh_type="collision",
        )
        input("Press Enter to continue...")