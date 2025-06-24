import os
import numpy as np
import urdfpy

from anybody.utils.vis_server import VizServer
from anybody.utils.path_utils import get_tmp_mesh_storage, get_robot_morphs_dir
from anybody.utils.utils import format_str
from .utils import update_urdf, get_5dof_base_link, update_joint_params


def get_joint_link(
    parent_link, child_link, origin, axis, joint_ll, joint_ul, joint_effort, joint_vel
):
    joint_link = f"""
    <joint name="{child_link}_joint" type="revolute">
        <origin xyz="{origin[0]} {origin[1]} {origin[2]}" rpy="0 0 0"/>
        <parent link="{parent_link}"/>
        <child link="{child_link}"/>
        <axis xyz="{axis[0]} {axis[1]} {axis[2]}"/>
        <limit lower="{joint_ll}" upper="{joint_ul}" effort="{joint_effort}" velocity="{joint_vel}"/>
    </joint>
"""
    return joint_link


def get_fixed_joint(parent_link, child_link, origin, rpy):
    joint_link = f"""
    <joint name="{child_link}_joint" type="fixed">
        <origin xyz="{origin[0]} {origin[1]} {origin[2]}" rpy="{rpy[0]} {rpy[1]} {rpy[2]}"/>
        <parent link="{parent_link}"/>
        <child link="{child_link}"/>
    </joint>
"""
    return joint_link


def get_single_flink(linkname, length, width, thickness, radius):
    # the link is a cuboidal link,
    # the center of the link is at (radius + length/2, 0, 0)
    # the link is oriented along the x-axis (length, width, thickness)

    # the joint is a revolute joint
    # the joint axis is along the y-axis
    # the joint is at (0 0 0)

    link = f"""
    <link name="{linkname}">
        <inertial>
            <origin xyz="{radius + length/2} 0 0" rpy="0 0 0"/>
            <mass value="1.0"/>
            <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
        </inertial>
        <visual>
            <origin xyz="{radius + length/2} 0 0" rpy="0 0 0"/>
            <geometry>
                <box size="{length} {width} {thickness}"/>
            </geometry>
        </visual>
        <collision>
            <origin xyz="{radius + length/2} 0 0" rpy="0 0 0"/>
            <geometry>
                <box size="{length} {width} {thickness}"/>
            </geometry>
        </collision>
    </link>
"""

    return link


def get_finger(
    finger_idx,
    n_links,
    link_lengths,
    base_radius,
    yaw,
    pitch=np.pi / 6,
    link_width=0.01,
    link_thickness=0.001,
    margin=0.001,
    joint_lower_limits=0.0,
    joint_upper_limits=np.pi / 2,
):
    # n_links: number of links in the finger
    # link_lengths: list of link lengths
    # link_width: width of each link
    # link_thickness: thickness of each link
    # link_radius: radius of each link
    # joint_upper_limits: list of joint upper limits
    # joint_lower_limits: list of joint lower limits

    if not isinstance(joint_upper_limits, (list, tuple)):
        joint_upper_limits = [joint_upper_limits] * n_links
    if not isinstance(joint_lower_limits, (list, tuple)):
        joint_lower_limits = [joint_lower_limits] * n_links

    finger = ""

    parent_linkname = "spherical_baselink"
    origin = [0, 0, 0]

    # first link

    linkname = f"link_f{finger_idx}_l0"
    link = get_single_flink(
        linkname,
        link_lengths[0],
        link_width,
        link_thickness,
        margin,
    )
    
    
    # origin depends on yaw
    # x = np.cos(yaw) * (base_radius + margin)
    # y = np.sin(yaw) * (base_radius + margin)
    
    xy = (base_radius + margin) * np.array([np.cos(yaw), np.sin(yaw)]) 

    joint = get_fixed_joint(
        parent_link=parent_linkname,
        child_link=linkname,
        origin=[*xy, 0],
        rpy=[0, pitch, yaw],
    )

    finger += link + joint

    for i in range(1, n_links):
        origin = [link_lengths[i - 1] + margin, 0, 0]
        linkname = f"link_f{finger_idx}_l{i}"
        link = get_single_flink(
            linkname,
            link_lengths[i],
            link_width,
            link_thickness,
            margin,
        )
        joint = get_joint_link(
            parent_link=f"link_f{finger_idx}_l{i-1}",
            child_link=linkname,
            origin=origin,
            axis=[0, 1, 0],
            joint_ll=joint_lower_limits[i],
            joint_ul=joint_upper_limits[i],
            joint_effort=1,
            joint_vel=1,
        )

        finger += link + joint

    return finger


def get_urdf(n_fingers, per_finger_lengths, equally_distanced=True, finger_params={}):
    # per_finger_lengths has n_fingers elements
    # each element is a list of link lengths for that finger

    base_radius = 0.01

    base_link = """
    <link name="base_link"/>
"""

    base_spherical_link = f"""
    <link name="spherical_baselink">
        <inertial>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <mass value="1.0"/>
            <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
        </inertial>
        <visual>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <sphere radius="{base_radius}"/>
            </geometry>        
        </visual>
        <collision>
            <origin xyz="0 0 0" rpy="0 0 0"/>
            <geometry>
                <sphere radius="{base_radius}"/>
            </geometry>
        </collision>
    </link>
"""
    dof5_link = get_5dof_base_link("spherical_baselink", pos_only=True)

    if n_fingers == 3 and not equally_distanced:
        u = np.random.uniform(2 * np.pi / 3, np.pi - np.pi / 12)
        thetas = [0, u, -u]
    elif n_fingers == 4 and not equally_distanced:
        u = np.random.uniform(np.pi / 6, np.pi / 3, 2)
        thetas = [u[0], u[1] + np.pi / 2, -u[0], -u[1] - np.pi / 2]
    elif n_fingers == 5 and not equally_distanced:
        # similar to 4 fingers and thumb
        u = np.random.uniform(np.pi / 3, np.pi / 2 - np.pi / 12)
        thetas4 = np.linspace(-u, u, 4, endpoint=True)
        theta5 = np.random.uniform(-np.pi / 12, np.pi / 12) + np.pi
        thetas = np.concatenate((thetas4, [theta5]))
    else:
        thetas = np.linspace(0, 2 * np.pi, n_fingers, endpoint=False)
    all_fingers = ""

    for idx, per_f_length in enumerate(per_finger_lengths):
        finger1 = get_finger(
            idx,
            len(per_f_length),
            per_f_length,
            base_radius=base_radius,
            yaw=thetas[idx],
            **finger_params,
        )
        all_fingers += finger1

    urdf = f"""<?xml version="1.0"?>
<robot name="tongs">
{base_link}
{base_spherical_link}
{dof5_link}
{all_fingers}
</robot>
"""
    return urdf


def save_urdfs(to_usd=False):
    # variables
    n_fingers = [2, 3, 4, 5]
    equidistant = [True, False]

    finger_link_lengths1 = [0.04, 0.03, 0.02]
    finger_link_lengths2 = [0.03, 0.05]

    for n in n_fingers:
        for eq_idx, eq in enumerate(equidistant):
            for fl_idx, fl in enumerate([finger_link_lengths1, finger_link_lengths2]):
                finger_params = {
                    "pitch": np.pi / 4,
                    "link_thickness": 0.002,
                    "link_width": 0.005,
                }

                if n == 5 and eq_idx == 1 and fl_idx == 0:
                    fls = [fl] * 4
                    fls.append([0.03, 0.02, 0.01])
                else:
                    fls = [fl] * n

                urdf = get_urdf(
                    n, fls, equally_distanced=eq, finger_params=finger_params
                )

                robo_name = f"finger_{n}f_{eq_idx}_{fl_idx}"

                robo_dir = get_robot_morphs_dir() / "mf_v1" / robo_name
                robo_dir.mkdir(parents=True, exist_ok=True)

                urdf_path = robo_dir / f"{robo_name}.urdf"

                with open(urdf_path, "w") as f:
                    f.write(urdf)

                urdf = update_joint_params(urdf_path)

                if to_usd:
                    usd_path = robo_dir / f"{robo_name}.usd"
                    if not os.path.exists(usd_path):
                        from anybody.utils.to_usd import ArgsCli, main
                        args_cli = ArgsCli(input=str(urdf_path), output=str(usd_path), headless=True)
                        main(args_cli)


if __name__ == "__main__":
    per_finger_length = [[0.1, 0.05, 0.02], [0.1, 0.05, 0.02], [0.1, 0.05, 0.02]]

    # per_finger_length = [[0.03, 0.05, 0.02]] * 5

    # per_finger_length = [[0.05, 0.03, 0.02]] * 4 + [[0.04, 0.02, 0.02]]

    finger_params = {
        "pitch": np.pi / 4,
        "link_thickness": 0.005,
        "link_width": 0.005,
    }

    urdf_path = get_tmp_mesh_storage() / "fingers.urdf"

    urdf = get_urdf(
        n_fingers=len(per_finger_length),
        per_finger_lengths=per_finger_length,
        equally_distanced=False,
        finger_params=finger_params,
    )
    with open(urdf_path, "w") as f:
        f.write(urdf)

    urdfpy_robot = urdfpy.URDF.load(urdf_path)
    vis = VizServer()

    # get all joint names, and their limits, and sample random joint values

    # joint_names = [j.name for j in urdfpy_robot.joints]
    # only revolute joints
    joint_names = [j.name for j in urdfpy_robot.joints if j.joint_type == "revolute"]

    # joint_lb = [-np.pi] * len(joint_names)
    # joint_ub = [np.pi] * len(joint_names)

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
