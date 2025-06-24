import numpy as np
import trimesh
import time
import threading

import abc
from urdfpy import URDF


def load_arm_with_gripper(
    env, gripper_type, base_pos=[0.0, 0.0, 0], base_ori=[0.0, 0.0, 0.0, 1.0]
):
    if gripper_type == "kinova":
        ur5_path = "gripper/ur5_kinova.urdf"
        gripper = GripperKinova3F
    # if gripper_type == "barret":
    #     ur5_path = "gripper/ur5_barret.urdf"
    #     gripper = GripperBarrettHand
    # if gripper_type == "robotiq140":
    #     ur5_path = "gripper/ur5_robotiq_140.urdf"
    #     gripper = GripperRobotiq2F140
    # if gripper_type == "planar":
    #     # x, y of the plane
    #     plane_shape = [0.1, 0.15]
    #     ur5_path = "gripper/ur5_planar.urdf"

    #     gripper = GripperPlanar
    # if gripper_type == "stick":
    #     # height and radius
    #     cylinder_shape = [0.1, 0.01]
    #     ur5_path = "gripper/ur5_stick.urdf"
    #     gripper = GripperPlanar

    # ur5 = env.pb_client.load_urdf(
    #     str(env.assets_dir / ur5_path),
    #     base_pos=base_pos, base_ori=base_ori)

    robot_urdfpy = URDF.load(str(env.assets_dir / ur5_path))

    homej = np.array([-1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi

    # Get revolute joint indices of robot (skip fixed joints).
    n_joints = env.pb_client.getNumJoints(ur5)
    all_joints = [env.pb_client.getJointInfo(ur5, i) for i in range(n_joints)]

    last_link_idx = None
    for joint_info in all_joints:
        if joint_info[1].decode("utf-8") == "ee_fixed_joint":
            last_link_idx = joint_info[0]
            break
    print(f"EE link index is: {last_link_idx}")

    # ee = gripper(env, ur5)
    # print(f"EE joint names: {ee.joint_names}")
    # arm_joints = [j[0] for j in all_joints if (j[2] == p.JOINT_REVOLUTE) and not (j[1].decode("utf-8") in ee.joint_names)]

    # env.joints = [j[0] for j in joints if j[2] == p.JOINT_REVOLUTE]

    # Move robot to home joint configuration.
    # for i in range(len(arm_joints)):
    #     env.pb_client.resetJointState(ur5, arm_joints[i], homej[i])

    # ee.open()

    # env.pb_client.setJointMotorControlArray(
    #     ur5,
    #     arm_joints,
    #     p.POSITION_CONTROL,
    #     targetPositions=homej
    # )
    gripper_config = {
        "arm_id": ur5,
        "homej": homej,
        "ee": ee,
        "ee_tip": last_link_idx,
        "ee_tip_name": "wrist_3_link",
        # 'ee_tio_name': 'ee_fixed_joint',
        "arm_joints": arm_joints,
        "urdfpy": robot_urdfpy,
    }

    return gripper_config


class GripperBase(abc.ABC):
    r"""Base class for all grippers.
    Any gripper should subclass this class.
    You have to implement the following class method:
        - load(): load URDF and return the body_id
        - configure(): configure the gripper (e.g. friction)
        - open(): open gripper
        - close(): close gripper
        - get_pos_offset(): return [x, y, z], the coordinate of the grasping center relative to the base
        - get_orn_offset(): the base orientation (in quaternion) when loading the gripper
        - get_vis_pts(open_scale): [(x0, y0), (x1, y1), (x2, y2s), ...], contact points for visualization (in world coordinate)
    """

    @abc.abstractmethod
    def configure(self):
        pass

    # @abc.abstractmethod
    # def open(self):
    #     pass

    # @abc.abstractmethod
    # def close(self):
    #     pass

    @abc.abstractmethod
    def move(self, val):
        pass

    @abc.abstractmethod
    def sample_action(self):
        pass


class GripperKinova3F(GripperBase):
    def __init__(self, env, arm_id):
        r"""Initialization of Kinova 3finger gripper
        specific args for Kinova 3finger gripper:
            - gripper_size: global scaling of the gripper when loading URDF
        """
        super().__init__()
        self.joint_names = [
            "j2n6s300_joint_finger_1",
            "j2n6s300_joint_finger_tip_1",
            "j2n6s300_joint_finger_2",
            "j2n6s300_joint_finger_tip_2",
            "j2n6s300_joint_finger_3",
            "j2n6s300_joint_finger_tip_3",
        ]

        # self._bullet_client = env.pb_client
        self._gripper_size = 1.0
        self.mount_gripper_id = arm_id
        self.n_links_before = 9

        # offset the gripper to a down facing pose for grasping
        # self._pos_offset = np.array([0, 0, 0.207 * self._gripper_size]) # offset from base to center of grasping
        # self._orn_offset = self._bullet_client.getQuaternionFromEuler([0, 0, -np.pi/2])

        # define force and speed (grasping)
        self._force = 1000
        self._grasp_speed = 1.5

        finger1_joint_ids = [0, 1]
        finger2_joint_ids = [2, 3]
        finger3_joint_ids = [4, 5]
        self._finger_joint_ids = (
            finger1_joint_ids + finger2_joint_ids + finger3_joint_ids
        )
        self._driver_joint_id = self._finger_joint_ids[0]
        self._follower_joint_ids = self._finger_joint_ids[1:]

        # self.configure(self.mount_gripper_id, self.n_links_before)

        self._joint_lower = 0.2
        self._joint_upper = 1.3

        self.action_dim = 6
        self.action_type = float

        self.actuated_joints = [
            id + self.n_links_before for id in self._finger_joint_ids
        ]
        joint_name = []
        for joint_id in self.actuated_joints:
            name = self._bullet_client.getJointInfo(self.mount_gripper_id, joint_id)[
                1
            ].decode("utf-8")
            joint_name.append(name)

        self.joint_names = joint_name

        self.end_fingers = [id + self.n_links_before for id in [1, 3, 5]]
        self.end_finger_names = [
            "j2n6s300_link_finger_tip_1",
            "j2n6s300_link_finger_tip_2",
            "j2n6s300_link_finger_tip_3",
        ]
        self.finger_contact_normals = None
        self.finger_contact_pts = None

    def get_finger_joint_indices(self, finger_idx):
        if finger_idx == 0:
            return [0, 1], self._joint_lower
        elif finger_idx == 1:
            return [2, 3], self._joint_lower
        elif finger_idx == 2:
            return [4, 5], self._joint_lower

    def contact_pts(self, mesh):
        if self.finger_contact_normals is not None:
            return self.finger_contact_pts, self.finger_contact_normals

        pts, face_indices = trimesh.sample.sample_surface(mesh, 5000)[:2]
        surface_normals = mesh.face_normals[face_indices]
        assert surface_normals.shape == (len(pts), 3)

        # pts = np.copy(mesh.vertices)
        first_quartile = np.percentile(pts[:, 2], 40)
        third_quartile = np.percentile(pts[:, 2], 60)
        chosen_z = np.logical_and(
            pts[:, 2] > first_quartile, pts[:, 2] < third_quartile
        )
        pts = pts[chosen_z]
        surface_normals = surface_normals[chosen_z]

        finger_tip_idx = np.argmax(pts[:, 0])
        tip_y = pts[finger_tip_idx][1]
        tip_x = pts[finger_tip_idx][0]

        pointy_edge_mask = pts[:, 0] > (tip_x - 0.001)
        inner_mask = np.logical_and(pts[:, 1] > tip_y, pts[:, 0] > (tip_x - 0.015))

        total_mask = np.logical_or(pointy_edge_mask, inner_mask)
        self.finger_contact_pts = pts[total_mask]
        self.finger_contact_normals = surface_normals[total_mask]

        return pts[total_mask], surface_normals[total_mask]

    def move(self, val: np.ndarray, absolute=False):
        assert len(val) == 6
        if absolute:
            target_pos = val
        else:
            target_pos = self._joint_lower * val + self._joint_upper * (1 - val)
        self._bullet_client.setJointMotorControlArray(
            self.mount_gripper_id,
            [id + self.n_links_before for id in self._finger_joint_ids],
            self._bullet_client.POSITION_CONTROL,
            targetPositions=target_pos,
            # targetPositions=[pos_sum-pos, pos, pos_sum-pos, pos, pos_sum-pos],
            forces=[self._force] * len(self._finger_joint_ids),
            positionGains=[1.2] * len(self._finger_joint_ids),
        )
        num_steps = 0
        max_num_steps = 50
        while num_steps < max_num_steps:
            cur_joint_state = self._bullet_client.getJointStates(
                self.mount_gripper_id,
                self.n_links_before + np.array(self._finger_joint_ids),
            )
            cur_pos = [j[0] for j in cur_joint_state]
            diff = np.abs(target_pos - cur_pos).sum()
            if diff < 1e-4:
                break
            # print(f"curent diff: {diff}")
            self._bullet_client.stepSimulation()
            num_steps += 1

        cur_joint_state = self._bullet_client.getJointStates(
            self.mount_gripper_id,
            self.n_links_before + np.array(self._finger_joint_ids),
        )
        cur_pos = [j[0] for j in cur_joint_state]

        # print(f"Target pos: {target_pos}, Current pos: {cur_pos}")
        return cur_pos

    def sample_action(self):
        return np.random.rand(6)

    def load(self, basePosition):
        gripper_urdf = "assets/gripper/kinova_3f/model.urdf"
        body_id = self._bullet_client.loadURDF(
            gripper_urdf,
            flags=self._bullet_client.URDF_USE_SELF_COLLISION,
            globalScaling=self._gripper_size,
            basePosition=basePosition,
        )
        return body_id

    def configure(self, mount_gripper_id, n_links_before):
        # Set friction coefficients for gripper fingers
        for i in range(
            n_links_before, self._bullet_client.getNumJoints(mount_gripper_id)
        ):
            self._bullet_client.changeDynamics(
                mount_gripper_id,
                i,
                lateralFriction=1.0,
                spinningFriction=1.0,
                rollingFriction=0.0001,
                frictionAnchor=True,
            )

    def step_constraints(self, mount_gripper_id, n_joints_before):
        pos = self._bullet_client.getJointState(
            mount_gripper_id, self._driver_joint_id + n_joints_before
        )[0]
        pos_sum = 1.4
        self._bullet_client.setJointMotorControlArray(
            mount_gripper_id,
            [id + n_joints_before for id in self._follower_joint_ids],
            self._bullet_client.POSITION_CONTROL,
            targetPositions=[pos_sum - pos, pos, pos_sum - pos, pos, pos_sum - pos],
            forces=[self._force] * len(self._follower_joint_ids),
            positionGains=[1.2] * len(self._follower_joint_ids),
        )
        return pos

    def open(self, open_scale=1.0):
        target_pos = (
            open_scale * self._joint_lower + (1 - open_scale) * self._joint_upper
        )  # recalculate scale because larger joint position corresponds to smaller open width
        self._bullet_client.setJointMotorControl2(
            self.mount_gripper_id,
            self._driver_joint_id + self.n_links_before,
            self._bullet_client.POSITION_CONTROL,
            targetPosition=target_pos,
            force=self._force,
        )
        for i in range(240 * 2):
            pos = self.step_constraints(self.mount_gripper_id, self.n_links_before)
            if np.abs(target_pos - pos) < 1e-5:
                break
            self._bullet_client.stepSimulation()

    def close(self):
        self._bullet_client.setJointMotorControl2(
            self.mount_gripper_id,
            self._driver_joint_id + self.n_links_before,
            self._bullet_client.VELOCITY_CONTROL,
            targetVelocity=self._grasp_speed,
            force=self._force,
        )
        for i in range(240 * 4):
            pos = self.step_constraints(self.mount_gripper_id, self.n_links_before)
            if pos > self._joint_upper:
                break
            self._bullet_client.stepSimulation()
