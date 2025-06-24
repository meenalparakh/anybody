import random
import numpy as np
import time
from scipy.spatial.transform import Rotation as R
from anybody.envs.dexterous.gripper import GripperBase
import os
import trimesh
from scipy.optimize import minimize, LinearConstraint
from urdfpy import URDF
import typing as T
from anybody.utils import utils
from anybody.utils.vis_server import VizServer

os.environ["PYOPENGL_PLATFORM"] = "egl"


def dist(cur_pos, target_pos):
    return np.square(cur_pos[:2] - target_pos[:2]).sum()


class GripperControl:
    def __init__(self, robot, vis=False) -> None:
        if vis:
            self.vis = VizServer()
        else:
            self.vis = None
        urdf_path = robot.robot_urdf
        self.urdfpy = URDF.load(urdf_path)

        self.all_joint_names: T.List[str] = robot.act_info["joint_names"]
        self.homej = robot.joint_pos

        self.arm_joint_names = self.all_joint_names[:6]
        self.ee_joint_names = self.all_joint_names[6:]

        self.n_arm_joints = len(self.arm_joint_names)
        self.n_ee_joints = len(self.ee_joint_names)

        self.ee_link_name = "wrist_3_link"
        self.ee_joint = "ee_fixed_joint"

        for l in self.urdfpy.links:
            if l.name == self.ee_link_name:
                self.urdfpy_ee_link = l
                break

        self.base_pose = robot.pose

        # bounds
        # lowerLimits=[-3 * np.pi / 2, -2.3562, -np.pi, -np.pi, -np.pi, -np.pi]
        # upperLimits=[-np.pi / 2, 0, np.pi, np.pi, np.pi, np.pi]
        # ee_lower = self.ee._joint_lower * np.ones(len(self.ee.actuated_joints))
        # ee_upper = self.ee._joint_upper * np.ones(len(self.ee.actuated_joints))
        # lowerLimits += ee_lower.tolist()
        # upperLimits += ee_upper.tolist()

        lowerLimits = robot.act_info["joint_lb"]
        upperLimits = robot.act_info["joint_ub"]

        self.joint_bounds = list(zip(lowerLimits, upperLimits))

        # finger infos
        self.end_finger_names = [
            "j2n6s300_link_finger_tip_1",
            "j2n6s300_link_finger_tip_2",
            "j2n6s300_link_finger_tip_3",
        ]
        self.contact_dict = self.get_contact_pts_and_normals()
        self._joint_lower = 0.2
        self._joint_upper = 1.3

    # ---------------------------------------------------------------------------
    # Robot Movement Functions: moving End Effector
    # ---------------------------------------------------------------------------

    def contact_pts(self, mesh):
        # if self.finger_contact_normals is not None:
        #     return self.finger_contact_pts, self.finger_contact_normals

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
        # self.finger_contact_pts = pts[total_mask]
        # self.finger_contact_normals = surface_normals[total_mask]

        return pts[total_mask], surface_normals[total_mask]

    def visualize_contact_pts(self, pos, dirs, name, vis):
        if self.vis is None:
            print("Skipping visualization")
            return
 
        pos = np.array(pos)
        dirs = np.array(dirs)
        vis.view_pcd(
            pos, colors=np.ones_like(pos) * [1.0, 1.0, 1.0], name=f"pts_{name}"
        )
        for pt_idx, normal in enumerate(dirs):
            vis.view_vector(
                pos[pt_idx], normal * 0.05, name=f"normals_{name}/vec_{pt_idx}"
            )

    def get_contact_pts_and_normals(self):
        meshes = []
        finger_links = []
        finger_contact_pts = []  # in finger frame
        finger_pt_normals = []
        for link in self.urdfpy.links:
            if link.name in self.end_finger_names:
                mesh = link.visuals[0].geometry.mesh.meshes[0]
                contact_pts, normals = self.contact_pts(mesh)
                finger_links.append(link)
                meshes.append(mesh)
                finger_contact_pts.append(contact_pts)
                finger_pt_normals.append(normals)

        return {
            "meshes": meshes,
            "finger_links": finger_links,
            "finger_contact_pts": finger_contact_pts,
            "finger_pt_normals": finger_pt_normals,
        }

    def forward_kinematics(self, joint_vals, finger_links):
        joint_dict = dict(zip(self.all_joint_names, joint_vals))
        fk_link = self.urdfpy.link_fk(cfg=joint_dict)

        poses = []
        for link in finger_links:
            finger_pose = fk_link[link]
            poses.append(finger_pose)

        return poses

    def optimize_ee_pose(self, targ_X_BE, init_joint_vals=None):
        def loss_fn(joint_vals):
            joint_dict = dict(zip(self.all_joint_names, joint_vals))
            fk_link = self.urdfpy.link_fk(cfg=joint_dict)
            X_BE = fk_link[self.urdfpy_ee_link]

            return np.linalg.norm(targ_X_BE - X_BE)

        if init_joint_vals is None:
            init_joint_vals = self.get_all_joint_vals()

        res = minimize(
            loss_fn,
            init_joint_vals,
            bounds=self.joint_bounds,
            method="L-BFGS-B",
            options={"disp": True},
        )
        return res.x

    def optimize_joints(
        self, pos, dirs, init_joint_vals=None, joint_mask=None, joint_vals_fixed=None
    ):
        finger_links = self.contact_dict["finger_links"]
        finger_contact_pts = self.contact_dict["finger_contact_pts"]
        finger_pt_normals = self.contact_dict["finger_pt_normals"]

        X_base = self.base_pose

        def loss_fn(joint_vals):
            if joint_mask is not None:
                assert joint_vals_fixed is not None
                
                indices = joint_mask.nonzero()[0]
                joint_vals[indices] = joint_vals_fixed[indices]

            poses = self.forward_kinematics(joint_vals, finger_links)
            assert len(pos) == len(dirs) == len(poses) == len(finger_links)

            total_loss = 0

            for idx in range(len(pos)):
                p = pos[idx].reshape((1, 3))
                d = dirs[idx].reshape((1, 3))
                cont_pts = finger_contact_pts[idx]
                cont_normals = finger_pt_normals[idx]
                X_finger = poses[idx]

                assert np.isclose(np.linalg.norm(d), 1.0, atol=1e-3)
                assert np.isclose(
                    np.linalg.norm(cont_normals, axis=1), 1.0, atol=1e-3
                ).all()

                cont_pts_tfd = utils.transform_pcd(cont_pts, X_base @ X_finger)
                cont_normals_tfd = utils.transform_normal(
                    cont_normals, X_base @ X_finger
                )

                l1 = np.linalg.norm(cont_pts_tfd - p, axis=1)
                l2 = (d * cont_normals_tfd).sum(axis=1) + 1

                losses = np.square(l1) + np.square(l2)
                total_loss += np.min(losses)

            # print(f"loss: {total_loss:.3f}")
            return total_loss

        # initial guess
        if init_joint_vals is None:
            init_joint_vals = self.get_all_joint_vals()
        assert len(init_joint_vals) == len(self.joint_bounds)

        res = minimize(
            loss_fn,
            init_joint_vals,
            bounds=self.joint_bounds,
            method="L-BFGS-B",
            options={"disp": True},
        )

        
        return res.x

    def optimize_joints_with_force_closure(
        self, init_vec, vec_to_pt, vec_bounds, k, margin=0.02, optimize_base=True
    ):
        finger_links = self.contact_dict["finger_links"]
        finger_contact_pts = self.contact_dict["finger_contact_pts"]
        finger_pt_normals = self.contact_dict["finger_pt_normals"]

        _X_base = self.base_pose

        def objective(x):
            # first 2k is the point coordinate, and the remaining k are the force magnitudes
            vecs = x[: 2 * k].reshape((k, 2))
            forces = x[2 * k : 3 * k].reshape((k, 1))
            joint_vals = x[3 * k : 3 * k + len(self.joint_bounds)]

            base_rtp = x[-3:]
            rcos = base_rtp[0] * np.cos(base_rtp[1])
            rsin = base_rtp[0] * np.sin(base_rtp[1])
            rot = R.from_euler("xyz", [0, 0, base_rtp[2]]).as_matrix()
            bp = np.eye(4)
            bp[:3, :3] = rot
            bp[:2, 3] = [rcos, rsin]

            if optimize_base:
                X_base = bp
            else:
                X_base = np.copy(_X_base)

            # forces = np.exp(forces)
            # forces = forces / forces.sum()
            pos = []
            dirs = []
            A = np.ones((6, k))
            for v_idx in range(k):
                pt, normal = vec_to_pt(vecs[v_idx])
                A[:3, v_idx] = -normal
                A[3:6, v_idx] = np.cross(pt, -normal)

                # displacing the pt along the normal direction
                pt = pt + margin * normal

                pos.append(pt)
                dirs.append(normal)

            diff = A @ forces
            loss1 = np.abs(diff).sum()

            poses = self.forward_kinematics(joint_vals, finger_links)
            assert len(pos) == len(dirs) == len(poses) == len(finger_links)

            total_loss = 0

            for idx in range(len(pos)):
                p = pos[idx].reshape((1, 3))
                d = dirs[idx].reshape((1, 3))
                cont_pts = finger_contact_pts[idx]
                cont_normals = finger_pt_normals[idx]
                X_finger = poses[idx]

                assert np.isclose(np.linalg.norm(d), 1.0, atol=1e-3)
                assert np.isclose(
                    np.linalg.norm(cont_normals, axis=1), 1.0, atol=1e-3
                ).all()

                cont_pts_tfd = utils.transform_pcd(cont_pts, X_base @ X_finger)
                cont_normals_tfd = utils.transform_normal(
                    cont_normals, X_base @ X_finger
                )

                l1 = np.linalg.norm(cont_pts_tfd - p, axis=1)
                l2 = (d * cont_normals_tfd).sum(axis=1) + 1

                losses = np.square(l1) + np.square(l2)

                # weight the loss by the force magnitude
                total_loss += np.min(losses) * forces[idx]

            # print(f"loss: {total_loss:.3f}")
            return 10 * total_loss + loss1

        v_lb = np.repeat(vec_bounds[:1], k, axis=0).flatten()
        v_ub = np.repeat(vec_bounds[1:], k, axis=0).flatten()
        # v_lb = np.repeat([[0.0, -np.pi/2]], k, axis=0).flatten()
        # v_ub = np.repeat([[2*np.pi, np.pi/2]], k, axis=0).flatten()
        f_lb = np.zeros(k)
        f_ub = np.ones(k)

        base_lb = np.array([0.3, -np.pi, -np.pi])
        base_ub = np.array([0.7, np.pi, np.pi])

        x_lb = np.concatenate([v_lb, f_lb])
        x_ub = np.concatenate([v_ub, f_ub])
        bounds = list(zip(x_lb, x_ub)) + self.joint_bounds + list(zip(base_lb, base_ub))
        init_joint_vals = self.get_all_joint_vals()
        # assert len(init_joint_vals) == len(self.joint_bounds)

        init_base = np.random.uniform(base_lb, base_ub)
        x0 = np.concatenate([init_vec, np.ones(k) / k, init_joint_vals, init_base])

        # we apply the constraint that f1 + f2 + ... + fk = 1
        # otherwise the result converges to an all-zero solution for forces
        A_const = np.concatenate(
            [np.zeros(2 * k), np.ones(k), np.zeros(len(init_joint_vals)), np.zeros(3)]
        )
        constraint = LinearConstraint(A=A_const, lb=0.999, ub=1.001)

        result = minimize(
            objective,
            x0=x0,
            bounds=bounds,
            constraints=constraint,
            #    method="L-BFGS-B",
            options={"disp": True},
        )

        if result.success:
            x = result.x
            vecs = x[: 2 * k].reshape((k, 2))
            pts, normals = [], []
            for v in vecs:
                pt, normal = vec_to_pt(v)
                pts.append(pt)
                normals.append(normal)

            pts = np.stack(pts)
            normals = np.stack(normals)

            print("Points:")
            print(np.round(pts, decimals=3))

            print("Normals:")
            print(normals)

            print("Force magnitudes:")
            forces = x[2 * k : 3 * k]
            # forces = forces / forces.sum()
            print(forces)
            joint_vals = x[3 * k : -3]

            b = x[-3:]
            # x_base = np.eye(4)
            # x_base[:2, 3] = b[:2]
            # x_base[:3, :3] = R.from_euler('xyz', [0, 0, b[2]]).as_matrix()

            # mesh = trimesh.creation.box([1.0, 1.0, 1.0])
            # visualize_mesh_with_pts(mesh, np.stack(pts), np.stack(normals))
            return pts, normals, joint_vals, b, forces, result.fun
        else:
            print("Optimization failed")
            return None, None, None, None, None, result.fun

    def optimize_finger_joints(self, pos, dirs, init_joint_vals):
        joint_mask = np.zeros(len(self.all_joint_names), dtype=bool)
        joint_mask[: self.n_arm_joints] = True
        joint_mask[self.n_arm_joints :] = False  # finger joints are not fixed
        init_joint_vals = np.array(init_joint_vals)

        return self.optimize_joints(
            pos, dirs, init_joint_vals, joint_mask, joint_vals_fixed=init_joint_vals
        )

    def get_finger_joint_indices(self, finger_idx):
        if finger_idx == 0:
            return [0, 1], self._joint_lower
        elif finger_idx == 1:
            return [2, 3], self._joint_lower
        elif finger_idx == 2:
            return [4, 5], self._joint_lower

    def free_fingers(self, j, forces):
        arm_joints = j[: self.n_arm_joints]
        ee_joints = j[self.n_arm_joints :]
        for idx, f in enumerate(forces):
            if f < 0.005:
                f_joints, f_vals = self.get_finger_joint_indices(idx)
                ee_joints[f_joints] = f_vals
        return np.concatenate([arm_joints, ee_joints])

    # def pre_grasp_1(self, pos, dirs, init_joint_vals, shift_ee = 0.05):
    def pre_grasp_1(self, init_joint_vals, forces, shift_ee=0.07, view=False):
        # pos and dirs are actual contact points on the object mesh
        # in stage 1 the fingers should be opened more broadly and
        # ee should be lifted along its z-axis

        # use initial joint vals to obtain the ee pose at the grasp
        # these joint vals correspond to lifted fingers.
        # we use this pose and bactrack along y axis to get shifted grasp pose

        # get ee pose
        joint_dict = dict(zip(self.all_joint_names, init_joint_vals))
        fk_link = self.urdfpy.link_fk(cfg=joint_dict)

        ee_pose = fk_link[self.urdfpy_ee_link]
        # self.env.viz.view_frames(X_base @ ee_pose, "grasp_ee_pose_1")

        # backtracking along y-axis
        y_axis = ee_pose[:3, 1]
        ee_pose[:3, 3] += (-shift_ee) * y_axis
        # self.env.viz.view_frames(X_base @ ee_pose, "grasp_pose_shifted")

        # get joint vals, while keeping the finger joints fixed
        pre_ee_joint_vals = self.optimize_ee_pose(ee_pose, init_joint_vals)

        # get arm joints and combine with initial finger joints
        pre_joints = np.concatenate(
            [
                pre_ee_joint_vals[: self.n_arm_joints],
                init_joint_vals[self.n_arm_joints :],
            ]
        )

        pre_joints_opened = self.free_fingers(pre_joints, forces)
        # move the arm

        self.set_joints(pre_joints_opened, vis=view, name="pregrasp_1")

    def lift(self, jvals, length=0.1, view=False):
        # get ee pose in world frame

        X_base = self.base_pose
        joint_dict = dict(zip(self.all_joint_names, jvals))
        fk_link = self.urdfpy.link_fk(cfg=joint_dict)

        ee_pose = fk_link[self.urdfpy_ee_link]
        X_WE = X_base @ ee_pose

        step_size = 0.05
        n_steps = int(length / step_size)
        l = step_size

        for _ in range(n_steps):
            # add the length to the z-axis
            target_X_WE = np.copy(X_WE)
            target_X_WE[2, 3] += l

            targ_X_BE = np.linalg.inv(X_base) @ target_X_WE

            # find the joint vals
            joint_vals = self.optimize_ee_pose(targ_X_BE, jvals)
            # reset the finger joints
            joint_vals[self.n_arm_joints :] = jvals[self.n_arm_joints :]

            self.set_joints(joint_vals, vis=view, name="lifted")

            l += step_size

    def close_fingers(self, pos, dirs, forces, jvals, view=False):
        # move the pos along the normal direction for firm grasp
        margin = 0.03
        pos = pos - margin * dirs

        js = self.optimize_finger_joints(pos, dirs, jvals)
        js[: self.n_arm_joints] = jvals[
            : self.n_arm_joints
        ]  # overriding the arm joints
        js = self.free_fingers(js, forces)
        self.set_joints(js, vis=view, name="grasp")
        return js

    def grasp(
        self,
        obj_pose,
        vec_to_pt_shape,
        vec_bounds,
        k=3,
        opt_base=True,
        view=True,
        object_mesh_dict={},
        object_pose_dict={},
    ):
        # generate the grasp pose

        vec_to_pt = lambda v: vec_to_pt_shape(v, obj_pose)

        max_iters = 50
        for _ in range(max_iters):
            
            init_vec = np.random.uniform(
                vec_bounds[0], vec_bounds[1], size=(k, 2)
            ).flatten()
            pts, normals, joint_vals, b, forces, fun = (
                self.optimize_joints_with_force_closure(
                    init_vec,
                    vec_to_pt,
                    vec_bounds,
                    k=k,
                    margin=0.02,
                    optimize_base=opt_base,
                )
            )
            found = fun < 1e-3
            print("Found:", found, "fun:", fun)

            if found:
                is_collision = utils.check_collision(
                    self.urdfpy,
                    self.base_pose,
                    dict(zip(self.all_joint_names, joint_vals)),
                    object_mesh_dict,
                    object_pose_dict,
                )
                if view:
                    self.vis.view_robot(
                        self.urdfpy,
                        self.all_joint_names,
                        joint_vals,
                        "solution1",
                        color=0x00FF00,
                        alpha=0.3,
                        mesh_type="visual",
                    )

                print("Collision detected:", is_collision)
                # input("Continue?")

                if is_collision:
                    print("Collision detected")
                    continue

                

                if opt_base:
                    # reset robot base
                    self.reset_base(b)

                    # self.vis.view_robot(joint_vals, "solution1", color=0x00FF00, alpha=0.3)

                self.pre_grasp_1(joint_vals, forces, view=view)

                # input("grasp?")

                # move gripper to grasp pose with fingers opened
                _jvals = self.free_fingers(joint_vals, forces)
                self.set_joints(_jvals, vis=view, name="pregrasp_2")

                # close the fingers
                _jvals = self.get_all_joint_vals()
                js = self.close_fingers(pts, normals, forces, _jvals, view=view)

                # lift
                _jvals = js
                # _jvals = self.get_all_joint_vals()
                self.lift(_jvals, length=0.2, view=view)

                break

    #########################################################################################
    ################### Robot control and simulation state functions ########################
    #########################################################################################

    def set_joints(self, joint_vals, vis=None, name="") -> None:
        raise NotImplementedError("Set joints not implemented")

    def reset_base(self, b) -> None:
        raise NotImplementedError("Reset base not implemented")
        r, theta, p = b
        y = r * np.sin(theta)
        x = r * np.cos(theta)

        quat = R.from_euler("xyz", [0, 0, p]).as_quat()
        self.pb_client.resetBasePositionAndOrientation(self.ur5, [x, y, 0.0], quat)
        self.base_pose = np.eye(4)
        self.base_pose[:2, 3] = [x, y]
        self.base_pose[:3, :3] = R.from_quat(quat).as_matrix()

    def get_all_joint_vals(self):
        return self.homej
