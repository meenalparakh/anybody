import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
import numpy as np
from scipy.spatial.transform import Rotation as R
import random
import trimesh

# from pyngrok import ngrok
import os

from .path_utils import get_tmp_mesh_storage
import matplotlib.colors as mc

class VizServer:
    def __init__(self, port_vis=6000) -> None:
        zmq_url = f"tcp://127.0.0.1:{port_vis}"
        self.mc_vis = meshcat.Visualizer(zmq_url=zmq_url)

        self.clear()

    def clear(self):
        self.mc_vis["scene"].delete()
        self.mc_vis["meshcat"].delete()
        self.mc_vis["/"].delete()

    def view_cylinder(self, ht, r, center):
        self.mc_vis["scene/cylinder"].set_object(
            g.Cylinder(ht, r), g.MeshLambertMaterial(color=0x22DD22, opacity=0.3)
        )

        X_pose = np.eye(4)
        X_pose[:3, :3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
        X_pose[:3, 3] = center
        self.mc_vis["scene/cylinder"].set_transform(X_pose)

    def view_cylinder_pose(self, name, ht, r, pose, color):
        cylinder_z = self.mc_vis[f"{name}"].set_object(
            g.Cylinder(ht, r), g.MeshLambertMaterial(color=color, opacity=0.7)
        )
        self.mc_vis[f"{name}"].set_transform(pose)

    def view_axis(self, name, axis_ht, axis_rad, pose, color):
        cone_ht, cone_rad = axis_ht * 0.2, axis_rad * 3.0
        cylinder_z = self.mc_vis[f"{name}/cylinder"].set_object(
            g.Cylinder(axis_ht, axis_rad),
            g.MeshLambertMaterial(color=color, opacity=0.7),
        )
        cone = self.mc_vis[f"{name}/cone"].set_object(
            g.Sphere(cone_rad), g.MeshLambertMaterial(color=color, opacity=0.7)
        )
        X_CyCo = np.eye(4)
        X_CyCo[:3, 3] = [0, axis_ht / 2.0, 0.0]
        X_CyCo[:3, :3] = R.from_euler("xyz", [-np.pi / 2, 0, 0]).as_matrix()

        self.mc_vis[f"{name}/cylinder"].set_transform(pose)
        self.mc_vis[f"{name}/cone"].set_transform(pose @ X_CyCo)

    def view_trimesh(self, name, mesh, pose, color, opacity=0.7):
        import trimesh

        tmp_obj_path = (get_tmp_mesh_storage() / "tmp_obj.obj")
        
        trimesh.exchange.export.export_mesh(mesh, tmp_obj_path)
        self.view_mesh(name, tmp_obj_path, pose, color, opacity)

    def view_mesh(self, name, obj_path, pose, color, opacity=0.7):
        if color is None:
            color = random.randrange(0, 2**24)

            # Converting that number from base-10
            # (decimal) to base-16 (hexadecimal)
            color = hex(color)
        self.mc_vis[f"scene/{name}"].set_object(
            g.ObjMeshGeometry.from_file(obj_path),
            g.MeshLambertMaterial(color=color, opacity=opacity),
        )
        self.mc_vis[f"scene/{name}"].set_transform(pose)

    def view_pcd(self, pts, colors=None, name="scene", size=0.001):
        # colors are in [0., 1.]
        if colors is None:
            colors = pts
        # self.mc_vis["scene"].delete()
        self.mc_vis["scene/" + name].set_object(
            g.PointCloud(pts.T, color=colors.T, size=size)
        )

    def view_vector(self, pos, vector, name="positioned_vector", color=0x0000FF):
        # axis_length = 0.25
        axis_length = np.linalg.norm(vector)
        radius = 0.005

        z_axis = vector / np.linalg.norm(vector)
        tmp_y_axis = [0.0, 0.0, 1]
        if np.linalg.norm(z_axis - tmp_y_axis) < 1e-2:
            tmp_y_axis = [1.0, 0.0, 0.0]
        x_axis = np.cross(z_axis, tmp_y_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)

        X = np.eye(4)
        X[:3, :3] = np.stack([x_axis, y_axis, z_axis]).T
        X[:3, 3] = pos

        X_FZ = np.eye(4)
        X_FZ[:3, :3] = R.from_euler("xyz", [np.pi / 2, 0, 0]).as_matrix()
        X_FZ[:3, 3] = [0, 0, axis_length / 2]
        self.view_axis(f"{name}/z", axis_length, radius, X @ X_FZ, color)

        base = self.mc_vis[f"{name}/base"].set_object(
            g.Sphere(radius * 2.0), g.MeshLambertMaterial(color=0x000000, opacity=0.7)
        )
        X_CyCo = np.eye(4)
        X_CyCo[:3, 3] = pos
        self.mc_vis[f"{name}/base"].set_transform(X_CyCo)

    def view_frames_pointed(self, pose, name="frame_pose"):
        X_WF = pose
        axis_length = 0.25
        radius = 0.005

        X_FZ = np.eye(4)
        X_FZ[:3, :3] = R.from_euler("xyz", [np.pi / 2, 0, 0]).as_matrix()
        X_FZ[:3, 3] = [0, 0, axis_length / 2]
        self.view_axis(f"{name}/z", axis_length, radius, X_WF @ X_FZ, 0x0000FF)

        X_FY = np.eye(4)
        X_FY[:3, :3] = R.from_euler("xyz", [0, 0, 0]).as_matrix()
        X_FY[:3, 3] = [0, axis_length / 2, 0]
        self.view_axis(f"{name}/y", axis_length, radius, X_WF @ X_FY, 0x00FF00)

        X_FX = np.eye(4)
        X_FX[:3, :3] = R.from_euler("xyz", [0, 0, -np.pi / 2]).as_matrix()
        X_FX[:3, 3] = [axis_length / 2, 0, 0]
        self.view_axis(f"{name}/x", axis_length, radius, X_WF @ X_FX, 0xFF0000)

    def view_frames(self, pose, name="frame_pose"):
        X_WF = pose
        axis_length = 0.25
        radius = 0.005

        X_FZ = np.eye(4)
        X_FZ[:3, :3] = R.from_euler("xyz", [np.pi / 2, 0, 0]).as_matrix()
        X_FZ[:3, 3] = [0, 0, axis_length / 2]
        self.view_cylinder_pose(f"{name}/z", axis_length, radius, X_WF @ X_FZ, 0x0000FF)

        X_FY = np.eye(4)
        X_FY[:3, :3] = R.from_euler("xyz", [0, 0, 0]).as_matrix()
        X_FY[:3, 3] = [0, axis_length / 2, 0]
        self.view_cylinder_pose(f"{name}/y", axis_length, radius, X_WF @ X_FY, 0x00FF00)

        X_FX = np.eye(4)
        X_FX[:3, :3] = R.from_euler("xyz", [0, 0, -np.pi / 2]).as_matrix()
        X_FX[:3, 3] = [axis_length / 2, 0, 0]
        self.view_cylinder_pose(f"{name}/x", axis_length, radius, X_WF @ X_FX, 0xFF0000)

    def view_position(self, position, name="point"):
        if len(position) == 2:
            position = [*position, 0.0]
        X = np.eye(4)
        X[:3, 3] = position
        self.view_frames(X, name=name)

    def view_robot(
        self,
        urdf_inst,
        joint_names,
        joint_vals,
        name,
        color,
        alpha=0.3,
        mesh_type="visual",
        joint_dict=None,
        base_pose=np.eye(4),
    ):
        from anybody.utils.path_utils import get_tmp_mesh_storage

        tmp_storage = get_tmp_mesh_storage()
        tmp_storage.mkdir(exist_ok=True)

        if joint_dict is None:
            joint_dict = dict(zip(joint_names, joint_vals))

        if mesh_type == "collision":
            fk = urdf_inst.collision_trimesh_fk(cfg=joint_dict)
        else:
            fk = urdf_inst.visual_trimesh_fk(cfg=joint_dict)

        to_delete = []
        for idx, (mesh, pose) in enumerate(fk.items()):
            fname = str(tmp_storage / f"mesh_{idx}.obj")
            # mesh to tmp obj
            trimesh.exchange.export.export_mesh(mesh, fname, include_normals=True)
            self.view_mesh(
                f"{name}/{idx}", fname, base_pose @ pose, color, opacity=alpha
            )
            to_delete.append(fname)

        for fname in to_delete:
            os.remove(fname)

    def view_ground(self, name, zval=0.0):
        self.mc_vis["scene/" + name].set_object(
            g.Box([1.0, 1.0, 0.001]), g.MeshLambertMaterial(color=0xAAAAAA, opacity=1.0)
        )
        X = np.eye(4)
        X[2, 3] = zval
        self.mc_vis["scene/" + name].set_transform(X)

    # def view_meshes(
    #     self,
    #     meshes, poses, name, color, alpha=0.3,
    # ):
    #     for idx, (mesh, pose) in enumerate(zip(meshes, poses)):
    #         self.view_trimesh(f"{name}/{idx}", mesh, pose, color, opacity=alpha)

    def view_link_geometry(self, link_info_dict, name="links"):
        for link_name, link_info in link_info_dict.items():
            geom_type = link_info[0]
            if geom_type == "box":
                self.mc_vis[f"{name}/{link_name}"].set_object(
                    g.Box(link_info[1]["dims"]),
                    g.MeshLambertMaterial(color=0xAAAAAA, opacity=1.0),
                )
            elif geom_type == "sphere":
                self.mc_vis[f"{name}/{link_name}"].set_object(
                    g.Sphere(link_info[1]["radius"]),
                    g.MeshLambertMaterial(color=0xAAAAAA, opacity=1.0),
                )
            elif geom_type == "cylinder":
                self.mc_vis[f"{name}/{link_name}"].set_object(
                    g.Cylinder(link_info[1]["height"], link_info[1]["radius"]),
                    g.MeshLambertMaterial(color=0xAAAAAA, opacity=1.0),
                )

    def view_spheres(self, centers, radii, name, color, alpha=0.3):
        for idx in range(len(centers)):
            center = centers[idx]
            radius = radii[idx]

            base = self.mc_vis[f"{name}/{idx}"].set_object(
                g.Sphere(radius), g.MeshLambertMaterial(color=color, opacity=alpha)
            )
            X = np.eye(4)
            X[:3, 3] = center
            self.mc_vis[f"{name}/{idx}"].set_transform(X)

    def visualize_collision_spheres(
        self,
        urdfpy_robot,
        joint_dict,
        collision_spheres,
        name="collision",
        color=0x0000FF,
    ):
        from anybody.utils.utils import transform_pcd

        link_fk = urdfpy_robot.link_fk(cfg=joint_dict)
        for link in link_fk:
            link_name = link.name
            if not link_name in collision_spheres:
                continue

            pose = link_fk[link]
            link_sphere_centers = [v[0] for v in collision_spheres[link_name]]
            link_sphere_radii = [v[1] for v in collision_spheres[link_name]]
            center_pts = np.stack(link_sphere_centers)
            transformed_centers = transform_pcd(center_pts, pose)
            print(pose, link_name)

            self.view_spheres(
                transformed_centers, link_sphere_radii, f"{name}/{link_name}", color
            )

    def view_fitted_box(self, mesh, mesh_origin, box_dim, box_transform, name="mesh"):
        # view mesh
        # self.view_trimesh(name, mesh, mesh_origin, 0x0000FF)

        # view box
        self.mc_vis[f"scene/fitted_box/{name}"].set_object(
            g.Box(box_dim), g.MeshLambertMaterial(color=0xAAAAAA, opacity=0.2)
        )
        self.mc_vis[f"scene/fitted_box/{name}"].set_transform(mesh_origin @ box_transform)


    def view_robot_fitted_isaac(self, urdf_inst, joint_dict, joint_names, joint_vals, isaac_link_info):
        from anybody.utils.path_utils import get_tmp_mesh_storage

        name = "robot"

        base_pose = np.eye(4)
        # mesh_color = 0x00FF00

        tmp_storage = get_tmp_mesh_storage()
        tmp_storage.mkdir(exist_ok=True)

        if joint_dict is None:
            joint_dict = dict(zip(joint_names, joint_vals))

        fk = urdf_inst.link_fk(cfg=joint_dict)
        
        import seaborn as sns

        link_colors = sns.color_palette("Spectral", len(urdf_inst.links))

        for l_idx, (link, link_pose) in enumerate(fk.items()):
            
            col = mc.rgb2hex(link_colors[l_idx])[1:]
            col = int(col, 16)
            print(link.name, col)
            self.view_trimesh(f"{name}/{link.name}", link.collision_mesh, base_pose @ link_pose, col)
            
            link_info = isaac_link_info[link.name]
            obj_type = link_info["obj_type"]
            if obj_type == "box":
                box_dims = link_info['shape_params']['dims']
                box_origin = link_info['origin']
                self.view_fitted_box(
                    link.collision_mesh, base_pose @ link_pose, box_dims, box_origin, name=f"{name}/{link.name}"
                )
            if obj_type == "cylinder":
                cylinder_origin = link_info['origin']
                shape_params = link_info['shape_params']
                r, ht = shape_params['radius'], shape_params['height']
                cylindrical_mesh = trimesh.creation.cylinder(radius=r, height=ht, transform=cylinder_origin)
                
                self.view_trimesh(f"fitted_cylinder/{name}/{link.name}", cylindrical_mesh, base_pose @ link_pose, 0x777777)
                                

    def view_robot_fitted(self, urdf_inst, joint_dict, joint_names, joint_vals, fit_shape="box"):
        # view robot
        from anybody.utils.path_utils import get_tmp_mesh_storage
        from anybody.utils.collision_utils import get_box_fit, get_cylinder_fit, get_best_fit

        name = "robot"

        base_pose = np.eye(4)
        # mesh_color = 0x00FF00

        tmp_storage = get_tmp_mesh_storage()
        tmp_storage.mkdir(exist_ok=True)

        if joint_dict is None:
            joint_dict = dict(zip(joint_names, joint_vals))

        # fk = urdf_inst.collision_trimesh_fk(cfg=joint_dict)
        fk = urdf_inst.visual_trimesh_fk(cfg=joint_dict)
        
        import seaborn as sns

        link_colors = sns.color_palette("coolwarm", len(fk))

        # to_delete = []
        for idx, (mesh, pose) in enumerate(fk.items()):
            
            col = mc.rgb2hex(link_colors[idx])[1:]
            col = int(col, 16)
            
            self.view_trimesh(f"{name}/{idx}", mesh, base_pose @ pose, col)

            # fname = str(tmp_storage / f"mesh_{idx}.obj")
            if fit_shape == 'box':
                box_origin, box_dim = get_box_fit(mesh)
                self.view_fitted_box(
                    mesh, base_pose @ pose, box_dim, box_origin, name=f"{name}/{idx}"
                )
            elif fit_shape == 'cylinder':
                cylinder_origin, (r, ht) = get_cylinder_fit(mesh)
                cylindrical_mesh = trimesh.creation.cylinder(radius=r, height=ht, transform=cylinder_origin)
                
                self.view_trimesh(f"fitted_cylinder/{name}/{idx}", cylindrical_mesh, base_pose @ pose, 0x777777)
                
            elif fit_shape == 'best':
                shape, fit_pose, fit_dims = get_best_fit(mesh)
                if shape == 'box':
                    self.view_fitted_box(
                        mesh, base_pose @ pose, fit_dims, fit_pose, name=f"{name}/{idx}"
                    )
                elif shape == 'cylinder':
                    r, ht = fit_dims
                    cylindrical_mesh = trimesh.creation.cylinder(radius=r, height=ht, transform=fit_pose)
                    self.view_trimesh(f"fitted_cylinder/{name}/{idx}", cylindrical_mesh, base_pose @ pose, 0x777777)
                
            else:
                raise ValueError("Invalid fit_shape")

    def view_box(self, box_dim, box_transform, name="box", color=0xAAAAAA, alpha=0.3):
        self.mc_vis[f"scene/{name}"].set_object(
            g.Box(box_dim), g.MeshLambertMaterial(color=color, opacity=alpha)
        )
        self.mc_vis[f"scene/{name}"].set_transform(box_transform)


    def view_voxels(self, voxels, voxel_size, workspace_bounds, name="voxels"):
        # the workspace bounds is divided into voxels
        # the voxels are of size voxel_size
        # workspace_bounds = [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        
        # get the grid size
        grid_dims = voxels.shape

        # get the min bounds
        min_bounds = workspace_bounds[:, 0]
        max_bounds = workspace_bounds[:, 1]
        
        # for each voxel, create a box of size voxel_size
        # compute the center of each voxel
        
        voxel_x_centers = np.arange(grid_dims[0]) * voxel_size + min_bounds[0] + voxel_size / 2
        voxel_y_centers = np.arange(grid_dims[1]) * voxel_size + min_bounds[1] + voxel_size / 2
        voxel_z_centers = np.arange(grid_dims[2]) * voxel_size + min_bounds[2] + voxel_size / 2
        
        
        # get non-zero indices
        voxel_indices = np.where(voxels)
        
        # get the centers of the non-zero voxels
        voxel_centers = np.stack([voxel_x_centers[voxel_indices[0]], voxel_y_centers[voxel_indices[1]], voxel_z_centers[voxel_indices[2]]], axis=-1)
                # view as pcd
        self.view_pcd(voxel_centers, name=name, size=voxel_size)
        

    def close(self):
        self.mc_vis.close()
