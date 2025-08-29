from __future__ import annotations

import trimesh
import argparse
import contextlib
import shutil
import isaacsim

from dataclasses import dataclass

import os

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.utils.assets import check_file_path
from isaaclab.utils.dict import print_dict
from pathlib import Path
from isaaclab.sim.schemas import schemas_cfg


@dataclass
class ArgsCli:
    input: str
    output: str
    headless: bool = True
    merge_joints: bool = False
    fix_base: bool = True
    make_instanceable: bool = False
    joint_stiffness: float = 100.0
    joint_damping: float = 1.0
    joint_target_type: str = "position"


def main(args):
    # check valid file path
    tmp_dir = "tmpmodel"
    if args.input.endswith(".gltf"):
        print(f"Converting to URDF first at {tmp_dir}")
        if not os.path.exists(tmp_dir):
            os.mkdir(tmp_dir)
        mesh = trimesh.load(args.input, file_type="gltf", force="mesh")
        trimesh.exchange.urdf.export_urdf(mesh, tmp_dir)
        urdf_path = tmp_dir + "/" + tmp_dir + ".urdf"
    else:
        urdf_path = args.input

    if not os.path.isabs(urdf_path):
        urdf_path = os.path.abspath(urdf_path)
    if not check_file_path(urdf_path):
        raise ValueError(f"Invalid file path: {urdf_path}")
    # create destination path
    dest_path = args.output
    parent_dir = Path(dest_path).parent.absolute()
    parent_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.isabs(dest_path):
        dest_path = os.path.abspath(dest_path)

    # Create Urdf converter config
    urdf_converter_cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=os.path.dirname(dest_path),
        usd_file_name=os.path.basename(dest_path),
        fix_base=args.fix_base,
        merge_fixed_joints=args.merge_joints,
        force_usd_conversion=True,
        make_instanceable=args.make_instanceable,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=args.joint_stiffness,
                damping=args.joint_damping,
            ),
            target_type=args.joint_target_type,
        ),
    )

    # Print info
    # print("-" * 80)
    # print("-" * 80)
    # print(f"Input URDF file: {urdf_path}")
    # print("URDF importer config:")
    # print_dict(urdf_converter_cfg.to_dict(), nesting=0)
    # print("-" * 80)
    # print("-" * 80)

    # Create Urdf converter and import the file
    urdf_converter = UrdfConverter(urdf_converter_cfg)

    # print output
    # print("URDF importer output:")
    # print(f"Generated USD file: {urdf_converter.usd_path}")
    # print("-" * 80)
    # print("-" * 80)

    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    # # Simulate scene (if not headless)
    # if not args.headless:
    #     # Open the stage with USD
    #     stage_utils.open_stage(urdf_converter.usd_path)
    #     # Reinitialize the simulation
    #     app = omni.kit.app.get_app_interface()
    #     # Run simulation
    #     with contextlib.suppress(KeyboardInterrupt):
    #         while True:
    #             # perform step
    #             app.update()


@dataclass
class MeshArgsCli:
    input: str
    output: str
    make_instanceable: bool = True
    collision_approximation: str = "convexDecomposition"
    mass: float = None


def mesh_to_usd_args(args):
    # check valid file path
    mesh_path = args.input
    if not os.path.isabs(mesh_path):
        mesh_path = os.path.abspath(mesh_path)
    if not check_file_path(mesh_path):
        raise ValueError(f"Invalid mesh file path: {mesh_path}")

    # create destination path
    dest_path = args.output
    if not os.path.isabs(dest_path):
        dest_path = os.path.abspath(dest_path)

    print(dest_path)
    print(os.path.dirname(dest_path))
    print(os.path.basename(dest_path))

    # Mass properties
    if args.mass is not None:
        mass_props = schemas_cfg.MassPropertiesCfg(mass=args.mass)
        rigid_props = schemas_cfg.RigidBodyPropertiesCfg()
    else:
        mass_props = None
        rigid_props = None

    # Collision properties
    collision_props = schemas_cfg.CollisionPropertiesCfg(
        collision_enabled=args.collision_approximation != "none"
    )

    # Create Mesh converter config
    mesh_converter_cfg = MeshConverterCfg(
        mass_props=mass_props,
        rigid_props=rigid_props,
        collision_props=collision_props,
        asset_path=mesh_path,
        force_usd_conversion=True,
        usd_dir=os.path.dirname(dest_path),
        usd_file_name=os.path.basename(dest_path),
        make_instanceable=args.make_instanceable,
        collision_approximation=args.collision_approximation,
    )

    # Print info
    print("-" * 80)
    print("-" * 80)
    print(f"Input Mesh file: {mesh_path}")
    print("Mesh importer config:")
    print_dict(mesh_converter_cfg.to_dict(), nesting=0)
    print("-" * 80)
    print("-" * 80)

    # Create Mesh converter and import the file
    mesh_converter = MeshConverter(mesh_converter_cfg)
    # print output
    print("Mesh importer output:")
    print(f"Generated USD file: {mesh_converter.usd_path}")
    print("-" * 80)
    print("-" * 80)
