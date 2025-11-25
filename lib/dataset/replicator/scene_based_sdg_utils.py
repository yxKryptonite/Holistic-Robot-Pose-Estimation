# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import random

import numpy as np
import omni.replicator.core as rep
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils import prims
from isaacsim.core.utils.bounds import compute_combined_aabb, compute_obb, create_bbox_cache, get_obb_corners
from isaacsim.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
from isaacsim.core.utils.semantics import remove_labels
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from isaacsim.core.prims import Articulation


# Add colliders to Gprim and Mesh descendants of the root prim
def add_colliders(root_prim, approx_type="convexHull"):
    # Iterate descendant prims (including root) and add colliders to mesh or primitive types
    for desc_prim in Usd.PrimRange(root_prim):
        if desc_prim.IsA(UsdGeom.Mesh) or desc_prim.IsA(UsdGeom.Gprim):
            # Physics
            if not desc_prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_api = UsdPhysics.CollisionAPI.Apply(desc_prim)
            else:
                collision_api = UsdPhysics.CollisionAPI(desc_prim)
            collision_api.CreateCollisionEnabledAttr(True)
        # Add mesh specific collision properties only to mesh types
        if desc_prim.IsA(UsdGeom.Mesh):
            # Add mesh collision properties to the mesh (e.g. collider aproximation type)
            if not desc_prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(desc_prim)
            else:
                mesh_collision_api = UsdPhysics.MeshCollisionAPI(desc_prim)
            mesh_collision_api.CreateApproximationAttr().Set(approx_type)


# Clear any previous semantic data in the stage
def remove_previous_semantics(stage, recursive: bool = False):
    prims = stage.Traverse()
    for prim in prims:
        remove_labels(prim, include_descendants=recursive)


# Run a simulation
def simulate_falling_objects(forklift_prim, assets_root_path, config, max_sim_steps=250, num_boxes=8):
    # Create the isaac sim world to run any physics simulations
    world = World(physics_dt=1.0 / 90.0, stage_units_in_meters=1.0)

    # Set a random relative offset to the pallet using the forklift transform as a base frame
    forklift_tf = omni.usd.get_world_transform_matrix(forklift_prim)
    pallet_offset_tf = Gf.Matrix4d().SetTranslate(Gf.Vec3d(random.uniform(-1, 1), random.uniform(-4, -3.6), 0))
    pallet_pos = (pallet_offset_tf * forklift_tf).ExtractTranslation()

    # Spawn a pallet prim at a random offset from the forklift
    pallet_prim = prims.create_prim(
        prim_path=f"/World/SimulatedPallet",
        position=pallet_pos,
        orientation=euler_angles_to_quat([0, 0, random.uniform(0, math.pi)]),
        usd_path=assets_root_path + config["pallet"]["url"],
        semantic_label=config["pallet"]["class"],
    )

    # Wrap the pallet as simulation ready with a simplified collider
    add_colliders(pallet_prim, approx_type="boundingCube")
    pallet_rigid_prim = SingleRigidPrim(prim_path=str(pallet_prim.GetPrimPath()))
    pallet_rigid_prim.enable_rigid_body_physics()

    # Use the height of the pallet as a spawn base for the boxes
    bb_cache = create_bbox_cache()
    spawn_height = bb_cache.ComputeLocalBound(pallet_prim).GetRange().GetSize()[2] * 1.1

    # Keep track of the last box to stop the simulation early once it stops moving
    last_box = None
    # Spawn boxes falling on the pallet
    for i in range(num_boxes):
        # Spawn the carbox prim by creating a new Xform prim and adding the USD reference to it
        box_prim = prims.create_prim(
            prim_path=f"/World/SimulatedCardbox_{i}",
            position=pallet_pos + Gf.Vec3d(random.uniform(-0.2, 0.2), random.uniform(-0.2, 0.2), spawn_height),
            orientation=euler_angles_to_quat([0, 0, random.uniform(0, math.pi)]),
            usd_path=assets_root_path + config["cardbox"]["url"],
            semantic_label=config["cardbox"]["class"],
        )

        # Get the next spawn height for the box
        spawn_height += bb_cache.ComputeLocalBound(box_prim).GetRange().GetSize()[2] * 1.1

        # Wrap the prim as simulation ready with a simplified collider
        add_colliders(box_prim, approx_type="boundingCube")
        box_rigid_prim = SingleRigidPrim(prim_path=str(box_prim.GetPrimPath()))
        box_rigid_prim.enable_rigid_body_physics()

        # Cache the rigid prim
        last_box = box_rigid_prim

    # Reset the world to handle the physics of the newly created rigid prims
    world.reset()

    # Simulate the world for the given number of steps or until the highest box stops moving
    for i in range(max_sim_steps):
        world.step(render=False)
        if last_box and np.linalg.norm(last_box.get_linear_velocity()) < 0.001:
            print(f"[scene_based_sdg] Simulation finished at step {i}..")
            break


# Register the boxes and materials randomizer graph
def register_scatter_boxes(pallet_prim, assets_root_path, config):
    # Calculate the bounds of the prim to create a scatter plane of its size
    bb_cache = create_bbox_cache()
    bbox3d_gf = bb_cache.ComputeLocalBound(pallet_prim)
    prim_tf_gf = omni.usd.get_world_transform_matrix(pallet_prim)

    # Calculate the bounds of the prim
    bbox3d_gf.Transform(prim_tf_gf)
    range_size = bbox3d_gf.GetRange().GetSize()

    # Get the quaterion of the prim in xyzw format from usd
    prim_quat_gf = prim_tf_gf.ExtractRotation().GetQuaternion()
    prim_quat_xyzw = (prim_quat_gf.GetReal(), *prim_quat_gf.GetImaginary())

    # Create a plane on the pallet to scatter the boxes on
    plane_scale = (range_size[0] * 0.8, range_size[1] * 0.8, 1)
    plane_pos_gf = prim_tf_gf.ExtractTranslation() + Gf.Vec3d(0, 0, range_size[2])
    plane_rot_euler_deg = quat_to_euler_angles(np.array(prim_quat_xyzw), degrees=True)
    scatter_plane = rep.create.plane(
        scale=plane_scale, position=plane_pos_gf, rotation=plane_rot_euler_deg, visible=False
    )

    cardbox_mats = [
        f"{assets_root_path}/Isaac/Environments/Simple_Warehouse/Materials/MI_PaperNotes_01.mdl",
        f"{assets_root_path}/Isaac/Environments/Simple_Warehouse/Materials/MI_CardBoxB_05.mdl",
    ]

    def scatter_boxes():
        cardboxes = rep.create.from_usd(
            assets_root_path + config["cardbox"]["url"], semantics=[("class", config["cardbox"]["class"])], count=5
        )
        with cardboxes:
            rep.randomizer.scatter_2d(scatter_plane, check_for_collisions=True)
            rep.randomizer.materials(cardbox_mats)
        return cardboxes.node

    rep.randomizer.register(scatter_boxes)


# Register the place cones randomizer graph
def register_cone_placement(forklift_prim, assets_root_path, config):
    # Get the bottom corners of the oriented bounding box (OBB) of the forklift
    bb_cache = create_bbox_cache()
    centroid, axes, half_extent = compute_obb(bb_cache, forklift_prim.GetPrimPath())
    larger_xy_extent = (half_extent[0] * 1.3, half_extent[1] * 1.3, half_extent[2])
    obb_corners = get_obb_corners(centroid, axes, larger_xy_extent)
    bottom_corners = [
        obb_corners[0].tolist(),
        obb_corners[2].tolist(),
        obb_corners[4].tolist(),
        obb_corners[6].tolist(),
    ]

    # Orient the cone using the OBB (Oriented Bounding Box)
    obb_quat = Gf.Matrix3d(axes).ExtractRotation().GetQuaternion()
    obb_quat_xyzw = (obb_quat.GetReal(), *obb_quat.GetImaginary())
    obb_euler = quat_to_euler_angles(np.array(obb_quat_xyzw), degrees=True)

    def place_cones():
        cones = rep.create.from_usd(
            assets_root_path + config["cone"]["url"], semantics=[("class", config["cone"]["class"])]
        )
        with cones:
            rep.modify.pose(position=rep.distribution.sequence(bottom_corners), rotation_z=obb_euler[2])
        return cones.node

    rep.randomizer.register(place_cones)


# Register light randomization graph
def register_lights_placement(robot_prim):
    bb_cache = create_bbox_cache()
    combined_range_arr = compute_combined_aabb(bb_cache, [robot_prim.GetPrimPath()])
    pos_min = (combined_range_arr[0], combined_range_arr[1], 6)
    pos_max = (combined_range_arr[3], combined_range_arr[4], 7)

    def randomize_lights():
        lights = rep.create.light(
            light_type="Sphere",
            color=rep.distribution.uniform((0.2, 0.1, 0.1), (0.9, 0.8, 0.8)),
            intensity=rep.distribution.uniform(2000, 4000),
            position=rep.distribution.uniform(pos_min, pos_max),
            scale=rep.distribution.uniform(1, 4),
            count=3,
        )
        return lights.node

    rep.randomizer.register(randomize_lights)


def place_prim_on_floor(prim):
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    bounds = bbox_cache.ComputeLocalBound(prim).GetRange()
    min_z = bounds.GetMin()[2]
    height_offset = abs(min_z)
    current_pos = prim.GetAttribute("xformOp:translate").Get()
    if current_pos:
        new_pos = Gf.Vec3d(current_pos[0], current_pos[1], height_offset)
        prim.GetAttribute("xformOp:translate").Set(new_pos)
    return height_offset


def spawn_table_variations(parent_path, table_configs, assets_root_path, target_height):
    """
    Spawns all tables from the config, uniformly scales them to match the target_height,
    and sets them to invisible by default.
    """
    table_prims = []
    for i, config in enumerate(table_configs):
        prim_path = f"{parent_path}/Table_{i}"

        prim = prims.create_prim(
            prim_path=prim_path,
            usd_path=assets_root_path + config["url"],
            semantic_label=config["class"],
        )

        # compute the height
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
        )
        bounds = bbox_cache.ComputeLocalBound(prim).GetRange()
        min_z = bounds.GetMin()[2]
        max_z = bounds.GetMax()[2]
        original_height = max_z - min_z

        # scale
        assert original_height > 0, f"Table at {prim_path} has zero height!"
        scale_ratio = target_height / original_height

        prim.GetAttribute("xformOp:scale").Set(
            Gf.Vec3f(scale_ratio, scale_ratio, scale_ratio)
        )
        offset_z = -(min_z * scale_ratio)
        prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(0, 0, offset_z))

        # disable physics
        for p in Usd.PrimRange(prim):
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                phys_api = UsdPhysics.RigidBodyAPI(p)
                phys_api.CreateRigidBodyEnabledAttr().Set(False)

        #  set invisible
        UsdGeom.Imageable(prim).MakeInvisible()
        table_prims.append(prim)

    return table_prims


def place_camera_in_shell(cam_container_prim, min_dist, max_dist, min_height, max_height):
    radius = np.random.uniform(min_dist, max_dist)
    theta = np.random.uniform(0, 2 * np.pi)
    z_height = np.random.uniform(min_height, max_height)

    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = z_height

    cam_container_prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(x, y, z))


def randomize_robot_joints(panda_robot, lower_limits, upper_limits):
    random_pos = np.random.uniform(lower_limits, upper_limits)
    world = World(physics_dt=1.0 / 90.0, stage_units_in_meters=1.0)
    world.reset()
    panda_robot.initialize()
    panda_robot.set_joint_positions(random_pos)
    world.step(render=False)
    return random_pos


def register_flying_distractors(
    distractor_grp_path,
    robot_collision_prims,
    asset_root_path,
    asset_list,
    volume_min,
    volume_max
):
    volume_prim = rep.create.cube(
        position=(
            (volume_min[0]+volume_max[0])/2,
            (volume_min[1]+volume_max[1])/2,
            (volume_min[2]+volume_max[2])/2
        ),
        scale=(
            volume_max[0]-volume_min[0],
            volume_max[1]-volume_min[1],
            volume_max[2]-volume_min[2]
        ),
        visible=False,
        name="DistractorVolume",
    )

    distractor_instances = []
    for i, asset in enumerate(asset_list):
        obj = rep.create.from_usd(
            usd=asset_root_path + asset["url"],
            semantics=[("class", asset["class"])],
            count=1,
            parent=distractor_grp_path,
            name=f"Distractor_{i}"
        )
        distractor_instances.append(obj)

    def randomize_distractors():
        distractor_group = rep.create.group(distractor_instances)

        with distractor_group:
            rep.randomizer.scatter_3d(
                volume_prims=volume_prim,
                no_coll_prims=robot_collision_prims,
                volume_excl_prims=robot_collision_prims,
                check_for_collisions=True,
            )
            rep.modify.pose(
                rotation=rep.distribution.uniform((0, 0, 0), (360, 360, 360)),
                scale=rep.distribution.uniform(0.8, 1.2)
            )
            rep.modify.visibility(rep.distribution.choice([True, False]))

        return distractor_group.node

    rep.randomizer.register(randomize_distractors)


def register_table_scatter(parent_path, robot_base_collision_prim, surface_plane, asset_list, assets_root_path):
    scatter_instances = []
    for i, asset in enumerate(asset_list):
        obj = rep.create.from_usd(
            usd=assets_root_path + asset["url"],
            semantics=[("class", asset["class"])],
            count=1,
            parent=parent_path,
            name=f"TableDist_{i}",
        )
        scatter_instances.append(obj)

    def randomize_table_items():
        group = rep.create.group(scatter_instances)

        with group:
            rep.randomizer.scatter_2d(
                surface_prims=surface_plane,
                no_coll_prims=[robot_base_collision_prim],
                check_for_collisions=True,
            )
            rep.modify.pose(
                rotation=rep.distribution.uniform((0, 0, 0), (0, 0, 360)),
                scale=rep.distribution.uniform(0.8, 1.2)
            )
            rep.modify.visibility(rep.distribution.choice([True, False]))

        return group.node

    rep.randomizer.register(randomize_table_items)
