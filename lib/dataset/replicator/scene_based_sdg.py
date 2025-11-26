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

"""Generate offline synthetic dataset"""

import argparse
import json
import math
import os
import random

import yaml
from isaacsim import SimulationApp

# Default config (will be updated/extended by any other passed config arguments)
config = {
    "launch_config": {
        "renderer": "RaytracedLighting",
        "headless": True,
    },
    "resolution": [640, 480],
    "rt_subframes": 4,
    "num_frames": 100,
    "env_url": "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd",
    "writer": "DreamWriter",
    "writer_config": {
        "output_dir": "sdg_output/panda_test_100",
        "resolution": [640, 480],
    },
    "clear_previous_semantics": True,
    "forklift": {
        "url": "/Isaac/Props/Forklift/forklift.usd",
        "class": "forklift",
    },
    "panda_robot": {
        "url": "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
        "class": "panda_robot",
    },
    "tables": [
        {
            "url": "/Isaac/Props/Mounts/thor_table.usd",
            "class": "table",
        },
        {
            "url": "/Isaac/Props/PackingTable/props/SM_HeavyDutyPackingTable_C02_01/SM_HeavyDutyPackingTable_C02_01.usd",
            "class": "table",
        },
        {
            "url": "/Isaac/Environments/Simple_Room/Props/table_low.usd",
            "class": "table",
        },
        {
            "url": "/Isaac/Environments/Hospital/Props/SM_SideTable_02a.usd",
            "class": "table",
        },
        {
            "url": "/Isaac/Environments/Office/Props/SM_TableA.usd",
            "class": "table",
        },
        {
            "url": "/Isaac/Environments/Office/Props/SM_TableB.usd",
            "class": "table",
        },
        {
            "url": "/Isaac/Environments/Office/Props/SM_TableC.usd",
            "class": "table",
        },
        {
            "url": "/Isaac/Environments/Office/Props/SM_TableD.usd",
            "class": "table",
        },
        {
            "url": "/Isaac/Environments/Office/Props/SM_TableD2.usd",
            "class": "table",
        },
    ],
    "cone": {
        "url": "/Isaac/Environments/Simple_Warehouse/Props/S_TrafficCone.usd",
        "class": "traffic_cone",
    },
    "pallet": {
        "url": "/Isaac/Environments/Simple_Warehouse/Props/SM_PaletteA_01.usd",
        "class": "pallet",
    },
    "cardbox": {
        "url": "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_04.usd",
        "class": "cardbox",
    },
    "distractors": [
        {
            "url": "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_04.usd",
            "class": "cardbox",
        },
        {
            "url": "/Isaac/Environments/Simple_Warehouse/Props/S_TrafficCone.usd",
            "class": "traffic_cone",
        },
        {
            "url": "/Isaac/Props/YCB/Axis_Aligned/009_gelatin_box.usd",
            "class": "009_gelatin_box",
        },
        {
            "url": "/Isaac/Props/YCB/Axis_Aligned/003_cracker_box.usd",
            "class": "003_cracker_box",
        },
        {
            "url": "/Isaac/Props/YCB/Axis_Aligned/004_sugar_box.usd",
            "class": "004_sugar_box",
        },
        {
            "url": "/Isaac/Props/YCB/Axis_Aligned/010_potted_meat_can.usd",
            "class": "010_potted_meat_can",
        },
        {
            "url": "/Isaac/Props/YCB/Axis_Aligned/035_power_drill.usd",
            "class": "035_power_drill",
        },
        {
            "url": "/Isaac/Props/YCB/Axis_Aligned/011_banana.usd",
            "class": "011_banana",
        },
        {
            "url": "/Isaac/Props/YCB/Axis_Aligned/006_mustard_bottle.usd",
            "class": "006_mustard_bottle",
        },
        {
            "url": "/Isaac/Props/YCB/Axis_Aligned/002_master_chef_can.usd",
            "class": "002_master_chef_can",
        },
        {
            "url": "/Isaac/Props/YCB/Axis_Aligned/005_tomato_soup_can.usd",
            "class": "005_tomato_soup_can",
        },
        {
            "url": "/Isaac/Props/YCB/Axis_Aligned/019_pitcher_base.usd",
            "class": "019_pitcher_base",
        },
        {
            "url": "/Isaac/Props/YCB/Axis_Aligned/007_tuna_fish_can.usd",
            "class": "007_tuna_fish_can",
        },
        {
            "url": "/Isaac/Props/YCB/Axis_Aligned/008_pudding_box.usd",
            "class": "008_pudding_box",
        }
    ],
    "close_app_after_run": True,
}

import carb

# Check if there are any config files (yaml or json) are passed as arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=False, help="Include specific config parameters (json or yaml))")
args, unknown = parser.parse_known_args()
args_config = {}
if args.config and os.path.isfile(args.config):
    print("File exist")
    with open(args.config, "r") as f:
        if args.config.endswith(".json"):
            args_config = json.load(f)
        elif args.config.endswith(".yaml"):
            args_config = yaml.safe_load(f)
        else:
            carb.log_warn(f"File {args.config} is not json or yaml, will use default config")
else:
    carb.log_warn(f"File {args.config} does not exist, will use default config")

# If there are specific writer parameters in the input config file make sure they are not mixed with the default ones
if "writer_config" in args_config:
    config["writer_config"].clear()

# Update the default config dictionay with any new parameters or values from the config file
config.update(args_config)

# Create the simulation app with the given launch_config
simulation_app = SimulationApp(launch_config=config["launch_config"])

import carb.settings

# Late import of runtime modules (the SimulationApp needs to be created before loading the modules)
import omni.replicator.core as rep
import omni.usd

# Custom util functions for the example
import scene_based_sdg_utils
from isaacsim.core.api import World
from isaacsim.core.utils import prims
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.stage import get_current_stage, open_stage
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, UsdGeom

from isaacsim.core.prims import Articulation
import numpy as np

# Get server path
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not get nucleus server path, closing application..")
    simulation_app.close()

# Open the given environment in a new stage
print(f"[scene_based_sdg] Loading Stage {config['env_url']}")
if not open_stage(assets_root_path + config["env_url"]):
    carb.log_error(f"Could not open stage{config['env_url']}, closing application..")
    simulation_app.close()

# Disable capture on play (data generation will be triggered manually)
rep.orchestrator.set_capture_on_play(False)

# Set DLSS to Quality mode (2) for best SDG results , options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

# Clear any previous semantic data in the loaded stage
if config["clear_previous_semantics"]:
    stage = get_current_stage()
    scene_based_sdg_utils.remove_previous_semantics(stage)

# Spawn a panda robot at a random pose
robot_usd = assets_root_path + config["panda_robot"]["url"]

rig_prim = prims.create_prim(
    prim_path="/World/RobotRig",
    prim_type="Xform",
    position=(0, 0, 0)
)

# create a common tabletop
tabletop_height = 0.7
tabletop_plane = rep.create.plane(
    position=(0, 0, tabletop_height),
    scale=1.5,
    visible=False,
    name="TabletopPlane",
)

# spawn the tables
table_container_path = "/World/RobotRig/TableContainer"
prims.create_prim(table_container_path, "Xform")

table_variations = scene_based_sdg_utils.spawn_table_variations(
    parent_path=table_container_path,
    table_configs=config["tables"],
    assets_root_path=assets_root_path,
    target_height=tabletop_height
)

panda_prim = prims.create_prim(
    prim_path="/World/RobotRig/Panda",
    usd_path=robot_usd,
    position=(0, 0, tabletop_height + 0.01),
    semantic_label="panda_robot"
)
# create some rough collision geometry for the robot
robot_path = str(panda_prim.GetPath())

link0_collision = rep.create.cylinder(
    parent=f"{robot_path}/panda_link0",
    position=(0, 0, 0.13), scale=(0.3, 0.3, 0.3),
    visible=False, name="Link0Collision"
)
link2_collision = rep.create.cylinder(
    parent=f"{robot_path}/panda_link2",
    position=(0, -0.1, 0),
    rotation=(90, 0, 0),
    scale=(0.25, 0.25, 0.3),
    visible=False, name="Link2Collision"
)
link4_collision = rep.create.cylinder(
    parent=f"{robot_path}/panda_link4",
    position=(0, 0, 0), scale=(0.2, 0.2, 0.25),
    visible=False, name="Link4Collision"
)
link5_collision = rep.create.cylinder(
    parent=f"{robot_path}/panda_link5",
    position=(0, 0, -0.1), scale=(0.2, 0.2, 0.4),
    visible=False, name="Link5Collision"
)
hand_collision = rep.create.sphere(
    parent=f"{robot_path}/panda_hand",
    position=(0, 0, 0.1), scale=0.25,
    visible=False, name="HandCollision"
)
robot_collision_prims = [
    link0_collision, link2_collision,
    link4_collision, link5_collision, hand_collision
]
robot_cam_container = prims.create_prim(
    prim_path="/World/RobotRig/RobotCamContainer",
    prim_type="Xform",
    position=(0, 0, 0)
)
robot_cam = rep.create.camera(
    parent="/World/RobotRig/RobotCamContainer",
    name="RobotCam"
)

# look-at target for the robot camera
robot_look_target_prim = rep.create.xform(
    position=(0, 0, 1.0),
    visible=False,
    parent="/World/RobotRig",
    name="RobotLookTarget",
)

rig_path = str(rig_prim.GetPath())
rig_group = rep.create.group([rig_path])

# flying distractors
flying_distractor_path = "/World/RobotRig/Distractors"
distractor_container = prims.create_prim(
    prim_path=flying_distractor_path,
    prim_type="Xform",
)
table_distractor_path = "/World/RobotRig/TableDistractors"
table_distractor_container = prims.create_prim(
    prim_path=table_distractor_path,
    prim_type="Xform",
)

# Register randomization graphs
scene_based_sdg_utils.register_flying_distractors(
    distractor_grp_path=flying_distractor_path,
    asset_root_path=assets_root_path,
    asset_list=config["distractors"],
    robot_collision_prims=robot_collision_prims,
    volume_min=(-1.5, -1.5, 0.5),
    volume_max=(1.5, 1.5, 2.5)
)
scene_based_sdg_utils.register_table_scatter(
    parent_path=table_distractor_path,
    robot_base_collision_prim=link0_collision,
    surface_plane=tabletop_plane,
    asset_list=config["distractors"],
    assets_root_path=assets_root_path,
)
scene_based_sdg_utils.register_lights_placement(rig_prim)

# Create render products for the custom cameras and attach them to the writer
resolution = config.get("resolution", (512, 512))
robot_rp = rep.create.render_product(robot_cam, resolution, name="RobotView")
# Disable the render products until SDG to improve perf by avoiding unnecessary rendering
rps = [robot_rp]
for rp in rps:
    rp.hydra_texture.set_updates_enabled(False)

# If output directory is relative, set it relative to the current working directory
if not os.path.isabs(config["writer_config"]["output_dir"]):
    config["writer_config"]["output_dir"] = os.path.join(os.getcwd(), config["writer_config"]["output_dir"])
print(f"[scene_based_sdg] Output directory={config['writer_config']['output_dir']}")

# Make sure the writer type is in the registry
from dream_writer import DreamWriter

rep.WriterRegistry.register(DreamWriter)

writer_type = config.get("writer", "BasicWriter")
if writer_type not in rep.WriterRegistry.get_writers():
    carb.log_error(f"Writer type {writer_type} not found in the registry, closing application..")
    simulation_app.close()

# Get the writer from the registry and initialize it with the given config parameters
writer = rep.WriterRegistry.get(writer_type)
writer_kwargs = config["writer_config"]
print(f"[scene_based_sdg] Initializing {writer_type} with: {writer_kwargs}")
writer.initialize(
    **writer_kwargs,
    semantic_classes=["panda_robot"] + [
        config["distractors"][i]["class"] for i in range(len(config["distractors"]))
    ]
)

# Attach writer to the render products
writer.attach(rps)

# Setup the randomizations to be triggered every frame
with rep.trigger.on_frame():
    rep.randomizer.randomize_lights()

    # Randomize the panda base pose
    with rig_group:
        rep.modify.pose(
            position=rep.distribution.uniform((-20, -2, 0), (-1, 3, 0)),
            rotation=rep.distribution.uniform((0, 0, 0), (0, 0, 360))
        )

    # robot camera
    with robot_look_target_prim:
        rep.modify.pose(
            position=rep.distribution.uniform((-0.4, -0.4, 0.4), (0.4, 0.4, 1.2))
        )
    # set robot cam's rotation
    with robot_cam:
        rep.modify.pose(look_at=robot_look_target_prim)

    # flying and table distractors
    rep.randomizer.randomize_distractors()
    rep.randomizer.randomize_table_items()

# Initialize the Robot
world = World(physics_dt=1.0 / 90.0, stage_units_in_meters=1.0)
world.reset()

panda_robot = Articulation(str(panda_prim.GetPrimPath()))

# Cache limits
dof_limits = panda_robot.get_dof_limits()
lower_limits = dof_limits[0, :, 0]
upper_limits = dof_limits[0, :, 1]

# Increase subframes if materials are not loaded on time, or ghosting artifacts appear on moving objects,
# see: https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/subframes_examples.html
rt_subframes = config.get("rt_subframes", -1)

# Enable the render products for SDG
for rp in rps:
    rp.hydra_texture.set_updates_enabled(True)

# Start the SDG
num_frames = config.get("num_frames", 0)
print(f"[scene_based_sdg] Running SDG for {num_frames} frames")
for i in range(num_frames):
    print(f"[scene_based_sdg] \t Capturing frame {i}")

    # toggle one table
    active_idx = random.randint(0, len(table_variations) - 1)
    for idx, table_prim in enumerate(table_variations):
        imageable = UsdGeom.Imageable(table_prim)
        if idx == active_idx:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()

    # randomize camera
    scene_based_sdg_utils.place_camera_in_shell(
        cam_container_prim=robot_cam_container,
        min_dist=1.5,
        max_dist=3.5,
        min_height=0.8,
        max_height=2.5
    )
    # randomize robot joints
    joint_positions = scene_based_sdg_utils.randomize_robot_joints(
        panda_robot, lower_limits, upper_limits
    )

    joint_map = {}
    for j in range(7):
        joint_map[f"panda_joint{j + 1}"] = joint_positions[j]

    joint_map["panda_hand_joint"] = 0.0
    joint_map["panda_finger_joint1"] = joint_positions[7]
    joint_map["panda_finger_joint2"] = joint_positions[8]

    writer.joint_positions = joint_map

    # Trigger any on_frame registered randomizers and the writers (delta_time=0.0 to avoid advancing the timeline)
    rep.orchestrator.step(delta_time=0.0, rt_subframes=rt_subframes)

# Wait for the data to be written to disk
rep.orchestrator.wait_until_complete()

# Cleanup writer and render products
writer.detach()
for rp in rps:
    rp.destroy()

# Check if the application should keep running after the data generation (debug purposes)
close_app_after_run = config.get("close_app_after_run", True)
if config["launch_config"]["headless"]:
    if not close_app_after_run:
        print(
            "[scene_based_sdg] 'close_app_after_run' is ignored when running headless. The application will be closed."
        )
elif not close_app_after_run:
    print("[scene_based_sdg] The application will not be closed after the run. Make sure to close it manually.")
    while simulation_app.is_running():
        simulation_app.update()
simulation_app.close()
