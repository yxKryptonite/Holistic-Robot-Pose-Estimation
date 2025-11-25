import omni.replicator.core as rep
import omni.usd
from pxr import Usd, UsdGeom, Gf, UsdPhysics
import numpy as np
import json
import os
import io


def check_bbox_area(self, bbox_data, size_limit):
    length = abs(bbox_data['x_min'] - bbox_data['x_max'])
    width = abs(bbox_data['y_min'] - bbox_data['y_max'])

    area = length * width
    if area > size_limit:
        return True
    else:
        return False


def get_robot_links_and_joints(prim):
    """
    Returns: (links, joints) where each is a dict of {name: path}
    """
    links = dict()
    joints = dict()
    for child in Usd.PrimRange(prim):
        path = child.GetPath()
        name = child.GetName()
        type_name = child.GetTypeName()

        is_joint = child.IsA(UsdPhysics.Joint) if hasattr(UsdPhysics, "Joint") else "Joint" in type_name

        if "joint" in name.lower() or is_joint:
            joints[name] = str(path)
        elif "link" in name or "finger" in name or "hand" in name:
            links[name] = str(path)

    return links, joints


class DreamWriter(rep.Writer):
    def __init__(self, output_dir: str, resolution: tuple, semantic_classes: list[str]):
        self.output_dir = output_dir
        self.resolution = resolution
        self.frame_id = 0

        self.backend = rep.BackendDispatch({"paths": {"out_dir": self.output_dir}})

        self.semantic_classes = "|".join(semantic_classes)
        # Register necessary annotators
        self.annotators = [
            rep.AnnotatorRegistry.get_annotator("rgb"),
            rep.AnnotatorRegistry.get_annotator("camera_params"),
            rep.AnnotatorRegistry.get_annotator(
                "bounding_box_2d_tight",
                init_params={"semanticFilter": f"class:{self.semantic_classes}"},
            ),
            rep.AnnotatorRegistry.get_annotator(
                "bounding_box_3d",
                init_params={"semanticFilter": f"class:{self.semantic_classes}"},
            ),
        ]

        # cache robot links and joints
        self.robot_links = None
        self.robot_joints = None

        # for _object_settings.json
        self.num_objects_required = len(semantic_classes)
        self.discovered_classes = {}

        # for sim_state
        self.sim_state_camera = None
        self.keypoint_orientations = dict()

        # to be set externally
        self.joint_positions = None

    def write(self, data):
        cam_params = data["camera_params"]
        # --- SAVE ONCE ---
        # 1. Save _camera_settings.json
        if self.frame_id == 0:
            self._write_camera_settings(cam_params)

        # TODO: 2. Save _object_settings.json

        # --- SAVE EVERY FRAME ---
        filename = f"{self.frame_id:06d}"
        # 3. Save RGB Image
        self.backend.write_image(f"{filename}.rgb.jpg", data["rgb"])

        # 4. camera_data
        camera_data = self._extract_camera_data(cam_params)

        # cache sim_state camera
        if self.sim_state_camera is None:
            pos_cm = camera_data["location_worldframe"]
            pos_m = [x / 100.0 for x in pos_cm]  # to m
            self.sim_state_camera = dict(
                name="sim_cam",
                pose=dict(
                    position=pos_m,
                    orientation=camera_data["quaternion_xyzw_worldframe"],
                )
            )

        # 5. objects
        gf_view = Gf.Matrix4d(*cam_params["cameraViewTransform"])
        gf_proj = Gf.Matrix4d(*cam_params["cameraProjection"])
        gf_view_proj = gf_view * gf_proj

        gl_to_cv = Gf.Matrix4d().SetScale(Gf.Vec3d(1, -1, -1))
        gf_world2cam_cv = gf_view * gl_to_cv

        objects = self._extract_objects(data, gf_view_proj, gf_world2cam_cv)

        # 6. sim_state - must be called after objects extraction
        sim_state = self._extract_sim_state(
            data,
            objects[0]
        )

        # 7. Build JSON Structure
        json_data = {
            "camera_data": camera_data,
            "objects": objects,
            "sim_state": sim_state,
        }

        # 8. Write JSON
        buf = io.BytesIO()
        buf.write(json.dumps(json_data, indent=4).encode())
        self.backend.write_blob(f"{filename}.json", buf.getvalue())

        print(f"[DreamWriter] Wrote frame {self.frame_id}")
        self.frame_id += 1

    def _write_camera_settings(self, cam_params):
        proj = cam_params["cameraProjection"].reshape(4, 4)
        w = int(cam_params["renderProductResolution"][0])
        h = int(cam_params["renderProductResolution"][1])

        fx = float(proj[0, 0] * w / 2.0)
        fy = float(proj[1, 1] * h / 2.0)
        cx = float(w / 2.0)
        cy = float(h / 2.0)
        hfov_deg = float(np.degrees(2 * np.arctan(1.0 / proj[0, 0])))

        camera_settings = {
            "id": "",
            "name": "",
            "intrinsic_settings": {
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "s": 0.0,
                "hfov": hfov_deg,
                "resolution": {"width": w, "height": h},
            },
            "captured_image_size": {"width": w, "height": h},
        }

        json_dict = {"camera_settings": [camera_settings]}

        buf = io.BytesIO()
        buf.write(json.dumps(json_dict, indent=4).encode())
        self.backend.write_blob("_camera_settings.json", buf.getvalue())

    def _flush_object_settings(self):
        """Writes the current state of discovered classes to disk."""
        robot_val = []
        robot_key = []
        if "panda_robot" in self.discovered_classes:
            robot_key = ["franka"]
            robot_val = [self.discovered_classes.pop("panda_robot")]
            robot_val[0]["class"] = "franka"
        exported_objects = robot_val + list(self.discovered_classes.values())
        exported_classes = robot_key + list(self.discovered_classes.keys())

        json_data = {
            "exported_object_classes": exported_classes,
            "exported_objects": exported_objects
        }

        buf = io.BytesIO()
        buf.write(json.dumps(json_data, indent=4).encode())
        self.backend.write_blob("_object_settings.json", buf.getvalue())

    def _extract_camera_data(self, cam_params):
        world2cam = Gf.Matrix4d(*cam_params["cameraViewTransform"])
        cam2world = world2cam.GetInverse()

        pos = cam2world.ExtractTranslation()
        pos = [pos[0] * 100.0, pos[1] * 100.0, pos[2] * 100.0]  # to cm
        rot = cam2world.ExtractRotationQuat()
        quat = [
            rot.GetImaginary()[0],
            rot.GetImaginary()[1],
            rot.GetImaginary()[2],
            rot.GetReal(),
        ]

        return {"location_worldframe": pos, "quaternion_xyzw_worldframe": quat}

    def _compute_robot_world_bbox(self, robot_prim_path):
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(robot_prim_path)

        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
        )
        bound = bbox_cache.ComputeWorldBound(prim)
        return bound.GetRange()

    def _extract_objects(self, data, gf_view_proj: Gf.Matrix4d, gf_world2cam_cv: Gf.Matrix4d):
        """
        Note: Visible objects will be included in bounding_box_2d_tight.
        """
        objects = []

        bbox_2d_data = data["bounding_box_2d_tight"]["data"]
        bbox_2d_info = data["bounding_box_2d_tight"]["info"]
        bbox_3d_info = data["bounding_box_3d"]["info"]
        bbox_3d_data = data["bounding_box_3d"]["data"]

        # first process 2D bboxes
        semantic_id_to_bbox_2d = {}
        for idx in range(len(bbox_2d_data)):
            semantic_id = bbox_2d_data[idx]["semanticId"]
            semantic_id_to_bbox_2d[semantic_id] = {
                "min": [float(bbox_2d_data[idx]["x_min"]), float(bbox_2d_data[idx]["y_min"])],
                "max": [float(bbox_2d_data[idx]["x_max"]), float(bbox_2d_data[idx]["y_max"])],
            }

        # then process 3D bboxes, pose, keypoints
        # and fill in the object dict
        for idx in range(len(bbox_3d_data)):
            semantic_id = bbox_3d_data[idx]["semanticId"]
            semantic_class = bbox_3d_info["idToLabels"][str(semantic_id)]["class"]

            # robot must always be included
            is_robot = "panda_robot" in semantic_class

            x_min = float(bbox_3d_data[idx]["x_min"])
            y_min = float(bbox_3d_data[idx]["y_min"])
            z_min = float(bbox_3d_data[idx]["z_min"])
            x_max = float(bbox_3d_data[idx]["x_max"])
            y_max = float(bbox_3d_data[idx]["y_max"])
            z_max = float(bbox_3d_data[idx]["z_max"])

            # transform to world frame
            raw_local2world = bbox_3d_data[idx]["transform"].flatten().tolist()
            gf_local2world = Gf.Matrix4d(*raw_local2world)
            gf_local2cam = gf_local2world * gf_world2cam_cv

            if is_robot:
                robot_range = self._compute_robot_world_bbox("/World/RobotRig/Panda")

                r_min = robot_range.GetMin()
                r_max = robot_range.GetMax()

                corners_world = [
                    Gf.Vec3d(r_min[0], r_min[1], r_min[2]),
                    Gf.Vec3d(r_max[0], r_min[1], r_min[2]),
                    Gf.Vec3d(r_min[0], r_max[1], r_min[2]),
                    Gf.Vec3d(r_max[0], r_max[1], r_min[2]),
                    Gf.Vec3d(r_min[0], r_min[1], r_max[2]),
                    Gf.Vec3d(r_max[0], r_min[1], r_max[2]),
                    Gf.Vec3d(r_min[0], r_max[1], r_max[2]),
                    Gf.Vec3d(r_max[0], r_max[1], r_max[2]),
                ]
            elif semantic_id in semantic_id_to_bbox_2d:
                # define 8 corners in local frame
                corners_local = [
                    Gf.Vec3d(x_min, y_min, z_min),
                    Gf.Vec3d(x_max, y_min, z_min),
                    Gf.Vec3d(x_min, y_max, z_min),
                    Gf.Vec3d(x_max, y_max, z_min),
                    Gf.Vec3d(x_min, y_min, z_max),
                    Gf.Vec3d(x_max, y_min, z_max),
                    Gf.Vec3d(x_min, y_max, z_max),
                    Gf.Vec3d(x_max, y_max, z_max),
                ]

                corners_world = [gf_local2world.Transform(c) for c in corners_local]
            else:
                continue

            corners_cam = [gf_world2cam_cv.Transform(c) for c in corners_world]

            # update discovered classes if needed
            if len(self.discovered_classes) < self.num_objects_required:
                if semantic_class not in self.discovered_classes:
                    # dims in cm
                    width = (x_max - x_min) * 100.0  # to cm
                    height = (y_max - y_min) * 100.0  # to cm
                    depth = (z_max - z_min) * 100.0  # to cm

                    self.discovered_classes[semantic_class] = {
                        "class": semantic_class,
                        "segmentation_class_id": int(semantic_id),
                        "segmentation_instance_id": 0,
                        "fixed_model_transform": [
                            [0, 0, 1, 0],
                            [-1, 0, 0, 0],
                            [0, -1, 0, 0],
                            [0, 0, 0, 1]
                        ],
                        "cuboid_dimensions": [height, depth, width]
                    }
                    self._flush_object_settings()

            # project cuboid corners to 2D
            projected_cuboid = self._project_points(corners_world, gf_view_proj)

            if is_robot:
                # manual bbox 2d for the robot if missing
                x_coords = [pt[0] for pt in projected_cuboid]
                y_coords = [pt[1] for pt in projected_cuboid]
                # clip to image bounds
                width, height = self.resolution
                min_x = max(0, min(x_coords))
                min_y = max(0, min(y_coords))
                max_x = min(width, max(x_coords))
                max_y = min(height, max(y_coords))

                if max_x <= min_x:
                    max_x = min_x + 1
                if max_y <= min_y:
                    max_y = min_y + 1

                bbox_2d_entry = {
                    "min": [float(min_x), float(min_y)],
                    "max": [float(max_x), float(max_y)],
                }
                semantic_id_to_bbox_2d[semantic_id] = bbox_2d_entry

            # cuboid centroid
            centroid_world = sum(corners_world, Gf.Vec3d(0, 0, 0)) / len(corners_world)
            centroid_cam = gf_world2cam_cv.Transform(centroid_world)
            projected_centroid = self._project_points([centroid_world], gf_view_proj)[0]

            # keypoints
            keypoints = []
            # Special handling for Robot
            if "panda_robot" in semantic_class.lower():
                keypoints, self.keypoint_orientations = self._get_robot_keypoints(
                    gf_view_proj, gf_world2cam_cv
                )
            else:
                keypoints.append(
                    {
                        "name": "Root",
                        "location": [
                            centroid_world[0] * 100.0, # to cm
                            centroid_world[1] * 100.0, # to cm
                            centroid_world[2] * 100.0, # to cm
                        ],
                        "projected_location": projected_centroid,
                    }
                )

            # pose
            trans = gf_local2cam.ExtractTranslation()
            rot = gf_local2cam.ExtractRotation().GetQuaternion()
            quat = [
                rot.GetImaginary()[0],
                rot.GetImaginary()[1],
                rot.GetImaginary()[2],
                rot.GetReal(),
            ]

            # convert pose transform translation to cm
            pose_transform_cm = np.array(bbox_3d_data[idx]["transform"])
            pose_transform_cm[3, 0] *= 100.0 # to cm
            pose_transform_cm[3, 1] *= 100.0 # to cm
            pose_transform_cm[3, 2] *= 100.0 # to cm

            obj_entry = {
                "class": semantic_class,
                "visibility": 1,
                "location": [trans[0] * 100.0, trans[1] * 100.0, trans[2] * 100.0], # to cm
                "quaternion_xyzw": quat,
                "pose_transform": pose_transform_cm.tolist(), # to cm
                "cuboid_centroid": [
                    centroid_cam[0] * 100.0, # to cm
                    centroid_cam[1] * 100.0, # to cm
                    centroid_cam[2] * 100.0, # to cm
                ],
                "projected_cuboid_centroid": projected_centroid,
                "bounding_box": semantic_id_to_bbox_2d[semantic_id],
                "cuboid": [[c[0] * 100.0, c[1] * 100.0, c[2] * 100.0] for c in corners_cam], # to cm
                "projected_cuboid": projected_cuboid,
                "keypoints": keypoints,
            }
            if "panda_robot" in semantic_class.lower():
                # insert to front
                objects.insert(0, obj_entry)
            else:
                objects.append(obj_entry)

        return objects

    def _get_robot_keypoints(self, gf_view_proj: Gf.Matrix4d, gf_world2cam_cv: Gf.Matrix4d):
        stage = omni.usd.get_context().get_stage()

        if self.robot_joints is None and self.robot_links is None:
            self.robot_links, self.robot_joints = get_robot_links_and_joints(
                stage.GetPrimAtPath("/World/RobotRig/Panda")
            )

        keypoints = []
        keypoint_orientations = dict()

        def _add_kp(name, path):
            prim = stage.GetPrimAtPath(path)

            gf_local2world = omni.usd.get_world_transform_matrix(prim)
            gf_local2cam = gf_local2world * gf_world2cam_cv
            pos_3d_world = gf_local2world.ExtractTranslation()
            pos_3d = gf_world2cam_cv.Transform(pos_3d_world)

            pos_2d = self._project_points([pos_3d_world], gf_view_proj)[0]

            keypoints.append(
                {
                    "name": name,
                    "location": [pos_3d[0] * 100.0, pos_3d[1] * 100.0, pos_3d[2] * 100.0], # to cm
                    "projected_location": pos_2d,
                }
            )

            # orientation
            quat = gf_local2cam.ExtractRotation().GetQuaternion()
            keypoint_orientations[name] = [
                quat.GetImaginary()[0],
                quat.GetImaginary()[1],
                quat.GetImaginary()[2],
                quat.GetReal(),
            ]

        for joint_name, joint_path in self.robot_joints.items():
            _add_kp(joint_name, joint_path)
        for link_name, link_path in self.robot_links.items():
            _add_kp(link_name, link_path)

        # add link8 to comply with the urdf
        hand_kp = next(kp for kp in keypoints if kp['name'] == 'panda_hand')
        link8_kp = hand_kp.copy()
        link8_kp['name'] = 'panda_link8'
        keypoints.append(link8_kp)

        # there's a rotation offset but ignore for now
        keypoint_orientations['panda_link8'] = keypoint_orientations['panda_hand']

        return keypoints, keypoint_orientations

    def _project_points(self, points_3d: list[Gf.Vec3d], gf_view_proj: Gf.Matrix4d):
        """
            https://learnopengl.com/Getting-started/Coordinate-Systems
        """
        width = self.resolution[0]
        height = self.resolution[1]
        points_2d = []

        for pt in points_3d:
            # clip space
            pt_h = Gf.Vec4d(pt[0], pt[1], pt[2], 1.0)
            pt_clip = pt_h * gf_view_proj

            # NDC space
            x_ndc = pt_clip[0] / pt_clip[3]
            y_ndc = pt_clip[1] / pt_clip[3]

            # screen space
            x_pixel = (x_ndc + 1) * 0.5 * width
            y_pixel = (1 - y_ndc) * 0.5 * height

            points_2d.append([x_pixel, y_pixel])

        return points_2d

    def _extract_sim_state_joints(self, data, robot_name="panda_arm_hand_DR"):
        stage = omni.usd.get_context().get_stage()
        if self.robot_joints is None:
            self.robot_links, self.robot_joints = get_robot_links_and_joints(
                stage.GetPrimAtPath("/World/RobotRig/Panda")
            )

        joints_list = []

        for name in self.robot_joints.keys():
            if "panda_" not in name:
                continue

            joints_list.append({
                "name": f"/{robot_name}/{name}",
                "position": self.joint_positions[name],
                "velocity": 0.0
            })
        return joints_list

    def _extract_sim_state_links(self, robot_obj, robot_name="panda_arm_hand_DR"):
        link_states = []
        for obj in robot_obj["keypoints"]:
            if "panda_" in obj["name"] and "joint" not in obj["name"]:
                link_states.append(
                    dict(
                        name=f"/{robot_name}/{obj['name']}",
                        pose=dict(
                            position=[x / 100.0 for x in obj["location"]],  # cm to m
                            orientation=self.keypoint_orientations[obj["name"]]
                        )
                    )
                )
        return link_states

    def _extract_sim_state(self, data, robot_obj):
        entities = [
            self.sim_state_camera,
            dict(
                name="panda_arm_hand_DR",
                pose=dict(
                    position=[x / 100.0 for x in robot_obj["location"]],  # cm to m
                    orientation=robot_obj["quaternion_xyzw"]
                )
            )
        ]
        link_states = self._extract_sim_state_links(robot_obj)
        joint_states = self._extract_sim_state_joints(data)

        return dict(
            entities=entities,
            links=link_states,
            joints=joint_states
        )
