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

        if "link" in name.lower():
            links[name] = str(path)
        elif "joint" in name.lower() or is_joint:
            joints[name] = str(path)

    return links, joints


class DreamWriter(rep.Writer):
    def __init__(self, output_dir, resolution):
        self.output_dir = output_dir
        self.resolution = resolution
        self.frame_id = 0

        self.backend = rep.BackendDispatch({"paths": {"out_dir": self.output_dir}})

        # Register necessary annotators
        self.annotators = [
            rep.AnnotatorRegistry.get_annotator("rgb"),
            rep.AnnotatorRegistry.get_annotator("camera_params"),
            rep.AnnotatorRegistry.get_annotator(
                "bounding_box_2d_tight",
                init_params={"semanticTypes": ["class"]}
            ),
            rep.AnnotatorRegistry.get_annotator(
                "bounding_box_3d",
                init_params={"semanticFilter": "class:*"}
            ),
        ]

        # cache robot links and joints
        self.robot_links = None
        self.robot_joints = None

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
        self.backend.write_image(f"{filename}.png", data["rgb"])

        # 4. camera_data
        camera_data = self._extract_camera_data(cam_params)

        # 5. objects
        gf_view = Gf.Matrix4d(*cam_params["cameraViewTransform"])
        gf_proj = Gf.Matrix4d(*cam_params["cameraProjection"])
        gf_view_proj = gf_view * gf_proj

        objects = self._extract_objects(data, gf_view_proj)

        # 6. Build JSON Structure
        json_data = {
            "camera_data": camera_data,
            "objects": objects,
        }

        # 7. Write JSON
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

    def _extract_camera_data(self, cam_params):
        world2cam = Gf.Matrix4d(*cam_params["cameraViewTransform"])
        cam2world = world2cam.GetInverse()

        pos = cam2world.ExtractTranslation()
        pos = [pos[0], pos[1], pos[2]]
        rot = cam2world.ExtractRotationQuat()
        quat = [
            rot.GetReal(),
            rot.GetImaginary()[0],
            rot.GetImaginary()[1],
            rot.GetImaginary()[2],
        ]

        return {"location_worldframe": pos, "quaternion_xyzw_worldframe": quat}

    def _extract_objects(self, data, gf_view_proj: Gf.Matrix4d):
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
            if semantic_id not in semantic_id_to_bbox_2d:
                continue  # skip non-visible objects

            semantic_class = bbox_2d_info["idToLabels"][str(semantic_id)]["class"]

            x_min = float(bbox_3d_data[idx]["x_min"])
            y_min = float(bbox_3d_data[idx]["y_min"])
            z_min = float(bbox_3d_data[idx]["z_min"])
            x_max = float(bbox_3d_data[idx]["x_max"])
            y_max = float(bbox_3d_data[idx]["y_max"])
            z_max = float(bbox_3d_data[idx]["z_max"])

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

            # transform to world frame
            raw_transform = bbox_3d_data[idx]["transform"].flatten().tolist()
            gf_transform = Gf.Matrix4d(*raw_transform)
            corners_world = [gf_transform.Transform(c) for c in corners_local]

            # project cuboid corners to 2D
            projected_cuboid = self._project_points(corners_world, gf_view_proj)

            # cuboid centroid
            centroid_world = sum(corners_world, Gf.Vec3d(0, 0, 0)) / len(corners_world)
            projected_centroid = self._project_points([centroid_world], gf_view_proj)[0]

            # keypoints
            keypoints = []
            # Special handling for Robot
            if "panda_robot" in semantic_class.lower():
                keypoints = self._get_robot_keypoints(gf_view_proj)
            else:
                keypoints.append(
                    {
                        "name": "Root",
                        "location": [
                            centroid_world[0],
                            centroid_world[1],
                            centroid_world[2],
                        ],
                        "projected_location": projected_centroid,
                    }
                )

            # pose
            trans = gf_transform.ExtractTranslation()
            rot = gf_transform.ExtractRotation().GetQuaternion()
            quat = [
                rot.GetReal(),
                rot.GetImaginary()[0],
                rot.GetImaginary()[1],
                rot.GetImaginary()[2],
            ]

            obj_entry = {
                "class": semantic_class,
                "visibility": 1,
                "location": [trans[0], trans[1], trans[2]],
                "quaternion_xyzw": quat,
                "pose_transform": bbox_3d_data[idx]["transform"].tolist(),
                "cuboid_centroid": [
                    centroid_world[0],
                    centroid_world[1],
                    centroid_world[2],
                ],
                "projected_cuboid_centroid": projected_centroid,
                "bounding_box": semantic_id_to_bbox_2d[semantic_id],
                "cuboid": [[c[0], c[1], c[2]] for c in corners_world],
                "projected_cuboid": projected_cuboid,
                "keypoints": keypoints,
            }
            objects.append(obj_entry)

        return objects

    def _get_robot_keypoints(self, gf_view_proj: Gf.Matrix4d):
        stage = omni.usd.get_context().get_stage()

        if self.robot_joints is None and self.robot_links is None:
            self.robot_links, self.robot_joints = get_robot_links_and_joints(
                stage.GetPrimAtPath("/World/RobotRig/Panda")
            )

        keypoints = []

        def _add_kp(name, path):
            prim = stage.GetPrimAtPath(path)

            transform = omni.usd.get_world_transform_matrix(prim)
            pos_3d = transform.ExtractTranslation()

            pos_2d = self._project_points([pos_3d], gf_view_proj)[0]

            keypoints.append(
                {
                    "name": name,
                    "location": [pos_3d[0], pos_3d[1], pos_3d[2]],
                    "projected_location": pos_2d,
                }
            )

        for joint_name, joint_path in self.robot_joints.items():
            _add_kp(joint_name, joint_path)
        for link_name, link_path in self.robot_links.items():
            _add_kp(link_name, link_path)

        return keypoints

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
