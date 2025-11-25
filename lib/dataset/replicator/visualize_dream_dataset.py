import json
import cv2
import os
import argparse
import glob
import numpy as np


def draw_projected_box(img, points, color=(0, 255, 0), thickness=2):
    """
    Draws a wireframe box from 8 projected corner points.
    Points are expected in binary order:
    0: min_x, min_y, min_z
    1: max_x, min_y, min_z
    2: min_x, max_y, min_z
    3: max_x, max_y, min_z
    4: min_x, min_y, max_z
    ...
    """
    if len(points) != 8:
        return

    # Convert to integer tuples
    pts = [tuple(map(int, p)) for p in points]

    # Define connections based on the generation order in DreamWriter
    # Bottom Face (z_min)
    cv2.line(img, pts[0], pts[1], color, thickness)
    cv2.line(img, pts[1], pts[3], color, thickness)
    cv2.line(img, pts[3], pts[2], color, thickness)
    cv2.line(img, pts[2], pts[0], color, thickness)

    # Top Face (z_max)
    cv2.line(img, pts[4], pts[5], color, thickness)
    cv2.line(img, pts[5], pts[7], color, thickness)
    cv2.line(img, pts[7], pts[6], color, thickness)
    cv2.line(img, pts[6], pts[4], color, thickness)

    # Vertical Pillars
    cv2.line(img, pts[0], pts[4], color, thickness)
    cv2.line(img, pts[1], pts[5], color, thickness)
    cv2.line(img, pts[2], pts[6], color, thickness)
    cv2.line(img, pts[3], pts[7], color, thickness)


def draw_keypoints(img, keypoints, color=(0, 255, 255)):
    """Draws small circles for keypoints."""
    for kp in keypoints:
        loc = kp.get("projected_location")
        if loc:
            center = (int(loc[0]), int(loc[1]))
            cv2.circle(img, center, 3, color, -1)  # Filled circle
            # Optional: Draw Text Name
            cv2.putText(img, kp['name'], center, cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)


def main():
    parser = argparse.ArgumentParser(description="Visualize DREAM Dataset Annotations")
    parser.add_argument(
        "--dir", type=str, default="_out_16825", help="Path to dataset folder"
    )
    parser.add_argument(
        "--save", action="store_true", help="Save output images instead of showing them"
    )
    args = parser.parse_args()

    # Find all JSON files
    json_files = sorted(glob.glob(os.path.join(args.dir, "*.json")))

    # Filter out settings files
    data_files = [f for f in json_files if not f.endswith("settings.json")]

    if not data_files:
        print(f"No data found in {args.dir}")
        return

    print(f"Found {len(data_files)} frames. Processing...")

    # Output directory for visualizations
    vis_dir = os.path.join(args.dir, "_visualizations")
    if args.save:
        os.makedirs(vis_dir, exist_ok=True)
        print(f"Saving visualizations to {vis_dir}")

    for json_path in data_files:
        # 1. Load JSON
        with open(json_path, "r") as f:
            data = json.load(f)

        # 2. Load Corresponding Image
        # DREAM usually uses .png or .rgb.jpg. We check what exists.
        base_path = os.path.splitext(json_path)[0]
        img_path = f"{base_path}.rgb.jpg"
        if not os.path.exists(img_path):
            img_path = f"{base_path}.png"

        if not os.path.exists(img_path):
            print(f"Image not found for {json_path}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        # 3. Draw Objects
        for obj in data.get("objects", []):
            obj_class = obj.get("class", "unknown")

            # Color logic: Robot = Green, Distractors = Red
            color = (0, 255, 0) if "panda" in obj_class.lower() else (0, 0, 255)

            # A. Draw Projected Cuboid (3D Box)
            if "projected_cuboid" in obj:
                draw_projected_box(img, obj["projected_cuboid"], color)

            # B. Draw 2D Bounding Box
            if "bounding_box" in obj:
                bbox = obj["bounding_box"]
                min_pt = (int(bbox["min"][0]), int(bbox["min"][1]))
                max_pt = (int(bbox["max"][0]), int(bbox["max"][1]))
                cv2.rectangle(img, min_pt, max_pt, (255, 255, 0), 1)  # Cyan for 2D bbox

            # C. Draw Keypoints
            if "keypoints" in obj:
                draw_keypoints(img, obj["keypoints"])

            # D. Draw Label
            centroid = obj.get("projected_cuboid_centroid")
            if centroid:
                cv2.putText(
                    img,
                    obj_class,
                    (int(centroid[0]), int(centroid[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

        # 4. Show or Save
        if args.save:
            filename = os.path.basename(img_path)
            save_path = os.path.join(vis_dir, filename)
            cv2.imwrite(save_path, img)
        else:
            cv2.imshow("Dataset Visualization", img)
            # Press 'q' to quit, space to next
            key = cv2.waitKey(0)
            if key == ord('q'):
                break

    if not args.save:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
