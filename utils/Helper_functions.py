import json
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_intrinsic_parameters_from_json(json_file_path):
    # Load the JSON data from the file
    with open(json_file_path, "r") as json_file:
        json_data = json.load(json_file)

    # Access the intrinsic parameters from the loaded JSON data
    intrinsic_matrix = json_data["IntrinsicMatrix"]
    focal_length = json_data["FocalLength"]
    principal_point = json_data["PrincipalPoint"]
    radial_distortion = json_data["RadialDistortion"]
    tangential_distortion = json_data["TangentialDistortion"]

    return {
        "IntrinsicMatrix": intrinsic_matrix,
        "FocalLength": focal_length,
        "PrincipalPoint": principal_point,
        "RadialDistortion": radial_distortion,
        "TangentialDistortion": tangential_distortion,
    }


def draw_detections_and_keypoints(
    frame,
    predictions,
    visibility_threshold=0.7,
    point_radius=3,
    point_color=(0, 0, 255),
    box_color=(255, 0, 0),
    label_color=(0, 255, 0),
    score_threshold=0.7,
):
    # Extract boxes, labels, scores, and keypoints
    boxes = predictions["boxes"].cpu().detach().numpy()
    labels = predictions["labels"].cpu().detach().numpy()
    scores = predictions["scores"].cpu().detach().numpy()
    keypoints = predictions["keypoints"].cpu().detach().numpy()
    drawn_keypoints = []

    # Loop through each detected instance
    for box, label, score, kpts in zip(boxes, labels, scores, keypoints):
        score_value = float(score)

        # Draw bounding box and label if score is above threshold
        if score_value >= score_threshold:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(
                frame,
                f"{label} {score_value:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                label_color,
                1,
            )

            # Draw keypoints with visibility above threshold
            for idx, (x, y, visibility) in enumerate(kpts):
                if visibility > visibility_threshold:
                    cv2.circle(
                        frame, (int(x), int(y)), point_radius, point_color, thickness=-1
                    )
                    cv2.putText(
                        frame,
                        str(idx),
                        (int(x), int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )
                    drawn_keypoints.append((x, y))
            for pair in POSE_PAIRS:
                part_from = pair[0]
                part_to = pair[1]

                if (
                    kpts[part_from, 2] > visibility_threshold
                    and kpts[part_to, 2] > visibility_threshold
                ):
                    cv2.line(
                        frame,
                        (
                            int(drawn_keypoints[part_from][0]),
                            int(drawn_keypoints[part_from][1]),
                        ),
                        (
                            int(drawn_keypoints[part_to][0]),
                            int(drawn_keypoints[part_to][1]),
                        ),
                        (0, 255, 0),
                        2,
                    )

    return frame, drawn_keypoints


def get_keypoint_depths(drawn_keypoints, depth_map):
    depths = []
    for x, y in drawn_keypoints:
        depth = depth_map[int(y)][int(x)]
        depths.append(depth)
    return depths


def combine_keypoints_with_depth(drawn_keypoints, depth_map):
    keypoints_with_depth = []
    for x, y in drawn_keypoints:
        depth = depth_map[int(y)][int(x)]
        keypoints_with_depth.append((x, y, depth))
    return keypoints_with_depth


POSE_PAIRS = [
    (0, 1),  # Nose to Left Eye
    (0, 2),  # Nose to Right Eye)
    (1, 3),  # Left Eye to Left Ear
    (2, 4),  # Right Eye to Right Ear
    (5, 7),  # Left Shoulder to Left Elbow
    (6, 8),  # Right Shoulder to Right Elbow
    (7, 9),  # Left Elbow to Left Wrist
    (8, 10),  # Right Elbow to Right Wrist
    (5, 6),  # Left Shoulder to Right Shoulder
    (5, 11),  # Left Shoulder to Left Hip
    (6, 12),  # Right Shoulder to Right Hip
    (11, 12),  # Left Hip to Right Hip
    (11, 13),  # Left Hip to Left Knee
    (12, 14),  # Right Hip to Right Knee
    (13, 15),  # Left Knee to Left Ankle
    (14, 16),  # Right Knee to Right Ankle
]


def plot_3d_points_with_lines(ax, points_3d, pose_pairs=POSE_PAIRS):
    # Clear the current plot
    ax.cla()

    # Extract X, Y, Z coordinates from the list of 3D points
    X = [point[0] for point in points_3d]
    Y = [point[1] for point in points_3d]
    Z = [point[2] for point in points_3d]

    # Plot the 3D points
    ax.scatter(X, Y, Z, c="b", marker="o")

    # Draw lines between the points based on pose_pairs
    for pair in pose_pairs:
        start, end = pair
        ax.plot([X[start], X[end]], [Y[start], Y[end]], [Z[start], Z[end]], c="r")

    plt.pause(0.01)  # Pause for a short duration before next update


def convert_to_3d_camera_coords(keypoints_2d_depth, intrinsic_params):
    # Extracting the intrinsic parameters from the provided dictionary
    fx = intrinsic_params["IntrinsicMatrix"][0][0]
    fy = intrinsic_params["IntrinsicMatrix"][1][1]
    cx = intrinsic_params["IntrinsicMatrix"][0][2]
    cy = intrinsic_params["IntrinsicMatrix"][1][2]

    keypoints_3d = []

    for u, v, d in keypoints_2d_depth:
        X = (u - cx) * d / fx
        Y = (v - cy) * d / fy
        Z = d
        keypoints_3d.append((X, Y, Z))

    return keypoints_3d
