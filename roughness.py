import numpy as np
from scipy.spatial.distance import cdist
from iri import iri
import pprint as pp
import os
import ast

COLOR_LABELS = {
    0: ("lane", [255, 255, 255]),  # White for lanes
    1: ("shoulder", [0, 0, 255]),  # Blue for shoulders
    2: ("chevrons", [128, 128, 0]),  # Olive for chevrons
    3: ("broken-line", [255, 0, 255]),  # Magenta for broken lines
    4: ("solid-line", [0, 255, 255]),  # Cyan for solid lines
    5: ("arrows", [255, 165, 0]),  # Orange for arrows
    6: ("vegetation", [0, 128, 0]),  # Green for vegetation
    7: ("traffic-sign", [255, 0, 0]),  # Red for traffic signs
    8: ("highway-guardrails", [128, 0, 128]),  # Purple for highway guardrails
    9: ("concrete-barriers", [255, 255, 0]),  # Yellow for concrete barriers
    10: ("light-pole", [0, 0, 0]),  # Black for light poles
    11: ("clutter", [192, 192, 192]),  # Gray for clutter
}
COLOR_VALUES = np.array([color for _, color in COLOR_LABELS.values()])
MAPPED_LABEL_IDX = {
    "lane": 0,
    "shoulder": 1,
    "chevrons": 2,
    "broken-line": 3,
    "solid-line": 4,
    "arrows": 5,
    "vegetation": 6,
    "traffic-sign": 7,
    "highway-guardrails": 8,
    "concrete-barriers": 9,
    "light-pole": 10,
    "clutter": 11,
}


# Normalization functions
def pc_normalize(pc, length=50, breadth=50, height=50):
    """Normalizes the point cloud data using a fixed range method."""
    x, y, z = pc[:, 0], pc[:, 1], pc[:, 2]
    x = x - np.min(x)
    y = y - np.min(y)
    z = z
    return np.round(np.stack((x, y, z), axis=-1), 6)


def normalize_point_cloud_data(data):
    """Normalizes the entire point cloud dataset."""
    coords = data[:, :3]
    coords_normalized = pc_normalize(coords)
    return np.hstack((coords_normalized, data[:, 3:]))


# Determine if normalization is required
def determine_normalization_strategy(points):
    """
    Determines whether to use global or local normalization based on the values in the point cloud.
    If any coordinate exceeds 50, use global normalization, otherwise use local (ground-truth) normalization.
    """
    if np.any(points[:, :3] > 50):  # Check if any x, y, or z value exceeds 50
        print("Detected values greater than 50, using global normalization.")
        return normalize_point_cloud_data  # Global normalization
    else:
        print("Using local (ground-truth) normalization.")
        return lambda data: data  # Local normalization (no change to the data)


# File reading functions
def read_txt_with_rgb(filename):
    """Reads TXT file and returns point cloud data as a NumPy array."""
    data = []
    with open(filename, "r") as f:
        for line in f:
            # Check if the line contains commas to determine the split method
            if "," in line:
                values = list(map(float, line.strip().split(",")))
            else:
                values = list(map(float, line.strip().split()))
            data.append(values)
    return np.array(data)


# Labeling functions
def label_points_by_color(points_rgb, COLOR_VALUES):
    """Labels points based on their RGB values by finding the closest predefined color."""
    distances = cdist(points_rgb, COLOR_VALUES, metric="euclidean")
    return np.argmin(distances, axis=1)


# Extract points based on start and end coordinate
def interpolate_start_end(start_point, end_point, num=500):
    """Interpolates between the start and end points."""
    x_start, y_start, z_start = start_point
    x_end, y_end, z_end = end_point
    x = np.linspace(x_start, x_end, num=num)
    y = np.linspace(y_start, y_end, num=num)
    # z = np.linspace(z_start, z_end, num=num)
    # save_point_cloud(np.stack((x, y), axis=-1), "interpolated_points.txt")
    return np.stack((x, y), axis=-1)


def get_points(points, start_point, end_point, threshold=0.05, batch_size=10000):
    """
    Filters points between the start and end points using normalization on the fly
    to handle large files.

    Args:
        points (np.ndarray): Array of points with shape (N, M), where the first three columns are x, y, z coordinates.
        start_point (tuple): Start point (x, y, z).
        end_point (tuple): End point (x, y, z).
        threshold (float): Maximum distance to consider a point close to the line.
        batch_size (int): Number of points to process in one batch.

    Returns:
        np.ndarray: Filtered points close to the interpolated line.
    """
    interpolated_points = interpolate_start_end(start_point, end_point)
    filtered_points = []

    num_points = points.shape[0]

    for i in range(0, num_points, batch_size):
        # Process points in batches
        batch_points = points[i : i + batch_size]

        # Calculate distances for the batch
        distances = np.linalg.norm(
            batch_points[:, :2][:, None, :] - interpolated_points[None, :, :],
            axis=2,
        )
        # Identify points in the batch with at least one distance below the threshold
        mask = np.any(distances <= threshold, axis=1)

        # Append the filtered points
        filtered_points.append(batch_points[mask])

    # Concatenate results from all batches
    return np.concatenate(filtered_points, axis=0)


# Extract points based on labels
def classify_points(data) -> dict:
    """
    Classifies the points based on the color labels.
    lane - broken line - arrows
    shoulder - solid line
    """

    classified_points = {}
    current_group = []
    current_class_label = None

    # Iterate through the data
    for i, point in enumerate(data):
        # Extract the current point's class label
        color = tuple(point[3:6])
        label_name = None

        # Find the label name corresponding to the color
        for key, (mapped_label, mapped_color) in COLOR_LABELS.items():
            if color == tuple(mapped_color):
                label_name = mapped_label
                break

        # If label_name is not found, skip the point
        if label_name is None:
            continue

        # Check if we are still in the same group
        if label_name == current_class_label:
            current_group.append(point)
        else:
            # If we are switching groups, process the current group
            if current_group:
                current_group = np.array(current_group)

                # Calculate the length of the group
                group_length = np.max(current_group[:, 0]) - np.min(current_group[:, 0])

                # Add the group to the classified points
                if current_class_label in classified_points:
                    classified_points[current_class_label]["points"] = np.vstack(
                        (classified_points[current_class_label]["points"], current_group)
                    )
                    classified_points[current_class_label]["length"] += group_length
                else:
                    classified_points[current_class_label] = {
                        "points": current_group,
                        "length": group_length,
                    }

            # Start a new group
            current_class_label = label_name
            current_group = [point]

    # Process the last group
    if current_group:
        current_group = np.array(current_group)
        group_length = np.max(current_group[:, 0]) - np.min(current_group[:, 0])

        if current_class_label in classified_points:
            classified_points[current_class_label]["points"] = np.vstack(
                (classified_points[current_class_label]["points"], current_group)
            )
            classified_points[current_class_label]["length"] += group_length
        else:
            classified_points[current_class_label] = {
                "points": current_group,
                "length": group_length,
            }

    return classified_points


# Calculation Function
def calculate_iri(data):
    # print(data.shape)
    # distance = cdist(np.array([start_point]), np.array([end_point]), metric="euclidean")
    # print(f"Distance: {distance}")
    # iri_value, _ = iri(data[:, :2], distance[0][0], -1, 0)
    # return iri_value

    data = data[:,[0,2]]
    _,unique_indices = np.unique(data[:,0], return_index=True)
    data_cleaned = data[unique_indices]
    data_cleaned = data_cleaned[np.argsort(data_cleaned[:,0])]
    x_coords = data[:,0]
    # come up with a better strategy for this
    segment_length = np.max(x_coords) - np.min(x_coords)

    iri_value = iri(data_cleaned,segment_length,data[0,0],0)

    return iri_value


# Saving Function
def save_point_cloud(points, output_file):
    """Saves the point cloud data to a file."""
    np.savetxt(
        output_file,
        points,
        delimiter=",",
        fmt="%.6f",
    )
    print(f"Filtered point cloud data saved to {output_file}")


"""
Test Case:
    # start_point, end_point = 
    #   [709627.33624268, 5648709.33099365, 1018.97027588], 
    #   [709677.22576904, 5648693.19976807, 1018.35424805]

    # start_point, end_point = 
    #   [709655.119271, 5648717.076043, 1019.693002], 
    #   [709654.937500, 5648695.00000, 1018.645020]
"""


def main():
    # file_path = input("Enter the file path: ")
    file_path = "combined_00215N_R1R1_18000_20000_section_8.txt"
    if not os.path.isfile(file_path):
        print("Invalid file path.")
        return

    # Read the point cloud data
    data = read_txt_with_rgb(file_path)
    if data is None:
        print(f"Failed to load data from {file_path}")
        return

    start_point = ast.literal_eval(input("Enter the start point: "))
    end_point = ast.literal_eval(input("Enter the end point: "))
    print(f"Start point: {start_point}")
    print(f"End point: {end_point}")
    print(f"Total points: {len(data)}")

    # Filter points within the bounding box
    filtered_points = get_points(data, start_point, end_point, 0.08)
    print(f"Filtered points: {len(filtered_points)}")

    # classification = classify_points(data)

    # Save the filtered point cloud data
    output_file = "filtered_point_cloud.txt"
    save_point_cloud(filtered_points, output_file)

    # Calculate IRI
    classified_points = classify_points(filtered_points)
    for point in classified_points:
        pp.pprint(f"{point}: {classified_points[point]}")
        # iri_value = calculate_iri(classified_points[point])
        # print(f"IRI: {iri_value}")

    # print(f"IRI: {iri_value}")
    # save_point_cloud(classified_points["lane_bl_arr"], "lane_bl_arr.txt")
    # save_point_cloud(classified_points["shoulder_sl"], "shoulder_sl.txt")


if __name__ == "__main__":
    main()