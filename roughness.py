import numpy as np
from scipy.spatial.distance import cdist
from iri import iri
import pprint as pp
import os
import re
import open3d as o3d
from scipy.spatial import Delaunay
from scipy.interpolate import CubicSpline, PchipInterpolator, Akima1DInterpolator
import pandas as pd


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


# Orientation function
def orient_point_cloud(data: np.ndarray, point1, point2):
    coords = data[:, :3]  # Extract x, y, z
    rgb = data[:, 3:] if data.shape[1] > 3 else np.zeros_like(coords)

    p1 = np.array(point1)
    p2 = np.array(point2)

    v_x = p2[:2] - p1[:2]  # Only x and y
    v_x = np.append(v_x, 0)  # Ensure z component is 0
    v_x = v_x / np.linalg.norm(v_x)  # Normalize
    v_y = np.array([-v_x[1], v_x[0], 0])  # 90-degree rotation in XY plane
    v_z = np.array([0, 0, 1])

    # Create the rotation matrix
    R = np.vstack([v_x, v_y, v_z]).T  # Columns are the new basis vectors

    translated_coords = coords - p1
    oriented_coords = np.dot(translated_coords, R)
    oriented_coords[:, 2] = coords[:, 2]  # Keep original Z
    oriented_data = np.hstack([oriented_coords, rgb])

    # Compute transformed points
    p1_transformed = np.array([0, 0, p1[2]])  # Keep original Z
    p2_transformed = np.dot(R.T, (p2 - p1))
    p2_transformed[2] = p2[2]  # Keep original Z for p2

    # Debug output for transformed points
    print("\n[DEBUG] Transformed Start Point:", p1_transformed)
    print("\n[DEBUG] Transformed End Point:", p2_transformed)

    return oriented_data, p1_transformed, p2_transformed


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


def get_points(points, start_point, end_point, threshold=0.025, batch_size=10000):

    start_point = np.array(start_point)
    end_point = np.array(end_point)

    # Vector representation of the line segment
    line_vector = end_point[:2] - start_point[:2]
    line_length_squared = np.dot(line_vector, line_vector)

    filtered_points = []

    num_points = points.shape[0]

    for i in range(0, num_points, batch_size):
        # Process points in batches
        batch_points = points[i : i + batch_size]

        # Calculate the vector from start_point to the batch points
        point_vectors = batch_points[:, :2] - start_point[:2]

        # Project the points onto the line (dot product)
        t = np.dot(point_vectors, line_vector) / line_length_squared

        # Clamp t to the range [0, 1] to ensure projections stay on the line segment
        t = np.clip(t, 0, 1)

        # Compute the closest points on the line segment
        projections = start_point[:2] + t[:, None] * line_vector

        # Calculate distances from the original points to the projections
        distances = np.linalg.norm(batch_points[:, :2] - projections, axis=1)

        # Identify points within the threshold distance
        mask = distances <= threshold

        # Adjust y-coordinate of the filtered points
        batch_points_filtered = batch_points[mask]
        batch_points_filtered[:, 1] = 0  # Set y = 0

        # Append the filtered points
        filtered_points.append(batch_points_filtered)

    # Concatenate results from all batches
    concatenated_points = np.concatenate(filtered_points, axis=0)

    # Remove duplicates
    seen = set()
    unique_points = []
    for row in concatenated_points:
        row_tuple = tuple(row)  # Convert to tuple for set lookup
        if row_tuple not in seen:
            seen.add(row_tuple)
            unique_points.append(row)

    # Convert back to NumPy array
    unique_points = np.array(unique_points)

    # Sort by x-coordinate
    sorted_indices = np.argsort(unique_points[:, 0])  # Sort by first column (x)
    sorted_points = unique_points[sorted_indices]

    return sorted_points

def classify_points(data) -> dict:
    """
    Classifies the points based on the color labels.
    lane - broken line - arrows
    shoulder - solid line
    """

    LANE_GROUP = {"lane", "broken-line", "arrows"}  # Keep these together
    SHOULDER_GROUP = {"shoulder", "solid-line"}  # Keep these together

    classified_points = {}
    current_group = []
    current_class_label = None
    group_counter = 1  # Used to give unique group names

    for i, point in enumerate(data):
        color = tuple(point[3:6])
        label_name = None

        for key, mapped_idx in MAPPED_LABEL_IDX.items():
            mapped_color = COLOR_LABELS[mapped_idx][1]
            if color == tuple(mapped_color):
                label_name = key
                break

        if label_name is None:
            continue

        if label_name in LANE_GROUP:
            normalized_label = "lane"
        elif label_name in SHOULDER_GROUP:
            normalized_label = "shoulder"
        else:
            normalized_label = label_name 

        if (
            (normalized_label == current_class_label)
            or (label_name in LANE_GROUP and current_class_label in LANE_GROUP)
            or (label_name in SHOULDER_GROUP and current_class_label in SHOULDER_GROUP)
        ):
            current_group.append(point)
        else:
            if len(current_group) > 5:  # Only create a new group if it has more than 5 points
                group_name = f"{current_class_label}_{group_counter}"
                classified_points[group_name] = np.array(current_group)
                group_counter += 1

            # Start a new group
            current_class_label = normalized_label
            current_group = [point]

    # Ensure the last group is added only if it has more than 5 points
    if len(current_group) > 5:
        group_name = f"{current_class_label}_{group_counter}"
        classified_points[group_name] = np.array(current_group)

    return classified_points

def get_mesh_points(
    points,
    start_point: np.ndarray,
    end_point: np.ndarray,
    uniform_spacing=0.05,
):
    """
    Return the points at uniform spacing along a line from start_point to end_point,
    ensuring they are interpolated on the mesh surface.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(points[:, 3:6] / 255)

    pcd_xy = np.asarray(pcd.points)[:, :2]
    tri = Delaunay(pcd_xy)  # 2D Delaunay triangulation

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = pcd.points
    mesh.triangles = o3d.utility.Vector3iVector(tri.simplices)
    mesh.compute_vertex_normals()

    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    mesh_vertices = mesh.vertex["positions"].numpy()
    mesh_triangles = tri.simplices  # Triangle indices
    mesh_colors = np.asarray(pcd.colors)  # Store original colors

    los_direction = end_point - start_point
    los_direction /= np.linalg.norm(los_direction)
    distance = np.linalg.norm(end_point - start_point)
    num_steps = int(np.floor(distance / uniform_spacing))

    filtered_points = []

    for i in range(num_steps):
        p = start_point + i * uniform_spacing * los_direction

        # Find which triangle contains this point
        simplex = tri.find_simplex(p[:2])

        if simplex >= 0:

            a_idx, b_idx, c_idx = mesh_triangles[simplex]
            a, b, c = mesh_vertices[a_idx], mesh_vertices[b_idx], mesh_vertices[c_idx]
            color_a, color_b, color_c = mesh_colors[a_idx], mesh_colors[b_idx], mesh_colors[c_idx]

            dist_a = np.linalg.norm(p[:2] - a[:2])
            dist_b = np.linalg.norm(p[:2] - b[:2])
            dist_c = np.linalg.norm(p[:2] - c[:2])

            # Color assignment on the closest vertex of the triangle

            if dist_a <= dist_b and dist_a <= dist_c:
                chosen_color = color_a
            elif dist_b <= dist_a and dist_b <= dist_c:
                chosen_color = color_b
            else:
                chosen_color = color_c
            # Compute Barycentric coordinates
            v0, v1, v2 = b - a, c - a, np.array([p[0] - a[0], p[1] - a[1], 0])
            d00 = np.dot(v0, v0)
            d01 = np.dot(v0, v1)
            d11 = np.dot(v1, v1)
            d20 = np.dot(v2, v0)
            d21 = np.dot(v2, v1)
            denom = d00 * d11 - d01 * d01
            beta = (d11 * d20 - d01 * d21) / denom
            gamma = (d00 * d21 - d01 * d20) / denom
            alpha = 1.0 - beta - gamma

            # Compute the interpolated point on the triangle (Whats this)
            interpolated_point = alpha * a + beta * b + gamma * c
            new_point = [p[0], p[1], interpolated_point[2]]
            filtered_points.append(np.concatenate([new_point, chosen_color * 255]))

    filtered_points = np.array(filtered_points)
    # filtered_points = np.hstack(
    #     (filtered_points, np.full((filtered_points.shape[0], 3), 255))
    # )
    filtered_points[:, 1] = 0

    return filtered_points


def get_interpolated_points(method_name, points):
    """Interpolator Options: CubicSpline, PCHIP, Akima"""
    x = points[:, 0]  # X coordinates
    z = points[:, 2]  # Z coordinates
    if points.shape[1] > 3:
        class_labels = points[:, 3:]
    else:
        class_labels = np.zeros((points.shape[0], 3))

    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    z_sorted = z[sorted_indices]
    class_labels_sorted = class_labels[sorted_indices]

    # Generate new points at every 0.05 interval
    x_new = np.arange(x_sorted.min(), x_sorted.max() + 0.05, 0.05)

    # Apply the selected interpolation method
    if method_name.lower() == "pchip":
        interpolator = PchipInterpolator(x_sorted, z_sorted)
    elif method_name.lower() == "cubicspline":
        interpolator = CubicSpline(x_sorted, z_sorted, bc_type="clamped")
    elif method_name.lower() == "akima":
        interpolator = Akima1DInterpolator(x_sorted, z_sorted)

    z_new = interpolator(x_new)

    # Assign class labels using nearest-neighbor approach
    indices = np.searchsorted(x_sorted, x_new, side="left")
    indices = np.clip(
        indices, 0, len(x_sorted) - 1
    )  # Ensure indices stay within bounds
    class_labels_new = class_labels_sorted[indices]

    # Combine into a single array (x, y=0, z, class_labels)
    interpolated_points = np.column_stack(
        (x_new, np.zeros_like(x_new), z_new, class_labels_new)
    )

    return interpolated_points

# Calculation Function
def calculate_iri(data):
    # print(data.shape)
    # distance = cdist(np.array([start_point]), np.array([end_point]), metric="euclidean")
    # print(f"Distance: {distance}")
    # iri_value, _ = iri(data[:, :2], distance[0][0], -1, 0)
    # return iri_value

    data = data[:, [0, 2]]
    _, unique_indices = np.unique(data[:, 0], return_index=True)
    data_cleaned = data[unique_indices]
    data_cleaned = data_cleaned[np.argsort(data_cleaned[:, 0])]
    x_coords = data[:, 0]
    start = np.min(x_coords)
    end = np.max(x_coords)
    # come up with a better strategy for this
    segment_length = end - start

    iri_value = iri(data_cleaned, segment_length, -1, 0)

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


def print_iri(points, output_file=None):
    if output_file is not None:
        np.savetxt(output_file, points, fmt="%.6f")
    print()
    print("[Total number of filtered points]", len(points))
    print()
    results = []
    classified_points = classify_points(points)
    for label, data in classified_points.items():
        print()
        print(f"##################  {label}  #################")
        num_points = len(data)  # Get the number of rows in the NumPy array
        print("Number of points in the classified dataset: ", num_points)
        iri_value = calculate_iri(data)
        print(f"IRI: {iri_value}")
        print()

        results.append((label,iri_value, num_points))
        ### OPTIONAL: Save the classified points to a file
        # print(f"{label}: {num_points} points")
        # np.savetxt(f"DEBUG_{label}.txt", data, fmt="%.6f")
    return results
def main():
    output_path = input("Enter the full path (including filename) to save the CSV output (e.g., /home/user/output.csv): ").strip()
    
    if not output_path.endswith(".csv"):
        print("Error: Output file must end with .csv")
        return

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        print(f"Error: Output directory '{output_dir}' does not exist.")
        return

    # Get input CSV file
    excel_path = input("Enter the path to the input CSV file: ").strip()
    if not os.path.isfile(excel_path):
        print("Error: Invalid CSV file path.")
        return

    # Read input CSV
    try:
        df = pd.read_csv(excel_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Validate required columns
    required_columns = ["file name", "start x,y", "end x,y"]
    if not all(col in df.columns for col in required_columns):
        print(f"Error: CSV file must contain the columns: {required_columns}")
        return

    df["file name"] = df["file name"].astype(str).str.strip()
    print(df["file name"])

    # Get folder containing point cloud files
    folder_selected = input("Enter the folder containing point cloud files: ").strip()
    if not os.path.isdir(folder_selected):
        print("Error: Invalid folder path.")
        return

    # Define CSV columns
    columns = ["File Name", "Method", "Class", "Start Point", "Length", "IRI Value", "Variance", "Number of Points"]

    # Create (or overwrite) the CSV file and write the header
    pd.DataFrame(columns=columns).to_csv(output_path, index=False, mode="w")

    # Process each row in the CSV
    for _, row in df.iterrows():
        file_name = row["file name"].strip()

        # Check if the corresponding file exists in the selected folder
        file_path = os.path.join(folder_selected, file_name)
        if not os.path.isfile(file_path):
            print(f"Skipping {file_name} (File not found in folder)")
            continue

        try:
            start_point = np.array(list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", row["start x,y"]))) + [1])
            end_point = np.array(list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", row["end x,y"]))) + [1])
        except Exception as e:
            print(f"Error parsing coordinates for {file_name}: {e}")
            continue

        data = read_txt_with_rgb(file_path)
        if data is None:
            print(f"Failed to load data from {file_path}")
            continue

        # Process point cloud
        oriented_data, start_point, end_point = orient_point_cloud(data, start_point, end_point)

        filtered_points = get_points(oriented_data, start_point, end_point)
        filtered_points_mesh = get_mesh_points(oriented_data, start_point, end_point, uniform_spacing=0.05)
        filtered_points_inter_pchip = get_interpolated_points("pchip", filtered_points)
        base_name = file_name.split(".")[0]

        # Save outputs
        np.savetxt(f"{base_name}_points.csv", filtered_points, delimiter=",", fmt="%.6f")
        np.savetxt(f"{base_name}_mesh.csv", filtered_points_mesh, delimiter=",", fmt="%.6f")
        np.savetxt(f"{base_name}_pchip.csv", filtered_points_inter_pchip, delimiter=",", fmt="%.6f")

        # Calculate IRI values
        print("\n---------------------- Points ----------------------\n")
        iri_values = print_iri(filtered_points, f"{base_name}_points.csv")
        print("\n---------------------- Mesh ----------------------\n")
        iri_mesh_values = print_iri(filtered_points_mesh, f"{base_name}_mesh.csv")
        print("\n---------------------- PCHIP Interpolation ----------------------\n")
        iri_pchip_values = print_iri(filtered_points_inter_pchip, f"{base_name}_pchip.csv")

        # Store results in a temporary list (processed_results)
        processed_results = []
        results = [base_name, "points", *iri_values, "mesh", *iri_mesh_values, "pchip", *iri_pchip_values]

        method_name = None  # Keeps track of the current method
        for cls_data in results[1:]:  # Skip the file name
            
            if isinstance(cls_data, str):  
                # If it's a string, it represents a method name (e.g., "points", "mesh", "pchip")
                method_name = cls_data
            else:
                # Process classification group under the current method
                class_name = cls_data[0]  # Extract class name
                num_points = cls_data[2]  # Extract number of points

                # Extract numerical values safely
                values_array = cls_data[1][0]  # This is a numpy array inside a tuple
                if values_array.shape[1] >= 4:  # Ensure there are at least 4 values
                    values = values_array[0]  # Extract first row
                    start_point, length, iri_value, variance = values[:4]
                else:
                    print(f"Skipping {class_name}: Not enough numerical values")
                    continue

                # Append structured data to the new list
                processed_results.append([base_name, method_name, class_name, start_point, length, iri_value, variance, num_points])

        # Append an empty row after processing each file for readability
        processed_results.append([""] * len(columns))

        # Convert results to a DataFrame and append to the CSV file
        df_temp = pd.DataFrame(processed_results, columns=columns)
        df_temp.to_csv(output_path, index=False, mode="a", header=False)

    print(f"Results saved to {output_path}")

    """
    Tests:
        # start_point = ast.literal_eval(input("Enter the start point: "))
        # end_point = ast.literal_eval(input("Enter the end point: "))
        # start_point = [709676.562500, 5648699.000000, 1018.541992]
        # end_point = [709629.062500, 5648710.500000, 1019.014526]

        # start_point =  [709627.33624268, 5648709.33099365, 1018.97027588]
        # end_point = [709677.22576904, 5648693.19976807, 1018.35424805]

        # start_point = [709627.621765, 5648712.299011, 1019.054260]
        # end_point = [709677.142029, 5648699.590759, 1018.554016]

        # # # vertical
        # start_point = [709655.119271, 5648717.076043, 1019.693002]
        # end_point = [709654.937500, 5648695.00000, 1018.645020]
    """
    # file_path = input("Enter the file path: ")
    ### TEST: Comment top, uncomment below
    # file_path = "combined_00215N_R1R1_18000_20000_section_8.txt"
    # if not os.path.isfile(file_path):
    #     print("Invalid file path.")
    #     return

    # # Read the point cloud data
    # data = read_txt_with_rgb(file_path)
    # if data is None:
    #     print(f"Failed to load data from {file_path}")
    #     return

    # ## comment out these lines if you want to use the test points
    # # orig_start_point = ast.literal_eval(
    # #     input("Enter the start point [start_x, start_y, start_z]: ")
    # # )
    # # orig_end_point = ast.literal_eval(
    # #     input("Enter the end point [end_x, end_y, end_z]: ")
    # # )
    # ### TEST: Comment top, uncomment below
    # # orig_start_point = [709628.4969, 5648707.5704, 1]
    # # orig_end_point = [709673.4494, 5648695.5671, 1]

    # # Multi class point testing (vertical line) 
    # orig_start_point = [709656.1615, 5648694.5150, 1]
    # orig_end_point = [709668.0676, 5648718.6052, 1]

    # # orienting the data by defining a new x axis
    # oriented_data, start_point, end_point = orient_point_cloud(
    #     data, orig_start_point, orig_end_point
    # )
    # print("start end ", start_point, end_point)

    # ### OPTIONAL: Save the oriented data
    # output_file = f"{'oriented_'}{file_path.split('/')[-1]}"
    # np.savetxt(output_file, oriented_data, fmt="%.6f")

    # print("---------------------- Points ----------------------")
    # # now, we can filter points
    # filtered_points = get_points(oriented_data, start_point, end_point)
    # output_file = f"filtered_points_{orig_start_point}_{orig_end_point}.txt"
    # np.savetxt(output_file, filtered_points, fmt="%.6f")

    # print_iri(filtered_points, output_file)

    
    # print("---------------------- Mesh ----------------------")
    # filtered_points_mesh = get_mesh_points(
    #     oriented_data, start_point, end_point, uniform_spacing=0.05
    # )
    # output_file = f"mesh_{orig_start_point}_{orig_end_point}.txt"
    # print_iri(filtered_points_mesh, output_file)

    # print("---------------------- PChip ----------------------")
    # filtered_points_inter_pchip = get_interpolated_points("pchip", filtered_points)
    # output_file = f"pchip_{orig_start_point}_{orig_end_point}.txt"
    # print_iri(filtered_points_inter_pchip, output_file)


if __name__ == "__main__":
    main()