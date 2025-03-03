import numpy as np
import pandas as pd
import os
import re
import open3d as o3d
from scipy.spatial import Delaunay
from scipy.interpolate import PchipInterpolator
from iri import iri




# Function to read point cloud data from a TXT file
def read_txt_with_rgb(filename):
    """Reads TXT file and returns point cloud data as a NumPy array, supporting both space and comma delimiters."""
    data = []
    with open(filename, "r") as f:
        for line in f:
            # Determine delimiter (comma if present, otherwise space)
            delimiter = "," if "," in line else " "
            values = list(map(float, line.strip().split(delimiter)))
            data.append(values)
    return np.array(data)

# Function to orient point cloud data based on start and end points
def orient_point_cloud(data, point1, point2):
    coords = data[:, :3]
    rgb = data[:, 3:] if data.shape[1] > 3 else np.zeros_like(coords)

    # Ensure start and end points assume z = 0
    p1 = np.array([point1[0], point1[1], 0])  
    p2 = np.array([point2[0], point2[1], 0])  

    v_x = p2 - p1
    v_x = v_x / np.linalg.norm(v_x)
    v_y = np.array([-v_x[1], v_x[0], 0])
    v_z = np.array([0, 0, 1])

    R = np.vstack([v_x, v_y, v_z]).T

    translated_coords = coords - p1
    oriented_coords = np.dot(translated_coords, R)
    oriented_coords[:, 2] = coords[:, 2]
    oriented_data = np.hstack([oriented_coords, rgb])

    return oriented_data, [0, 0, p1[2]], np.dot(R.T, (p2 - p1))

# Function to extract points along the specified line segment
def get_points(points, start_point, end_point, threshold=0.025):
    start_point = np.array(start_point)
    end_point = np.array(end_point)
    line_vector = end_point[:2] - start_point[:2]
    line_length_squared = np.dot(line_vector, line_vector)

    filtered_points = []
    for point in points:
        point_vector = point[:2] - start_point[:2]
        t = np.dot(point_vector, line_vector) / line_length_squared
        t = np.clip(t, 0, 1)
        projection = start_point[:2] + t * line_vector
        distance = np.linalg.norm(point[:2] - projection)

        if distance <= threshold:
            point[1] = 0  # Set y = 0
            filtered_points.append(point)

    return np.array(filtered_points)

# Function to compute IRI
def calculate_iri(data):
    data = data[:, [0, 2]]
    _, unique_indices = np.unique(data[:, 0], return_index=True)
    data_cleaned = data[unique_indices]
    data_cleaned = data_cleaned[np.argsort(data_cleaned[:, 0])]
    segment_length = np.max(data[:, 0]) - np.min(data[:, 0])

    return iri(data_cleaned, segment_length, -1, 0)

# Function to compute IRI from a mesh-based surface
def get_mesh_points(points, start_point, end_point, spacing=0.05):
    """Interpolates points on a mesh surface for more accurate IRI calculations."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])

    pcd_xy = np.asarray(pcd.points)[:, :2]
    tri = Delaunay(pcd_xy)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = pcd.points
    mesh.triangles = o3d.utility.Vector3iVector(tri.simplices)
    mesh.compute_vertex_normals()

    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    mesh_vertices = mesh.vertex["positions"].numpy()
    mesh_triangles = tri.simplices

    los_direction = end_point - start_point
    los_direction /= np.linalg.norm(los_direction)
    distance = np.linalg.norm(end_point - start_point)
    num_steps = int(distance / spacing)

    filtered_points = []
    for i in range(num_steps):
        p = start_point + i * spacing * los_direction
        simplex = tri.find_simplex(p[:2])
        if simplex >= 0:
            a, b, c = mesh_vertices[mesh_triangles[simplex]]
            v0, v1, v2 = b - a, c - a, np.array([p[0] - a[0], p[1] - a[1], 0])
            d00, d01, d11 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v1, v1)
            d20, d21 = np.dot(v2, v0), np.dot(v2, v1)
            denom = d00 * d11 - d01 * d01
            beta = (d11 * d20 - d01 * d21) / denom
            gamma = (d00 * d21 - d01 * d20) / denom
            alpha = 1.0 - beta - gamma
            interpolated_point = alpha * a + beta * b + gamma * c
            new_point = [p[0], p[1], interpolated_point[2]]
            filtered_points.append(new_point)

    return np.array(filtered_points)

# Function to compute IRI using PCHIP interpolation
def get_interpolated_points(points):
    """Interpolates using PCHIP for smooth road roughness estimation."""
    x, z = points[:, 0], points[:, 2]
    # Ensure x-values are strictly increasing
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    z_sorted = z[sorted_indices]
    # Remove duplicates
    unique_indices = np.unique(x_sorted, return_index=True)[1]
    x_sorted = x_sorted[unique_indices]
    z_sorted = z_sorted[unique_indices]

    x_new = np.arange(x_sorted.min(), x_sorted.max() + 0.05, 0.05)
    interpolator = PchipInterpolator(x_sorted, z_sorted)
    z_new = interpolator(x_new)

    return np.column_stack((x_new, np.zeros_like(x_new), z_new))

# Function to process a single file
def process_file(file_path, start_point, end_point):
    print(f"\nProcessing: {file_path}")
    
    data = read_txt_with_rgb(file_path)
    if data is None or len(data) == 0:
        print(f"Failed to load data from {file_path}")
        return
    
    start_point = np.array([start_point[0], start_point[1], 0], dtype=float)
    end_point = np.array([end_point[0], end_point[1], 0], dtype=float)

    oriented_data, start_transformed, end_transformed = orient_point_cloud(data, start_point, end_point)
    
    # Direct extraction-based IRI
    filtered_points = get_points(oriented_data, start_transformed, end_transformed)
    iri_direct = calculate_iri(filtered_points)

    # Mesh-based IRI
    mesh_points = get_mesh_points(oriented_data, start_transformed, end_transformed)
    iri_mesh = calculate_iri(mesh_points)

    # PCHIP Interpolation-based IRI
    interpolated_points = get_interpolated_points(filtered_points)
    iri_pchip = calculate_iri(interpolated_points)

    print(f"IRI Direct: {iri_direct}")
    print(f"IRI Mesh-Based: {iri_mesh}")
    print(f"IRI PCHIP Interpolation: {iri_pchip}")
    return [file_path, iri_direct[0][0][0], iri_direct[0][0][1], iri_direct[0][0][2], iri_direct[0][0][3],
        iri_mesh[0][0][0], iri_mesh[0][0][1], iri_mesh[0][0][2], iri_mesh[0][0][3],
        iri_pchip[0][0][0], iri_pchip[0][0][1], iri_pchip[0][0][2], iri_pchip[0][0][3]]


# Main function
def main():




    output_path = input("Enter the full path (including filename) to save the CSV output (e.g., /home/user/output.csv): ")
    # Ensure the user entered a valid file path
    if os.path.isdir(output_path):
        print("Error: You entered a directory. Please enter a full file path ending with .csv")
        return

    results = []  # Initialize list to store results
    
    excel_path = input("Enter the path to the Excel file: ")
    if not os.path.isfile(excel_path):
        print("Invalid Excel file path.")
        return

    df = pd.read_excel(excel_path)
    required_columns = ["file name", "start x,y", "end x,y"]
    if not all(col in df.columns for col in required_columns):
        print("Excel file must contain columns: 'file name', 'start x,y', 'end x,y'")
        return

    df["file name"] = df["file name"].astype(str).str.strip()

    folder_selected = input("Enter the folder containing point cloud files: ")
    if not os.path.isdir(folder_selected):
        print("Invalid folder path.")
        return

    for file_name in os.listdir(folder_selected):
        if not file_name.endswith(".txt"):
            continue
        file_path = os.path.join(folder_selected, file_name)

        match = df[df["file name"] == file_name.replace(".txt", "")]
        if match.empty:
            continue

        start_point = list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", match["start x,y"].values[0])))
        end_point = list(map(float, re.findall(r"[-+]?\d*\.\d+|\d+", match["end x,y"].values[0])))

 
        
        iri_values = process_file(file_path, start_point, end_point)

        if iri_values:
            results.append(iri_values)

        
        # Save results to CSV
        columns = ["File Name", "IRI Direct - IRI", "IRI Direct - Length", "IRI Direct - Other1", "IRI Direct - Other2",
           "IRI Mesh - IRI", "IRI Mesh - Length", "IRI Mesh - Other1", "IRI Mesh - Other2",
           "IRI PCHIP - IRI", "IRI PCHIP - Length", "IRI PCHIP - Other1", "IRI PCHIP - Other2"]
        results_df = pd.DataFrame(results, columns=columns)
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")



if __name__ == "__main__":
    main()

