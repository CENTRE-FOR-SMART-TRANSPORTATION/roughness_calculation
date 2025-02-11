import open3d as o3d
import numpy as np
from scipy.spatial import Delaunay


class PointCloudMeshViewer:
    def __init__(self, filename):
        self.pcd = self.load_point_cloud(filename)
        self.pcd = self.normalize(self.pcd)
        self.mesh = self.create_mesh(self.pcd)
        self.current_mode = "pointcloud"  # Start with point cloud view
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window()
        self.vis.get_render_option().background_color = np.array([0, 0, 0])
        self.setup_visualizer()

    def load_point_cloud(self, filename):
        """Load point cloud from a space-separated text file (x y z r g b)."""
        data = np.loadtxt(filename)
        points, colors = data[:, :3], data[:, 3:6] / 255.0  # Normalize colors

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Estimate normals for meshing
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        return pcd

    def normalize(self, pcd):
        """Normalize point cloud to fit within a unit cube."""
        points = np.asarray(pcd.points)
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        x = x - np.min(x)
        y = y - np.min(y)
        z = z
        pcd.points = o3d.utility.Vector3dVector(np.stack((x, y, z), axis=1))
        return pcd

    def create_mesh(self, pcd):
        """Convert point cloud to mesh using Poisson reconstruction."""
        pcd_xy = np.asarray(pcd.points)[:, :2]
        tri = Delaunay(pcd_xy)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = pcd.points
        mesh.triangles = o3d.utility.Vector3iVector(tri.simplices)
        mesh.compute_vertex_normals()
        return mesh

    def setup_visualizer(self):
        """Set up Open3D visualizer with keyboard callbacks."""
        self.vis.create_window("Point Cloud & Mesh Viewer")

        # Add initial geometry (point cloud)
        self.vis.add_geometry(self.pcd)

        # Key bindings for switching modes
        self.vis.register_key_callback(ord("P"), self.show_pointcloud)
        self.vis.register_key_callback(ord("M"), self.show_mesh)

        print("Press 'P' to view Point Cloud, 'M' to view Mesh")
        self.vis.run()

    def show_pointcloud(self, vis):
        """Switch to point cloud view."""
        print("Switching to Point Cloud View")
        vis.clear_geometries()
        vis.add_geometry(self.pcd)

    def show_mesh(self, vis):
        """Switch to mesh view."""
        print("Switching to Mesh View")
        vis.clear_geometries()
        vis.add_geometry(self.mesh)


if __name__ == "__main__":
    filename = "combined_00215N_R1R1_18000_20000_section_8.txt"  # Change this to your actual file
    PointCloudMeshViewer(filename)
