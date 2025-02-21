import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import roughness as r
import numpy as np
import platform

isMacOS = platform.system() == "Darwin"


class PointCloudApp:
    MENU_OPEN = 1
    MENU_EXPORT = 2
    MENU_QUIT = 3
    MENU_ABOUT = 11

    def __init__(self):
        # Create a window
        self.window = gui.Application.instance.create_window("Open3D", 1024, 768)
        w = self.window  # to make the code more concise

        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)
        self._scene.scene.set_background(np.asarray([52, 52, 52, 255]) / 255)
        self._scene.set_on_mouse(self.on_mouse_down)
        self._em = w.theme.font_size
        self.geometries = {}

        # Load point cloud
        self.point_cloud = o3d.geometry.PointCloud()

        # Store selected points
        self._selected_points = []

        # Set up layout
        w.add_child(self._scene)
        if gui.Application.instance.menubar is None:
            if isMacOS:
                app_menu = gui.Menu()
                app_menu.add_item("About", PointCloudApp.MENU_ABOUT)
                app_menu.add_separator()
                app_menu.add_item("Quit", PointCloudApp.MENU_QUIT)
            file_menu = gui.Menu()
            file_menu.add_item("Open...", PointCloudApp.MENU_OPEN)
            file_menu.add_item("Export Current Image...", PointCloudApp.MENU_EXPORT)
            if not isMacOS:
                file_menu.add_separator()
                file_menu.add_item("Quit", PointCloudApp.MENU_QUIT)
            help_menu = gui.Menu()
            help_menu.add_item("About", PointCloudApp.MENU_ABOUT)

            menu = gui.Menu()
            if isMacOS:
                menu.add_menu("Example", app_menu)
                menu.add_menu("File", file_menu)
            else:
                menu.add_menu("File", file_menu)
                menu.add_menu("Help", help_menu)
            gui.Application.instance.menubar = menu

        w.set_on_menu_item_activated(PointCloudApp.MENU_OPEN, self._on_menu_open)
        w.set_on_menu_item_activated(PointCloudApp.MENU_EXPORT, self._on_menu_export)
        w.set_on_menu_item_activated(PointCloudApp.MENU_QUIT, self._on_menu_quit)
        w.set_on_menu_item_activated(PointCloudApp.MENU_ABOUT, self._on_menu_about)

    def get_selected_points(self):
        return self._selected_points

    def set_selected_points(self, points):
        self._selected_points.append(points)
        if len(self._selected_points) == 2:
            start_point = self._selected_points[0]
            end_point = self._selected_points[1]
            self._selected_points = []

            points = self.convert_to_numpy(self.point_cloud)
            oriented, start_point, end_point = r.orient_point_cloud(
                points, start_point, end_point
            )

            print("-------------------Points-------------------")
            filtered_points = r.get_points(oriented, start_point, end_point)
            output_file = f"points_{start_point}_{end_point}.txt"
            r.print_iri(filtered_points, output_file)

            print("-------------------Mesh-------------------")
            filtered_points_mesh = r.get_mesh_points(oriented, start_point, end_point)
            output_file = f"mesh_{start_point}_{end_point}.txt"
            r.print_iri(filtered_points_mesh, output_file)

            print("-------------------PChip-------------------")
            filtered_points_inter_pchip = r.get_interpolated_points(
                "pchip", filtered_points
            )
            output_file = f"pchip_{start_point}_{end_point}.txt"
            r.print_iri(filtered_points_inter_pchip, output_file)

            filtered_pcd = self.convert_to_open3d(filtered_points)
            filtered_pcd_mesh = self.convert_to_open3d(filtered_points_mesh)
            filtered_pcd_inter_pchip = self.convert_to_open3d(
                filtered_points_inter_pchip
            )
            print(filtered_points.shape)
            print(filtered_points_mesh.shape)
            print(filtered_points_inter_pchip.shape)

            self.point_cloud = self.convert_to_open3d(oriented)
            self.clear_geometries()
            self.add_geometry("PointCloud", self.point_cloud)
            self.add_geometry("FilteredPointCloud", filtered_pcd)
            self.add_geometry("FilteredPointCloudMesh", filtered_pcd_mesh)
            self.add_geometry("FilteredPointCloudInterPchip", filtered_pcd_inter_pchip)
            save_point_cloud_to_txt(filtered_pcd, "filtered_line_pcd")
            save_point_cloud_to_txt(filtered_pcd_mesh, "filtered_line_pcd_mesh")
            save_point_cloud_to_txt(
                filtered_pcd_inter_pchip, "filtered_line_pcd_inter_pchip"
            )

    def convert_to_open3d(self, points):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        if points.shape[1] == 6:
            pcd.colors = o3d.utility.Vector3dVector(points[:, 3:] / 255)
        return pcd

    def convert_to_numpy(self, pcd):
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) * 255
        return np.concatenate((points, colors), axis=1)

    def _on_menu_open(self):
        dlg = gui.FileDialog(
            gui.FileDialog.OPEN, "Choose file to load", self.window.theme
        )
        dlg.add_filter(".txt", "TXT files (.txt)")
        dlg.add_filter("", "All files")

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_dialog_done)
        self.window.show_dialog(dlg)

    def _on_file_dialog_cancel(self):
        self.window.close_dialog()

    def _on_load_dialog_done(self, filename):
        self.window.close_dialog()
        self.load(filename)

    def _on_menu_export(self):
        dlg = gui.FileDialog(
            gui.FileDialog.SAVE, "Choose file to save", self.window.theme
        )
        dlg.add_filter(".png", "PNG files (.png)")
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_export_dialog_done)
        self.window.show_dialog(dlg)

    def _on_export_dialog_done(self, filename):
        self.window.close_dialog()
        frame = self._scene.frame
        self.export_image(filename, frame.width, frame.height)

    def _on_menu_quit(self):
        gui.Application.instance.quit()

    def _on_menu_about(self):
        em = self.window.theme.font_size
        dlg = gui.Dialog("About")
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("Roughness calculator"))
        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)
        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def _on_about_ok(self):
        self.window.close_dialog()

    def load(self, path):
        self._scene.scene.clear_geometry()

        if path.endswith(".txt"):
            new_pcd = r.read_txt_with_rgb(path)
            new_pcd = r.normalize_point_cloud_data(new_pcd)
            self.point_cloud = self.convert_to_open3d(new_pcd)
            save_point_cloud_to_txt(self.point_cloud, "loaded_pcd")
            # self.point_cloud, self.mesh = r.get_mesh_points(new_pcd)
        else:
            new_pcd = o3d.io.read_point_cloud(path)
            self.point_cloud = new_pcd

        self.add_geometry("PointCloud", self.point_cloud)

    def export_image(self, path, width, height):

        def on_image(image):
            img = image

            quality = 9  # png
            if path.endswith(".jpg"):
                quality = 100
            o3d.io.write_image(path, img, quality)

        self._scene.scene.scene.render_to_image(on_image)

    def on_mouse_down(self, event):
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(
            gui.KeyModifier.CTRL
        ):

            def depth_callback(depth_image):
                x = event.x - self._scene.frame.x
                y = event.y - self._scene.frame.y
                depth = np.asarray(depth_image)[y, x]

                if depth == 1.0:  # clicked on nothing (i.e. the far plane)
                    text = ""
                else:
                    world = self._scene.scene.camera.unproject(
                        x,
                        y,
                        depth,
                        self._scene.frame.width,
                        self._scene.frame.height,
                    )
                    text = "({:.3f}, {:.3f}, {:.3f})".format(
                        world[0], world[1], world[2]
                    )
                    print(f"Selected point: {text}")
                    self.set_selected_points(world)

            self._scene.scene.scene.render_to_depth_image(depth_callback)
            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED

    def add_geometry(self, name: str, geometry) -> None:
        if name not in self.geometries and geometry is not None:
            self.geometries[name] = geometry
            self._scene.scene.add_geometry(
                name,
                geometry,
                rendering.MaterialRecord(),  # rendering.MaterialRecord()
            )
            bounds = self._scene.scene.bounding_box
            self._scene.setup_camera(60, bounds, bounds.get_center())
        else:
            self.update_geometry(name, geometry)

    def clear_geometries(self):
        self._scene.scene.clear_geometry()
        self.geometries = {}

    def update_geometry(self, name: str, geometry: o3d.geometry.Geometry):
        if name in self.geometries and geometry is not None:
            self.geometries[name] = geometry
            self._scene.scene.remove_geometry(name)
            self._scene.scene.add_geometry(
                name,
                geometry,
                rendering.MaterialRecord(),  # rendering.MaterialRecord()
            )
            bounds = self._scene.scene.bounding_box
            self._scene.setup_camera(60, bounds, bounds.get_center())
        else:
            self.add_geometry(name, geometry)


def save_point_cloud_to_txt(point_cloud, file_name):
    """
    Save the points in a point cloud to a .txt file.

    Args:
        point_cloud: An open3d.geometry.PointCloud or open3d.cuda.pybind.geometry.PointCloud object.
        file_name: The name of the .txt file to save the points.
    """
    if isinstance(point_cloud, o3d.t.geometry.PointCloud):
        o3d.t.io.write_point_cloud(f"{file_name}.ply", point_cloud, write_ascii=True)
    else:
        o3d.io.write_point_cloud(f"{file_name}.ply", point_cloud, write_ascii=True)

    print(f"Saved {file_name}.")


if __name__ == "__main__":
    gui.Application.instance.initialize()
    w = PointCloudApp()
    gui.Application.instance.run()
