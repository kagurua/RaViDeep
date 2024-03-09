import vtk
from vtk.util.numpy_support import numpy_to_vtk
from scipy.spatial import ConvexHull
import numpy as np

def rotate(point, roll, pitch):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])

    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])

    R = np.dot(R_y, R_x)
    point = np.array(point)
    rotated_point = np.dot(R, point)
    return rotated_point


class CustomInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    def __init__(self, rate=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rotation_rate = rate

    def Rotate(self):
        if self.GetInteractor().GetShiftKey():
            dx, dy = self.GetDeltaMousePosition()
            self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().GetActiveCamera().Azimuth(self.rotation_rate * dx)
            self.GetInteractor().GetRenderWindow().GetRenderers().GetFirstRenderer().GetActiveCamera().Elevation(self.rotation_rate * dy)
            self.GetInteractor().GetRenderWindow().Render()
        else:
            super().Rotate()

    def GetDeltaMousePosition(self):
        last_pos = self.GetInteractor().GetLastEventPosition()
        current_pos = self.GetInteractor().GetEventPosition()
        dx = current_pos[0] - last_pos[0]
        dy = current_pos[1] - last_pos[1]
        return dx, dy


class PointCloudVisualizer:
    def __init__(self):
        self.renderer = vtk.vtkRenderer()
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        self.interactor.SetInteractorStyle(CustomInteractorStyle())
        self.camera_settings = None

    def add_point_cloud(self, points, categories):
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(numpy_to_vtk(points, deep=True))

        colors = vtk.vtkUnsignedCharArray()
        colors.SetNumberOfComponents(3)
        color_map = {0: [150, 75, 0], 1: [191, 0, 191], 2: [191, 191, 0]}
        if categories is None:
            for _ in range(points.shape[0]):
                colors.InsertNextTuple([0, 0, 0])
        else:
            for category in categories:
                colors.InsertNextTuple(color_map[category])

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)
        polydata.GetPointData().SetScalars(colors)

        point_size = vtk.vtkVertexGlyphFilter()
        point_size.SetInputData(polydata)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(point_size.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(3)

        self.renderer.AddActor(actor)
        self.renderer.SetBackground(0.97, 0.97, 0.97)

    def visualize(self):
        self.render_window.Render()
        self.interactor.Start()
        self.capture_camera_settings()

    def capture_camera_settings(self):
        camera = self.renderer.GetActiveCamera()
        self.camera_settings = camera.GetPosition(), camera.GetFocalPoint(), camera.GetViewUp()

    def apply_camera_settings(self, camera_settings):
        self.renderer.GetActiveCamera().SetPosition(*camera_settings[0])
        self.renderer.GetActiveCamera().SetFocalPoint(*camera_settings[1])
        self.renderer.GetActiveCamera().SetViewUp(*camera_settings[2])
        self.renderer.ResetCameraClippingRange()

    def save_screenshot(self, filename, width=1920, height=1080):
        self.render_window.SetSize(width, height)
        self.renderer.SetBackground(0.97, 0.97, 0.97)
        self.render_window.Render()

        window_to_image_filter = vtk.vtkWindowToImageFilter()
        window_to_image_filter.SetInput(self.render_window)
        window_to_image_filter.SetScale(1)
        window_to_image_filter.ReadFrontBufferOff()
        window_to_image_filter.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetFileName(filename)
        writer.SetInputConnection(window_to_image_filter.GetOutputPort())
        writer.Write()
        self.render_window.Finalize()
        self.interactor.TerminateApp()

    def save_screenshot_with_3d_labels(self, points, labels, filename):
        linesPolyData = vtk.vtkPolyData()
        points_vtk = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        pointIdx = 0
        points_map = {}
        roll = 0.175
        pitch = -0.055
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:
                continue
            class_points = points[labels == label, :3]
            hull_2d = ConvexHull(class_points[:, :2])
            min_height = np.min(class_points[:, 2])
            max_height = np.max(class_points[:, 2])
            center_point = np.mean(class_points[hull_2d.vertices, :2], axis=0)
            for vertex in hull_2d.vertices:
                point_2d = class_points[vertex]
                rotate_point = rotate([point_2d[0]-center_point[0], point_2d[1]-center_point[1], 0], roll, pitch)
                points_vtk.InsertNextPoint(point_2d[0], point_2d[1], min_height+rotate_point[2])
                points_vtk.InsertNextPoint(point_2d[0], point_2d[1], max_height+rotate_point[2])
                points_map[vertex] = pointIdx
                pointIdx += 2
            for simplex in hull_2d.simplices:
                for i in range(2):
                    line = vtk.vtkLine()
                    idx0 = points_map[simplex[i]]
                    idx1 = idx0 + 1
                    line.GetPointIds().SetId(0, idx0)
                    line.GetPointIds().SetId(1, idx1)
                    lines.InsertNextCell(line)
        linesPolyData.SetPoints(points_vtk)
        linesPolyData.SetLines(lines)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(linesPolyData)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0, 0, 1)
        actor.GetProperty().SetLineWidth(2)
        self.renderer.AddActor(actor)
        self.save_screenshot(filename)
