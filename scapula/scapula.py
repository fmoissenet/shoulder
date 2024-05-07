from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from plyfile import PlyData

from .helpers import DataHelpers, PlotHelpers, MatrixHelpers


def _compute_isb_coordinate_system(landmarks: dict[str, np.array]) -> np.array:
    origin = landmarks["AA"][:3]
    x = landmarks["AA"][:3] - landmarks["TS"][:3]
    z = landmarks["TS"][:3] - landmarks["IA"][:3]
    y = np.cross(z, x)
    z = np.cross(x, y)
    x /= np.linalg.norm(x)
    y /= np.linalg.norm(y)
    z /= np.linalg.norm(z)
    lcs = np.eye(4)
    lcs[:3, 0] = x
    lcs[:3, 1] = y
    lcs[:3, 2] = z
    lcs[:3, 3] = origin
    return lcs


def _load_scapula_geometry(file_path: str) -> np.ndarray:
    extension = file_path.split(".")[-1]
    if extension == "ply":
        data = None
        with open(file_path, "rb") as f:
            plydata = PlyData.read(f)
            tp = np.asarray(plydata["vertex"])
            data = np.array((tp["x"], tp["y"], tp["z"]))
            data = np.concatenate((data, np.ones((1, data.shape[1]))))
        data

    else:
        raise NotImplementedError(f"The file extension {extension} is not supported yet.")

    return data


class DataType(Enum):
    RAW = 1
    RAW_NORMALIZED = 2
    LOCAL = 3


class Scapula:
    def __init__(
        self, filepath: str, predefined_landmarks: dict[str, np.ndarray] = None, reference_scapula: "Scapula" = None
    ) -> None:
        self.raw_data = _load_scapula_geometry(filepath)
        self.normalized_raw_data = DataHelpers.rough_normalize(self.raw_data)

        if reference_scapula is None:
            # If this is the reference scapula, we must compute the landmarks
            self.landmarks = self._compute_landmarks(predefined_landmarks)

            # Project the scapula in its local reference frame based on ISB
            self.gcs = _compute_isb_coordinate_system(self.landmarks)

            self.data = MatrixHelpers.transpose_homogenous_matrix(self.gcs) @ self.normalized_raw_data

        else:
            # Otherwise, we align the current scapula to the reference scapula and find the landmarks
            self.data = MatrixHelpers.icp(self.normalized_raw_data, reference_scapula.get_data(DataType.LOCAL))

    def get_data(self, data_type: DataType) -> np.ndarray:
        if data_type == DataType.RAW:
            return self.raw_data
        elif data_type == DataType.RAW_NORMALIZED:
            return self.normalized_raw_data
        elif data_type == DataType.LOCAL:
            return self.data
        else:
            raise ValueError("Unsupported data type")

    def _compute_landmarks(self, predefined_landmarks: dict[str, np.ndarray] = None) -> dict[str, np.ndarray]:
        landmark_names = ["IA", "TS", "AA", "AC"]
        if predefined_landmarks is None:
            landmarks = self.plot_pickable_geometry(points_name=landmark_names, data_type=DataType.RAW_NORMALIZED)
        else:
            landmarks = predefined_landmarks

        if False in [name in landmarks.keys() for name in landmark_names]:
            raise RuntimeError("Not all required points were selected")

        return landmarks

    def plot_geometry(
        self,
        ax: plt.Axes = None,
        data_type: DataType = DataType.LOCAL,
        plot_axes: bool = True,
        show_now: bool = False,
        **kwargs,
    ) -> None | plt.Axes:
        if ax is None:
            fig = plt.figure(f"Scapula")
            ax = fig.add_subplot(111, projection="3d")

        data = self.get_data(data_type)
        if "s" not in kwargs:
            kwargs["s"] = 1
        if "alpha" not in kwargs:
            kwargs["alpha"] = 0.3
        if "color" not in kwargs:
            kwargs["color"] = "b"
        if "marker" not in kwargs:
            kwargs["marker"] = "."
        ax.scatter(data[0, :], data[1, :], data[2, :], picker=5, **kwargs)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.set_box_aspect([1, 1, 1])

        if plot_axes:
            PlotHelpers.show_axes(ax, np.eye(4))

        if show_now:
            plt.show()
            return None
        else:
            return ax

    def plot_pickable_geometry(
        self, points_name: list[str], data_type: DataType = DataType.LOCAL
    ) -> dict[str, np.array]:
        # Prepare the figure
        fig = plt.figure(f"Pick the points")
        ax = fig.add_subplot(111, projection="3d")

        data = self.get_data(data_type)
        self.plot_geometry(ax=ax, data_type=data_type, plot_axes=False)
        scatter = ax.scatter(np.nan, np.nan, np.nan, ".", c="r", s=50)

        # Add the next button
        axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        bnext = Button(axnext, "Next")

        def on_pick_point(event):
            picked_index = int(event.ind[0])

            picked_point[0] = data[:, picked_index]
            scatter._offsets3d = (
                picked_point[0][np.newaxis, 0],
                picked_point[0][np.newaxis, 1],
                picked_point[0][np.newaxis, 2],
            )
            scatter.set_sizes([50])
            fig.canvas.draw_idle()

        def on_confirmed_point(event):
            # Save the previous point
            if event is not None:
                picked_points[points_name[current_point[0]]] = picked_point[0]

            current_point[0] += 1

            if current_point[0] == len(points_name):
                plt.close(fig)
                return

            picked_point[0] = np.array([np.nan, np.nan, np.nan])
            scatter._offsets3d = (
                picked_point[0][np.newaxis, 0],
                picked_point[0][np.newaxis, 1],
                picked_point[0][np.newaxis, 2],
            )

            point_name = points_name[current_point[0]]
            ax.title.set_text(f"Pick the {point_name} then close the window")
            fig.canvas.draw_idle()

        # Setup the connection
        picked_points = {}
        current_point = [-1]
        picked_point = [np.array([np.nan, np.nan, np.nan])]
        on_confirmed_point(None)
        fig.canvas.mpl_connect("pick_event", on_pick_point)
        bnext.on_clicked(on_confirmed_point)
        plt.show()

        return picked_points
