from enum import Enum

import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData

from .helpers import DataHelpers, PlotHelpers, MatrixHelpers


def _compute_landmarks(data, predefined_landmarks):
    landmark_names = ["IA", "TS", "AA", "AC"]
    if predefined_landmarks is None:
        landmarks = PlotHelpers.pickable_scapula_geometry_plot(points_name=landmark_names, data=data)
    else:
        landmarks = predefined_landmarks

    if False in [name in landmarks.keys() for name in landmark_names]:
        raise RuntimeError("Not all required points were selected")

    return landmarks


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
    NORMALIZED = 2
    LOCAL = 3
    GCS = 4


class Scapula:
    def __init__(self, filepath: str, predefined_landmarks: dict[str, np.ndarray]) -> None:
        self.raw_data = _load_scapula_geometry(filepath)
        self.normalized_data = DataHelpers.rough_normalize(self.raw_data)

        self.landmarks = _compute_landmarks(self.normalized_data, predefined_landmarks)

        # Project the scapula in its local reference frame based on ISB
        self.gcs = _compute_isb_coordinate_system(self.landmarks)
        self.data = MatrixHelpers.transpose_homogenous_matrix(self.gcs) @ self.normalized_data

    def get_data(self, data_type: DataType) -> np.ndarray:
        if data_type == DataType.RAW:
            return self.raw_data
        elif data_type == DataType.NORMALIZED:
            return self.normalized_data
        elif data_type == DataType.LOCAL:
            return self.data
        elif data_type == DataType.GCS:
            return self.gcs
        else:
            raise ValueError("Unsupported data type")

    def plot_geometry(
        self, data_type: DataType = DataType.LOCAL, plot_axes: bool = True, show_now: bool = False
    ) -> None:
        fig = plt.figure(f"Scapula")
        ax = fig.add_subplot(111, projection="3d")

        data = self.get_data(data_type)
        ax.scatter(data[0, :], data[1, :], data[2, :], ".", s=1, picker=5, alpha=0.3)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.set_box_aspect([1, 1, 1])

        if plot_axes:
            PlotHelpers.show_axes(ax, np.eye(4))

        if show_now:
            plt.show()
