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


class ScapulaDataType(Enum):
    RAW = 1
    RAW_NORMALIZED = 2
    LOCAL = 3


class Scapula:
    def __init__(self, filepath: str) -> None:
        """
        Private constructor to load the scapula geometry from a file. This should not be called directly.

        Args:
        filepath: path to the scapula geometry file

        Returns:
        None
        """

        self.raw_data = _load_scapula_geometry(filepath)
        self.normalized_raw_data = DataHelpers.rough_normalize(self.raw_data)

        self.landmarks = None  # This will be filled by the private constructor
        self.gcs = None  # This will be filled by the private constructor
        self.data = None  # This will be filled by the public constructor

    @classmethod
    def from_landmarks(cls, filepath: str, predefined_landmarks: dict[str, np.ndarray] = None) -> "Scapula":
        """
        Public constructor to create a Scapula object from landmarks.

        Args:
        filepath: path to the scapula geometry file
        predefined_landmarks: dictionary containing the landmarks as keys and the coordinates as values. If None,
        the user will be prompted to select the landmarks on the scapula geometry.

        Returns:
        Scapula object
        """
        scapula = cls(filepath=filepath)

        ## Fill the mandatory fields that are not filled by the private constructor

        # Compute or points the landmarks
        scapula.landmarks = scapula._define_landmarks(predefined_landmarks)

        # Project the scapula in its local reference frame based on ISB
        scapula.gcs = _compute_isb_coordinate_system(scapula.landmarks)

        # Compute the local data
        scapula.data = MatrixHelpers.transpose_homogenous_matrix(scapula.gcs) @ scapula.normalized_raw_data

        # Return the scapula object
        return scapula

    @classmethod
    def from_reference_scapula(cls, filepath: str, reference_scapula: "Scapula") -> "Scapula":
        """
        Public constructor to create a Scapula object from a reference scapula.

        Args:
        filepath: path to the scapula geometry file
        reference_scapula: reference scapula object

        Returns:
        Scapula object
        """
        scapula = cls(filepath=filepath)

        ## Fill the mandatory fields that are not filled by the private constructor

        # Find the global coordinate system transformation
        gcs_T = MatrixHelpers.icp(scapula.normalized_raw_data, reference_scapula.get_data(ScapulaDataType.LOCAL))
        scapula.gcs = MatrixHelpers.transpose_homogenous_matrix(gcs_T)

        # Compute the local data
        scapula.data = gcs_T @ scapula.normalized_raw_data

        # Return the scapula object
        return scapula

    def get_data(self, data_type: ScapulaDataType) -> np.ndarray:
        if data_type == ScapulaDataType.RAW:
            return self.raw_data
        elif data_type == ScapulaDataType.RAW_NORMALIZED:
            return self.normalized_raw_data
        elif data_type == ScapulaDataType.LOCAL:
            return self.data
        else:
            raise ValueError("Unsupported data type")

    def _define_landmarks(self, predefined_landmarks: dict[str, np.ndarray] = None) -> dict[str, np.ndarray]:
        landmark_names = ["IA", "TS", "AA", "AC"]
        if predefined_landmarks is None:
            landmarks = self.plot_pickable_geometry(
                points_name=landmark_names, data_type=ScapulaDataType.RAW_NORMALIZED
            )
        else:
            landmarks = predefined_landmarks

        if False in [name in landmarks.keys() for name in landmark_names]:
            raise RuntimeError("Not all required points were selected")

        return landmarks

    def plot_geometry(
        self,
        ax: plt.Axes = None,
        data_type: ScapulaDataType = ScapulaDataType.LOCAL,
        show_axes: bool = True,
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

        if show_axes:
            if data_type == ScapulaDataType.LOCAL:
                PlotHelpers.show_axes(ax, np.eye(4))
            elif data_type == ScapulaDataType.RAW_NORMALIZED:
                PlotHelpers.show_axes(ax, self.gcs)
            else:
                raise ValueError("Unsupported data type")

        if show_now:
            plt.show()
            return None
        else:
            return ax

    def plot_pickable_geometry(
        self, points_name: list[str], data_type: ScapulaDataType = ScapulaDataType.LOCAL
    ) -> dict[str, np.array]:
        # Prepare the figure
        fig = plt.figure(f"Pick the points")
        ax = fig.add_subplot(111, projection="3d")

        data = self.get_data(data_type)
        self.plot_geometry(ax=ax, data_type=data_type, show_axes=False)
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
