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


def _load_scapula_geometry(file_path: str, is_left: bool) -> np.ndarray:
    """
    Load the scapula geometry from a file.

    Args:
    file_path: path to the scapula geometry file
    is_left: whether the scapula is the left one or a right one. If left, the scapula will be mirrored.

    Returns:
    data: the scapula geometry as a numpy array
    """
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

    if is_left:
        data[0, :] *= -1

    return data


class ScapulaDataType(Enum):
    RAW = 1
    RAW_NORMALIZED = 2
    LOCAL = 3


class Scapula:
    def __init__(self, filepath: str, is_left: bool) -> None:
        """
        Private constructor to load the scapula geometry from a file. This should not be called directly.

        Args:
        filepath: path to the scapula geometry file
        is_left: whether the scapula is the left one or a right one. If left, the scapula will be mirrored.

        Returns:
        None
        """

        self.raw_data = _load_scapula_geometry(filepath, is_left=is_left)
        self.normalized_raw_data = DataHelpers.rough_normalize(self.raw_data)

        self._landmarks: dict[str, np.ndarray] = None  # This will be filled by the private constructor
        self.gcs: np.ndarray = None  # This will be filled by the private constructor
        self.data: np.ndarray = None  # This will be filled by the public constructor

    @classmethod
    def from_landmarks(
        cls, filepath: str, predefined_landmarks: dict[str, np.ndarray] = None, is_left: bool = False
    ) -> "Scapula":
        """
        Public constructor to create a Scapula object from landmarks.

        Args:
        filepath: path to the scapula geometry file
        predefined_landmarks: dictionary containing the landmarks as keys and the coordinates as values in RAW_NORMALIZED.
        If None, the user will be prompted to select the landmarks on the scapula geometry.
        is_left: whether the scapula is the left one or a right one. If left, the scapula will be mirrored.

        Returns:
        Scapula object
        """
        scapula = cls(filepath=filepath, is_left=is_left)

        # Fill the mandatory fields that are not filled by the private constructor
        scapula._fill_mandatory_fields(predefined_landmarks=predefined_landmarks)

        # Return the scapula object
        return scapula

    @classmethod
    def from_reference_scapula(
        cls,
        filepath: str,
        reference_scapula: "Scapula",
        shared_indices_with_reference: bool = False,
        is_left: bool = False,
    ) -> "Scapula":
        """
        Public constructor to create a Scapula object from a reference scapula.

        Args:
        filepath: path to the scapula geometry file
        reference_scapula: reference scapula object
        shared_indices_with_reference: whether to use the same indices for both scapulas (much faster as it skips the
        nearest neighbor search)
        is_left: whether the scapula is the left one or a right one. If left, the scapula will be mirrored.

        Returns:
        Scapula object
        """
        scapula = cls(filepath=filepath, is_left=is_left)

        # Find the landmarks so we can call the from_landmarks constructor

        # Compute the local data
        gcs_T = MatrixHelpers.icp(
            scapula.normalized_raw_data,
            reference_scapula.get_data(ScapulaDataType.LOCAL),
            share_indices=shared_indices_with_reference,
        )
        local_data = gcs_T @ scapula.normalized_raw_data

        # Find the landmarks in the normalized data
        reference_landmarks = reference_scapula.landmarks(ScapulaDataType.LOCAL, as_array=True)
        _, landmark_idx = MatrixHelpers.nearest_neighbor(reference_landmarks[:3, :], local_data[:3, :])
        landmarks = {
            key: scapula.normalized_raw_data[:, landmark_idx[i]] for i, key in enumerate(scapula.landmark_names)
        }

        # Create and return the scapula object
        scapula._fill_mandatory_fields(predefined_landmarks=landmarks)
        return scapula

    def _fill_mandatory_fields(self, predefined_landmarks: dict[str, np.ndarray] = None) -> None:
        """
        Fill the mandatory fields of the scapula object left unfilled by the private constructor.
        This should be called after the public constructors.

        Args:
        predefined_landmarks: dictionary containing the landmarks as keys and the coordinates as values in RAW_NORMALIZED.
        If None, the user will be prompted to select the landmarks on the scapula geometry.

        Returns:
        None
        """
        self._landmarks = self._define_landmarks(predefined_landmarks)

        # Project the scapula in its local reference frame based on ISB
        self.gcs = _compute_isb_coordinate_system(self._landmarks)

        # Compute the local data
        self.data = MatrixHelpers.transpose_homogenous_matrix(self.gcs) @ self.normalized_raw_data

    def get_data(self, data_type: ScapulaDataType) -> np.ndarray:
        """
        Get the scapula data in the desired format.

        Args:
        data_type: the desired pose of the scapula data

        Returns:
        data: the scapula data in the desired format
        """
        if data_type == ScapulaDataType.RAW:
            return self.raw_data
        elif data_type == ScapulaDataType.RAW_NORMALIZED:
            return self.normalized_raw_data
        elif data_type == ScapulaDataType.LOCAL:
            return self.data
        else:
            raise ValueError("Unsupported data type")

    @property
    def landmark_names(self):
        """
        Get the names of the landmarks.

        Returns:
        landmark_names: the names of the landmarks
        """
        return ["IA", "TS", "AA", "AC"]

    def _define_landmarks(self, predefined_landmarks: dict[str, np.ndarray] = None) -> dict[str, np.ndarray]:
        """
        Define the landmarks of the scapula.

        Args:
        predefined_landmarks: dictionary containing the landmarks as keys and the coordinates as values in RAW_NORMALIZED.
        If None, the user will be prompted to select the landmarks on the scapula geometry.

        Returns:
        landmarks: the landmarks of the scapula
        """
        if predefined_landmarks is None:
            landmarks = self.plot_pickable_geometry(
                points_name=self.landmark_names, data_type=ScapulaDataType.RAW_NORMALIZED
            )
        else:
            landmarks = predefined_landmarks

        if False in [name in landmarks.keys() for name in self.landmark_names]:
            raise RuntimeError("Not all required points were selected")

        return landmarks

    def landmarks(
        self, data_type: ScapulaDataType = ScapulaDataType.RAW_NORMALIZED, as_array: bool = False
    ) -> dict[str, np.array] | np.ndarray:
        """
        Get the landmarks in the desired format.

        Args:
        data_type: the desired pose of the landmarks
        as_array: whether to return the landmarks as a numpy array or as a dictionary

        Returns:
        landmarks: the landmarks in the desired format
        """

        if data_type == ScapulaDataType.RAW_NORMALIZED:
            out = self._landmarks
        elif data_type == ScapulaDataType.LOCAL:
            gcs_T = MatrixHelpers.transpose_homogenous_matrix(self.gcs)
            out = {key: gcs_T @ val for key, val in self._landmarks.items()}
        else:
            raise ValueError("Unsupported data type")

        if as_array:
            return np.array([val for val in out.values()]).T
        else:
            return out

    def plot_geometry(
        self,
        ax: plt.Axes = None,
        data_type: ScapulaDataType = ScapulaDataType.LOCAL,
        show_axes: bool = True,
        show_landmarks: bool = True,
        show_now: bool = False,
        **kwargs,
    ) -> None | plt.Axes:
        """
        Plot the scapula geometry.

        Args:
        ax: matplotlib axis to plot the scapula geometry
        data_type: the desired pose of the scapula data
        show_axes: whether to show the axes of the scapula
        show_landmarks: whether to show the landmarks of the scapula
        show_now: whether to show the plot now or return the axis
        **kwargs: additional arguments to pass to the scatter function

        Returns:
        None if show_now is True, the axis otherwise
        """
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

        if show_landmarks:
            landmarks = self.landmarks(data_type, as_array=True)
            ax.scatter(landmarks[0, :], landmarks[1, :], landmarks[2, :], c="g", s=50)

        if show_now:
            plt.show()
            return None
        else:
            return ax

    def plot_pickable_geometry(
        self, points_name: list[str], data_type: ScapulaDataType = ScapulaDataType.LOCAL
    ) -> dict[str, np.array]:
        """
        Plot the scapula geometry and allow the user to pick the points.

        Args:
        points_name: the names of the points to pick
        data_type: the desired pose of the scapula data

        Returns:
        picked_points: dictionary containing the picked points
        """
        # Prepare the figure
        fig = plt.figure(f"Pick the points")
        ax = fig.add_subplot(111, projection="3d")

        data = self.get_data(data_type)
        self.plot_geometry(ax=ax, data_type=data_type, show_landmarks=False, show_axes=False)
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
