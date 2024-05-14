import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from plyfile import PlyData

from .enums import ScapulaDataType, JointCoordinateSystem
from .helpers import DataHelpers, PlotHelpers, MatrixHelpers


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

        # The following fields need to be filled by the public constructor
        self._landmarks: dict[str, np.ndarray] = None  # The landmarks of the scapula in RAW_NORMALIZED
        self._gcs: np.ndarray = None  # The coordinate system based on ISB that gets data from RAW_NORMALIZED to LOCAL
        self.local_data: np.ndarray = None  # The scapula data in LOCAL

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
        if shared_indices_with_reference:
            _, landmark_idx = MatrixHelpers.nearest_neighbor(
                reference_landmarks[:3, :], reference_scapula.local_data[:3, :]
            )
        else:
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
        self._gcs = JointCoordinateSystem.ISB(self._landmarks)
        # self._gcs = _compute_isb_coordinate_system(self._landmarks)

        # Compute the local data
        self.local_data = MatrixHelpers.transpose_homogenous_matrix(self._gcs) @ self.normalized_raw_data

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
            return self.local_data
        else:
            raise ValueError("Unsupported data type")

    @property
    def landmark_names(self):
        """
        Get the names of the landmarks.

        Returns:
        landmark_names: the names of the landmarks
        """
        return ["IA", "TS", "AA", "GC", "CP", "SA", "AT"]

    @property
    def landmarks_long_names(self):
        """
        Get the long names of the landmarks.

        Returns:
        landmark_names: the long names of the landmarks
        """
        return [
            "Inferior Angulus",
            "Trighonum Spinae",
            "Acromion Angle",
            "Glenoid Center",
            "Coracoid Process",
            "Superior Angle",
            "Acromial Tip",
        ]

    def get_joint_coordinates_system(
        self,
        jcs_type: JointCoordinateSystem = JointCoordinateSystem.ISB,
        data_type: ScapulaDataType = ScapulaDataType.LOCAL,
    ) -> np.ndarray:
        """
        Get the joint coordinate system based on the landmarks.

        Args:
        jcs_type: the desired joint coordinate system
        data_type: the desired pose of the scapula data

        Returns:
        jcs: the joint coordinate system as a 4x4 matrix
        """
        return jcs_type(self.landmarks(data_type))

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
            gcs_T = MatrixHelpers.transpose_homogenous_matrix(self._gcs)
            out = {key: gcs_T @ val for key, val in self._landmarks.items()}
        else:
            raise ValueError("Unsupported data type")

        if as_array:
            return np.array([val for val in out.values()]).T
        else:
            return out

    def _define_landmarks(self, predefined_landmarks: dict[str, np.ndarray] = None) -> dict[str, np.ndarray]:
        """
        Define the landmarks of the scapula.

        Args:
        predefined_landmarks: dictionary containing the landmarks as keys and the coordinates as values in RAW_NORMALIZED.
        If None, the user will be prompted to select the landmarks on the scapula geometry.

        Returns:
        landmarks: the landmarks of the scapula
        """
        landmarks = predefined_landmarks

        if landmarks is None or False in [name in landmarks.keys() for name in self.landmark_names]:
            landmarks = self.plot_pickable_geometry(
                points_name=self.landmark_names, data_type=ScapulaDataType.RAW_NORMALIZED, initial_guesses=landmarks
            )

        return landmarks

    def plot_geometry(
        self,
        ax: plt.Axes = None,
        data_type: ScapulaDataType = ScapulaDataType.LOCAL,
        show_jcs: tuple[JointCoordinateSystem] = None,
        show_landmarks: bool = True,
        show_now: bool = False,
        **kwargs,
    ) -> None | plt.Axes:
        """
        Plot the scapula geometry.

        Args:
        ax: matplotlib axis to plot the scapula geometry
        data_type: the desired pose of the scapula data
        show_jcs: list of joint coordinate systems to show
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

        if show_jcs is not None:
            for jcs in show_jcs:
                PlotHelpers.show_axes(jcs(self.landmarks(data_type)), ax=ax)

        if show_landmarks:
            landmarks = self.landmarks(data_type, as_array=True)
            ax.scatter(landmarks[0, :], landmarks[1, :], landmarks[2, :], c="g", s=50)

        if show_now:
            plt.show()
            return None
        else:
            return ax

    def plot_pickable_geometry(
        self,
        points_name: list[str],
        data_type: ScapulaDataType = ScapulaDataType.LOCAL,
        initial_guesses: dict[str, np.array] = None,
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
        scatter = ax.scatter(np.nan, np.nan, np.nan, ".", c="r")
        scatter.set_sizes([50])

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
            fig.canvas.draw_idle()

        def on_confirmed_point(event):
            # Save the previous point
            if event is not None:
                picked_points[points_name[current_point[0]]] = picked_point[0]

            current_point[0] += 1

            if current_point[0] == len(points_name):
                plt.close(fig)
                return

            point_name = points_name[current_point[0]]

            if point_name in initial_guesses.keys():
                picked_point[0] = initial_guesses[point_name]
            else:
                picked_point[0] = np.array([np.nan, np.nan, np.nan])

            scatter._offsets3d = (
                picked_point[0][np.newaxis, 0],
                picked_point[0][np.newaxis, 1],
                picked_point[0][np.newaxis, 2],
            )

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

    @staticmethod
    def compute_average_reference_system_from_reference(
        scapulas: list["Scapula"],
        jcs_type: JointCoordinateSystem,
        reference_system: JointCoordinateSystem = None,
    ) -> np.ndarray:
        """
        Compute the average reference system from the reference coordinate system to the desired coordinate system for all the
        scapulas. If None is passed as the reference system, the average reference system will be computed based on the
        desired coordinate system.

        Args:
        scapulas: list of scapulas
        jcs_type: the desired joint coordinate system

        Returns:
        average_matrix: the average reference system
        """
        # Do not care about translation, so set the origin of all the reference frames to the same point
        all_rt = []
        for scapula in scapulas:
            if reference_system is None:
                rt_reference = np.eye(4)
            else:
                rt_reference = scapula.get_joint_coordinates_system(reference_system)
            rt = scapula.get_joint_coordinates_system(jcs_type)

            all_rt.append(MatrixHelpers.transpose_homogenous_matrix(rt_reference) @ rt)

        return MatrixHelpers.average_matrices(all_rt, compute_std=True)

    @staticmethod
    def plot_systems_in_reference_scapula(
        reference_scapula: "Scapula", scapulas: list["Scapula"], jcs_type: JointCoordinateSystem
    ):
        """
        Plot the average reference system in the reference scapula for all the scapulas. The translation is not taken
        into account so it is easier to compare the axes.

        Args:
        reference_scapula: the reference scapula
        scapulas: list of scapulas
        jcs_type: the desired joint coordinate system
        """
        # Do not care about translation, so set the origin of all the reference frames to the same point
        reference_origin = reference_scapula.get_joint_coordinates_system(jcs_type)[:3, 3]
        all_rt = []
        for scapula in scapulas:
            # scapula = reference_scapula
            rt = scapula.get_joint_coordinates_system(jcs_type)
            rt[:3, 3] = reference_origin
            all_rt.append(rt)

        average_matrix = MatrixHelpers.average_matrices(all_rt)

        # Plot the average scapula with all the axes
        ax = reference_scapula.plot_geometry(show_now=False, marker="o", color="b", s=5, alpha=0.1, show_jcs=[jcs_type])
        for rt in all_rt:
            PlotHelpers.show_axes(rt, ax=ax)
        PlotHelpers.show_axes(axes=average_matrix, ax=ax, linewidth=10)
        PlotHelpers.show()
