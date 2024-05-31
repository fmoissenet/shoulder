import math
import os
from typing import Generator

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from plyfile import PlyData
from stl import Mesh

from .enums import ScapulaDataType, JointCoordinateSystem
from .matrix_helper import DataHelpers, MatrixHelpers
from .plot_helper import PlotHelpers


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

    elif extension == "stl":
        mesh = Mesh.from_file(file_path)
        all_vertices = mesh.vectors.reshape(-1, 3)
        data = np.unique(all_vertices, axis=0).T
        data = MatrixHelpers.from_euler(np.array([-np.pi / 4, 0, 0]), "xyz") @ data
        data = np.concatenate((data, np.ones((1, data.shape[1]))))

    else:
        raise NotImplementedError(f"The file extension {extension} is not supported yet.")

    if is_left:
        data[0, :] *= -1

    return data


class Scapula:
    def __init__(self, geometry: str | np.ndarray, is_left: bool) -> None:
        """
        Private constructor to load the scapula geometry from a file. This should not be called directly.

        Args:
        geometry: Whether a file path to the scapula geometry or the scapula geometry itself
        is_left: whether the scapula is the left one or a right one. If left, the scapula will be mirrored.

        Returns:
        None
        """

        if isinstance(geometry, str):
            self.raw_data = _load_scapula_geometry(geometry, is_left=is_left)
        elif isinstance(geometry, np.ndarray):
            self.raw_data = geometry
        else:
            raise ValueError("Invalid geometry type")
        self.normalized_raw_data = DataHelpers.rough_normalize(self.raw_data)

        # The following fields need to be filled by the public constructor
        self._landmarks: dict[str, np.ndarray] = None  # The landmarks of the scapula in RAW_NORMALIZED
        self._gcs: np.ndarray = None  # The coordinate system based on ISB that gets data from RAW_NORMALIZED to LOCAL
        self.local_data: np.ndarray = None  # The scapula data in LOCAL

    @classmethod
    def from_landmarks(
        cls,
        geometry: str | np.ndarray,
        predefined_landmarks: dict[str, np.ndarray] = None,
        is_left: bool = False,
    ) -> "Scapula":
        """
        Public constructor to create a Scapula object from landmarks.

        Args:
        geometry: Whether a file path to the scapula geometry or the scapula geometry itself
        predefined_landmarks: dictionary containing the landmarks as keys and the coordinates as values in RAW_NORMALIZED.
        If None, the user will be prompted to select the landmarks on the scapula geometry.
        is_left: whether the scapula is the left one or a right one. If left, the scapula will be mirrored.

        Returns:
        Scapula object
        """
        scapula = cls(geometry=geometry, is_left=is_left)

        # Fill the mandatory fields that are not filled by the private constructor
        scapula._fill_mandatory_fields(predefined_landmarks=predefined_landmarks)

        # Return the scapula object
        return scapula

    @classmethod
    def from_reference_scapula(
        cls,
        geometry: str | np.ndarray,
        reference_scapula: "Scapula",
        shared_indices_with_reference: bool = False,
        is_left: bool = False,
    ) -> "Scapula":
        """
        Public constructor to create a Scapula object from a reference scapula.

        Args:
        geometry: Whether a file path to the scapula geometry or the scapula geometry itself
        reference_scapula: reference scapula object
        shared_indices_with_reference: whether to use the same indices for both scapulas (much faster as it skips the
        nearest neighbor search)
        is_left: whether the scapula is the left one or a right one. If left, the scapula will be mirrored.

        Returns:
        Scapula object
        """
        scapula = cls(geometry=geometry, is_left=is_left)

        # Find the landmarks so we can call the from_landmarks constructor

        # Compute the local data
        expected_maximal_error = 0.01  # 1% of the scapula size
        gcs_T, error = MatrixHelpers.icp(
            scapula.normalized_raw_data,
            reference_scapula.get_data(ScapulaDataType.LOCAL),
            share_indices=shared_indices_with_reference,
            return_points_error=True,
        )
        if not shared_indices_with_reference and error > expected_maximal_error:
            # Try a second type but rotating the values to see if it improves the result
            # It is useless to try this in a shared indices scenario as the indices ensure that the rotation is correct
            # by definition
            gcs_T_tp, error_tp = MatrixHelpers.icp(
                scapula.normalized_raw_data,
                reference_scapula.get_data(ScapulaDataType.LOCAL),
                share_indices=False,
                initial_rt=MatrixHelpers.from_euler([np.pi], "z", homogenous=True) @ gcs_T,
                return_points_error=True,
            )
            # Take the best result
            if error_tp < error:
                gcs_T = gcs_T_tp

        local_data = gcs_T @ scapula.normalized_raw_data

        # Find the landmarks in the normalized data
        reference_landmarks = reference_scapula.landmarks(ScapulaDataType.LOCAL, as_array=True)
        if shared_indices_with_reference:
            _, landmark_idx = MatrixHelpers.nearest_neighbor(
                reference_landmarks[:3, :], reference_scapula.local_data[:3, :]
            )
        else:
            _, landmark_idx = MatrixHelpers.nearest_neighbor(reference_landmarks[:3, :], local_data[:3, :])
        landmarks = {key: scapula.raw_data[:, landmark_idx[i]] for i, key in enumerate(scapula.landmark_names)}

        # Create and return the scapula object
        scapula._fill_mandatory_fields(predefined_landmarks=landmarks)
        return scapula

    @classmethod
    def generator(
        cls,
        reference_scapula: "Scapula",
        models_folder: str,
        number_to_generate: int = 1,
        model: str = "A",
        mode_ranges=None,
    ) -> Generator["Scapula", None, None]:
        """
        Function to generate a surface model from a PCA statistical model, based on the chosen statistical model type
        It yields the produced surface model

        Args:
        reference_scapula: the reference scapula
        models_folder: directory where the statistical models are located
        number_to_generate: number of models to generate
        model: statistical model type (A, P or H), default is A
        mode_ranges: dictionary with the number of modes for each statistical model, default is {"A": 7, "P": 8, "H": 18}

        Returns:
        Iterator of the produced surface model
        """

        def _generate(
            mode: list[int], mean_model: np.ndarray, pca_eigen_vectors: np.ndarray, pca_eigen_values: np.ndarray
        ) -> "Scapula":
            """
            Function to create a surface model from a PCA statistical model

            Args:
            mode: number of modes to apply
            mean_model: mean model
            pca_eigen_vectors: eigen vectors estimated by PCA
            pca_eigen_values: eigen values estimated by PCA

            Returns:
            The produced surface model
            """

            # b is a vector of floats to apply to each mode of variation
            full_b = np.zeros(pca_eigen_values.shape)
            for i in mode:
                # A uniformly randomized value between statistically acceptable bounds +/-3 * sqrt(eigen_value)
                new_value = np.random.uniform(-3 * math.sqrt(pca_eigen_values[i]), 3 * math.sqrt(pca_eigen_values[i]))
                full_b[i] = new_value

            # Calculate P * b
            pca_applied = pca_eigen_vectors.dot(full_b)

            # Create a new model: mean_model + pca_applied
            generated_model = mean_model.T.flatten() + pca_applied
            geometry = np.reshape(generated_model, (-1, 3)).T
            return Scapula.from_reference_scapula(
                geometry=geometry,
                reference_scapula=reference_scapula,
                shared_indices_with_reference=True,
                is_left=False,
            )

        def _read_vertices(path) -> np.ndarray:
            """
            Function to read the vertices of a 3D model

            Args:
            path: path to the 3D model

            Returns:
            vertices of the 3D model
            """
            ply_data = PlyData.read(path)
            vertices_tp = np.asarray(ply_data["vertex"])
            return np.array((vertices_tp["x"], vertices_tp["y"], vertices_tp["z"]))

        def _load_statistical_model() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            """
            Function to load the statistical model and its parameters

            Returns:
            mean model, vertices of the mean model, eigen vectors, eigen values
            """
            if model not in ("A", "P", "H"):
                raise ValueError("Invalid model type, choose between 'A', 'P' or 'H'")

            model_stat = {
                "A": {
                    "mean_model_path": os.path.join(models_folder, "PJ116_scapula_A_avg.ply"),
                    "pca_eigen_vectors": os.path.join(models_folder, "PJ116_eigen_vectors_scapula_A.csv"),
                    "pca_eigen_values": os.path.join(models_folder, "PJ116_eigen_values_scapula_A.csv"),
                },
                "P": {
                    "mean_model_path": os.path.join(models_folder, "PJ116_scapula_P_avg.ply"),
                    "pca_eigen_vectors": os.path.join(models_folder, "PJ116_eigen_vectors_scapula_P.csv"),
                    "pca_eigen_values": os.path.join(models_folder, "PJ116_eigen_values_scapula_P.csv"),
                },
                "H": {
                    "mean_model_path": os.path.join(models_folder, "FHOrtho_scapula_avg.ply"),
                    "pca_eigen_vectors": os.path.join(models_folder, "PJ116_eigen_vectors_scapula_FHOrtho.csv"),
                    "pca_eigen_values": os.path.join(models_folder, "PJ116_eigen_values_scapula_FHOrtho.csv"),
                },
            }

            mean_model = _read_vertices(model_stat[model]["mean_model_path"])
            pca_eigen_vectors = np.loadtxt(model_stat[model]["pca_eigen_vectors"], delimiter=";", dtype=float)
            pca_eigen_values = np.loadtxt(model_stat[model]["pca_eigen_values"], delimiter=";", dtype=float)

            return mean_model, pca_eigen_vectors, pca_eigen_values

        if mode_ranges is None:
            mode_ranges = {"A": 7, "P": 8, "H": 18}
        mode = range(mode_ranges[model])

        # Read mean model and PCA parameters (eigen vectors and values) according to the chosen statistical model
        statistical_model = _load_statistical_model()

        # Generate new models: new_model = mean_model + P * b
        for _ in range(number_to_generate):
            yield _generate(mode, *statistical_model)

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
        return ["AA", "AC", "AI", "GC", "IE", "SE", "TS"]

    @property
    def landmarks_long_names(self):
        """
        Get the long names of the landmarks.

        Returns:
        landmark_names: the long names of the landmarks
        """
        return [
            "Acromion Angle",
            "Dorsal of Acromioclavicular joint",
            "Angulus Inferior",
            "Glenoid Center",
            "Inferior Edge of glenoid",
            "Superior Edge of glenoid",
            "Trighonum Spinae",
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
        predefined_landmarks: dictionary containing the landmarks as keys and the coordinates as values in RAW.
        If None, the user will be prompted to select the landmarks on the scapula geometry.

        Returns:
        landmarks: the landmarks of the scapula in RAW_NORMALIZED
        """
        landmarks = predefined_landmarks

        # Convert the indices to coordinates
        for name in landmarks.keys():
            if name not in landmarks:
                continue
            if isinstance(landmarks[name], int):
                pass
            elif isinstance(landmarks[name], np.ndarray):
                # Find the closest point to the predefined landmark
                _, idx = MatrixHelpers.nearest_neighbor(landmarks[name][:3, None], self.raw_data[:3, :])
                landmarks[name] = idx[0]
            else:
                raise ValueError(f"The landmark {name} should be indices or a 3x1 or a 4x1 array")

        # Make sure all the points are defined
        if landmarks is None or False in [name in landmarks.keys() for name in self.landmark_names]:
            landmarks = self.plot_pickable_geometry(
                points_name=self.landmark_names, data_type=ScapulaDataType.RAW_NORMALIZED, initial_guesses=landmarks
            )

        # Convert the landmarks indices to the normalized data
        for name in landmarks.keys():
            landmarks[name] = self.normalized_raw_data[:, landmarks[name]]

        return landmarks

    def plot_geometry(
        self,
        ax: plt.Axes = None,
        data_type: ScapulaDataType = ScapulaDataType.LOCAL,
        show_jcs: tuple[JointCoordinateSystem] = None,
        show_landmarks: bool = True,
        show_now: bool = False,
        landmarks_color: str = "g",
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
            ax.scatter(landmarks[0, :], landmarks[1, :], landmarks[2, :], c=landmarks_color, s=50)

        if show_now:
            plt.show()
            return None
        else:
            return ax

    def plot_pickable_geometry(
        self,
        points_name: list[str],
        data_type: ScapulaDataType = ScapulaDataType.LOCAL,
        initial_guesses: dict[str, int] = None,
    ) -> dict[str, int]:
        """
        Plot the scapula geometry and allow the user to pick the points.

        Args:
        points_name: the names of the points to pick
        data_type: the desired pose of the scapula data
        initial_guesses: dictionary containing the initial guesses as keys and the indices as values

        Returns:
        picked_points: dictionary containing the indices of the picked points
        """
        # Prepare the figure
        fig = plt.figure(f"Pick the points")
        ax = fig.add_subplot(111, projection="3d")

        data = self.get_data(data_type)
        self.plot_geometry(ax=ax, data_type=data_type, show_landmarks=False)
        scatter = ax.scatter(np.nan, np.nan, np.nan, ".", c="r")
        scatter.set_sizes([50])
        if initial_guesses is None:
            initial_guesses = {}

        # Add the next button
        axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        bnext = Button(axnext, "Next")

        def on_pick_point(event):
            picked_index[0] = int(event.ind[0])

            picked_point = data[:, picked_index[0]]
            scatter._offsets3d = (picked_point[np.newaxis, 0], picked_point[np.newaxis, 1], picked_point[np.newaxis, 2])
            fig.canvas.draw_idle()

        def on_confirmed_point(event):
            # Save the previous point
            if event is not None:
                picked_points[points_name[current_point[0]]] = picked_index[0]

            current_point[0] += 1

            if current_point[0] == len(points_name):
                plt.close(fig)
                return

            point_name = points_name[current_point[0]]

            if point_name in initial_guesses.keys():
                picked_index[0] = initial_guesses[point_name]
                picked_point = data[:, initial_guesses[point_name]]
            else:
                picked_index[0] = None
                picked_point = np.array([np.nan, np.nan, np.nan])
            scatter._offsets3d = (picked_point[np.newaxis, 0], picked_point[np.newaxis, 1], picked_point[np.newaxis, 2])

            ax.title.set_text(f"Pick the {point_name} then close the window")
            fig.canvas.draw_idle()

        # Setup the connection
        picked_points = {}
        current_point = [-1]
        picked_index = [None]

        on_confirmed_point(None)
        fig.canvas.mpl_connect("pick_event", on_pick_point)
        bnext.on_clicked(on_confirmed_point)
        plt.show()

        return picked_points

    @staticmethod
    def change_frame_of_reference(
        scapulas: list["Scapula"],
        jcs_type: JointCoordinateSystem,
        reference_system: JointCoordinateSystem = None,
    ) -> list["Scapula"]:
        """
        Change the frame of reference of the scapulas to the desired coordinate system. If None is passed as the reference
        system, no transformation will be applied.

        Args:
        scapulas: list of scapulas
        jcs_type: the desired joint coordinate system
        reference_system: the reference coordinate system

        Returns:
        scapulas: list of scapulas with the new frame of reference
        """
        all_rt = []
        for scapula in scapulas:
            if reference_system is None:
                rt_reference = np.eye(4)
            else:
                rt_reference = scapula.get_joint_coordinates_system(reference_system)
            rt = scapula.get_joint_coordinates_system(jcs_type)

            all_rt.append(MatrixHelpers.transpose_homogenous_matrix(rt_reference) @ rt)
        return all_rt

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
