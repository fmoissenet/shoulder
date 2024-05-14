import numpy as np
import matplotlib.pyplot as plt


from .enums import JointCoordinateSystem


class DataHelpers:
    @staticmethod
    def rough_normalize(data: np.ndarray) -> np.ndarray:
        """
        Roughly normalize the data by dividing it by the maximum range. Hopefully this is the inferior angulus to
        the coracoid process

        Args:
        data: numpy array of shape (4, N) representing the data

        Returns:
        out: numpy array of shape (4, N) representing the normalized data
        """
        x_range = np.max(data[0, :]) - np.min(data[0, :])
        y_range = np.max(data[1, :]) - np.min(data[1, :])
        z_range = np.max(data[2, :]) - np.min(data[2, :])
        scale = np.sqrt(x_range**2 + y_range**2 + z_range**2)
        out = np.ones((4, data.shape[1]))
        out[:3, :] = data[:3, :] / scale
        return out


class MatrixHelpers:
    @staticmethod
    def transpose_homogenous_matrix(homogenous_matrix: np.ndarray) -> np.ndarray:
        """
        Transpose a homogenous matrix, following the formula:
        [R^T   , -R^T * t]
        [  0   ,     1   ]

        Args:
        homogenous_matrix: 4x4 matrix representing the homogenous matrix

        Returns:
        out: 4x4 matrix representing the transposed homogenous matrix
        """
        out = np.eye(4)
        out[:3, :3] = homogenous_matrix[:3, :3].transpose()
        out[:3, 3] = -out[:3, :3] @ homogenous_matrix[:3, 3]
        return out

    @staticmethod
    def compute_transformation(gcs1: np.ndarray, gcs2: np.ndarray) -> np.ndarray:
        """
        Compute the transformation matrix from gcs1 to gcs2, following the formula:
        gcs1.T @ gcs2

        Args:
        gcs1: 4x4 matrix representing the origin global coordinate system
        gcs2: 4x4 matrix representing the destination global coordinate system

        Returns:
        out: 4x4 matrix representing the transformation matrix from gcs1 to gcs2
        """
        return MatrixHelpers.transpose_homogenous_matrix(gcs1) @ gcs2

    @staticmethod
    def subtract_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Subtract two vectors, following the formula:
        [a - b]
        [  1  ]

        Args:
        a: numpy array of shape (4, N) representing the first vector
        b: numpy array of shape (4, N) representing the second vector

        Returns:
        out: numpy array of shape (4, N) representing the subtracted vectors
        """
        out = a - b
        out[3, :] = 1
        return out

    @staticmethod
    def icp(
        points1: np.ndarray,
        points2: np.ndarray,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        nb_points1_required: int = 3000,
        nb_points2_required: int = 3000,
        share_indices: bool = False,
    ):
        """
        Perform ICP to align points1 to points2.

        Args:
        points1: numpy array of shape (3, N) representing the first point cloud
        points2: numpy array of shape (3, M) representing the second point cloud
        max_iterations: maximum number of iterations for ICP
        tolerance: convergence threshold
        nb_points1_required: number of points to use in the first point cloud
        nb_points2_required: number of points to use in the second point cloud
        share_indices: whether to use the same indices for both point clouds (much faster as it skips the nearest neighbor search)

        Returns:
        aligned_points: numpy array of shape (3, N) representing the aligned first point cloud
        """

        # Initial transformation (identity)
        r = np.eye(3)
        t = np.zeros((3, 1))

        # Copy the points
        pts1_mean = np.concatenate([np.mean(points1[:3, :], axis=1, keepdims=True), [[1]]])
        pts1_zeroed = MatrixHelpers.subtract_vectors(points1, pts1_mean)

        # Iterate
        pts1_slice_jumps = 1
        pts2_slice_jumps = 1
        if not share_indices:
            pts1_slice_jumps = points1.shape[1] // nb_points1_required
            pts2_slice_jumps = points2.shape[1] // nb_points2_required

        pts2 = points2[:3, ::pts2_slice_jumps]
        prev_error = np.inf
        rt = np.eye(4)
        for _ in range(max_iterations):
            pts1 = (rt @ pts1_zeroed[:, ::pts1_slice_jumps])[:3, :]

            # Find the nearest neighbors
            if share_indices:
                indices = np.arange(pts1.shape[1])
            else:
                __, indices = MatrixHelpers.nearest_neighbor(pts1, pts2)

            # Compute the transformation
            rt_step = MatrixHelpers.compute_best_fit_transformation(pts1, pts2[:, indices])
            rt = rt_step @ rt

            # Check convergence
            squared_error = np.sum((np.eye(4) - rt_step) ** 2)
            if np.abs(prev_error - squared_error) < tolerance:
                break
            prev_error = squared_error

        # Reproject to the initial pose of the first point cloud
        rt[:3, 3] -= (rt[:3, :3] @ pts1_mean[:3, :])[:3, 0]
        return rt

    @staticmethod
    def from_euler(angles, sequence):
        """
        Create a rotation matrix from Euler angles.

        Args:
        angles: numpy array of shape (3,) representing the Euler angles
        sequence: string representing the sequence of the Euler angles

        Returns:
        out: numpy array of shape (3, 3) representing the rotation matrix
        """
        out = np.eye(3)
        for angle, axis in zip(angles, sequence):
            if axis == "x":
                rotation = np.array(
                    [
                        [1, 0, 0],
                        [0, np.cos(angle), -np.sin(angle)],
                        [0, np.sin(angle), np.cos(angle)],
                    ]
                )
            elif axis == "y":
                rotation = np.array(
                    [
                        [np.cos(angle), 0, np.sin(angle)],
                        [0, 1, 0],
                        [-np.sin(angle), 0, np.cos(angle)],
                    ]
                )
            elif axis == "z":
                rotation = np.array(
                    [
                        [np.cos(angle), -np.sin(angle), 0],
                        [np.sin(angle), np.cos(angle), 0],
                        [0, 0, 1],
                    ]
                )
            out = np.dot(out, rotation)
        return out

    def average_matrices(matrices: list[np.ndarray], compute_std: bool = False) -> np.ndarray:
        """
        Compute the average of a list of matrices. If compute_std is True, also compute the standard deviation of the
        rotation matrices. This is done by computing the standard deviation of the angle of the rotation matrices. To
        compute the angle, we use the formula:
        angle = arccos((trace(R1.T @ R2) - 1) / 2)
        where "R1.T @ R2" is the rotation matrix between a matrix and the average matrix.

        Args:
        matrices: list of numpy arrays of shape (4, 4) representing the homogenous matrices to average
        compute_std: whether to compute the standard deviation of the rotation matrices

        Returns:
        average: numpy array of shape (4, 4) representing the average matrix
        std: tuple of floats representing the standard deviation of the rotation and translation matrices respectively
        """

        # Dispatch the input matrices
        rotations = np.array([np.array(mat[:3, :3]).T for mat in matrices]).T

        # Compute the average of the rotation matrices
        rotations_mean = np.mean(rotations, axis=2)
        u, _, v_T = np.linalg.svd(rotations_mean)
        average_rotation = u @ v_T

        # Compute the average of the translation vectors
        translations = np.array([np.array(mat[:3, 3]).T for mat in matrices]).T
        average_translation = np.mean(translations, axis=1)

        # Create the average matrix
        out = np.eye(4)
        out[:3, :3] = average_rotation
        out[:3, 3] = average_translation

        if not compute_std:
            return out

        # If we get here, we need to compute the standard deviation of the homogeneous matrices
        error_angles = []
        for i in range(rotations.shape[2]):
            error_angles.append(np.arccos((np.trace(rotations[:, :, i].T @ average_rotation) - 1) / 2))
        rotation_std = np.sqrt(np.mean(np.array(error_angles) ** 2))

        translation_std = np.std(np.linalg.norm(translations, axis=0))

        return out, (rotation_std, translation_std)

    def compute_best_fit_transformation(points1, points2):
        """
        Compute the transformation (R, t) that aligns points1 to points2, based on the algorithm from kabsch.

        Args:
        points1: numpy array of shape (3, N) representing the source point cloud
        points2: numpy array of shape (3, N) representing the destination point cloud

        Returns:
        transformation: numpy array of shape (4, 4) representing the average transformation matrix
        """
        # Compute centroids
        centroid1 = np.mean(points1, axis=1, keepdims=True)
        centroid2 = np.mean(points2, axis=1, keepdims=True)

        # Subtract centroids
        centered1 = points1 - centroid1
        centered2 = points2 - centroid2

        # Compute covariance matrix
        H = np.dot(centered1, centered2.T)

        # Singular Value Decomposition
        U, _, Vt = np.linalg.svd(H)

        # Rotation matrix
        r = np.dot(Vt.T, U.T)

        # Translation vector
        t = centroid2 - np.dot(r, centroid1)

        return np.concatenate([np.concatenate([r, t], axis=1), [[0, 0, 0, 1]]])

    @staticmethod
    def nearest_neighbor(src, dst):
        """
        Find the nearest (Euclidean distance) neighbor in dst for each point in src, based on the formula:
        d = sum((src - dst)^2)

        Args:
        src: numpy array of shape (3, N) representing the source point cloud
        dst: numpy array of shape (3, M) representing the destination point cloud

        Returns:
        distances: numpy array of shape (N,) containing the distances to the nearest neighbors
        indices: numpy array of shape (N,) containing the indices of the nearest neighbors in dst
        """
        squared_distances = np.ndarray(src.shape[1])
        indices = np.ndarray(src.shape[1], dtype=int)

        for i in range(src.shape[1]):
            tp = np.sum((dst - src[:, i : i + 1]) ** 2, axis=0)
            squared_distances[i] = np.min(tp)
            indices[i] = np.argmin(tp)

        return squared_distances, indices

    @staticmethod
    def export_average_to_latex(
        average_matrix: dict[str, tuple[np.ndarray, tuple[float, float]]], reference_system: JointCoordinateSystem
    ):
        """
        Export the average reference system to a LaTeX table. The values are expected to be in the format output by
        the average_matrices function with the compute_std flag set to True and put in a dictionary with the key being
        the name of the reference system.

        Args:
        average_matrix (dict[str, tuple[np.ndarray, float, float]]): The average reference system including the standard deviations
        of the form {key: (average_matrix, standard_deviations of rotation, standard_deviations of translation)}
        """

        all_average_str = []
        for key, average in average_matrix.items():
            key = key.replace("_", "\\_")

            row = f"{key} & \\begin{{tabular}}{{cccc}}\n"
            row += f"\\\\\n".join("&".join(f"{value:.2f}" for value in row) for row in average[0])
            row += f"\\end{{tabular}} & "
            row += f"{average[1][0]:.2f} / {average[1][1]:.2f}\\\\\n"

            all_average_str.append(row)

        all_average_str = f"\\cmidrule(lr){{1-3}}\n".join(all_average_str)

        latex_content = f"""
\\documentclass{{article}}
\\usepackage{{amsmath}}
\\usepackage{{array}}
\\usepackage{{booktabs}}
\\usepackage{{geometry}}
\\usepackage{{makecell}}

\\begin{{document}}

\\begin{{table}}[h!]
\\centering
\\begin{{tabular}}{{lcc}}
\\toprule
\\makecell{{\\textbf{{From {reference_system.name}}} \\\\ \\textbf{{to}} }} & \\makecell{{\\textbf{{Average transformation}} \\\\ \\textbf{{matrix}}}} & \\makecell{{\\textbf{{SD}} \\\\ \\textbf{{(Angle/Trans)}}}} \\\\
\\midrule
{all_average_str}
\\bottomrule
\\end{{tabular}}
\\caption{{Average transformation matrix from {reference_system.name} and Standard Deviation (SD) of the rotation angle and translation.}}
\\label{{tab:summary}}
\\end{{table}}

\\end{{document}}
        """

        with open("average_transformations.tex", "w") as file:
            file.write(latex_content)


class PlotHelpers:
    @staticmethod
    def show_axes(axes: np.ndarray, ax: plt.Axes = None, linewidth: float = 1.0):
        """
        Show the axes in the plot.

        Args:
        ax: matplotlib axis
        axes: 4x4 matrix representing the axes
        """

        origin = axes[:3, 3]
        x = axes[:3, 0]
        y = axes[:3, 1]
        z = axes[:3, 2]
        ax.quiver(*origin, *x, color="r", linewidth=linewidth)
        ax.quiver(*origin, *y, color="g", linewidth=linewidth)
        ax.quiver(*origin, *z, color="b", linewidth=linewidth)

    @staticmethod
    def show():
        """
        Easy way to show the plot.
        """
        plt.show()
