import numpy as np
import matplotlib.pyplot as plt


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

    def average_matrices(matrices: list[np.ndarray]) -> np.ndarray:
        """
        Compute the average of a list of matrices.

        Args:
        matrices: numpy array of shape (N, 4, 4) representing the list of matrices

        Returns:
        average: numpy array of shape (4, 4) representing the average matrix
        """

        # Dispatch the input matrices
        rotations = np.array([np.array(mat[:3, :3]).T for mat in matrices]).T

        # FOR DEBUG
        # a = MatrixHelpers.from_euler(np.random.random((3,)) * 0.2, "xyz")
        # b = MatrixHelpers.from_euler(np.random.random((3,)) * 0.3, "xyz")
        # c = MatrixHelpers.from_euler(np.random.random((3,)) * 0.1, "xyz")
        # rotations = np.array([[a.T, a.T, b.T, b.T, c.T, c.T]]).squeeze().T

        # Compute the average of the rotation matrices
        rotations_mean = np.mean(rotations, axis=2)
        u, s, v_T = np.linalg.svd(rotations_mean)
        average_rotation = np.dot(u, v_T)

        # Compute a "standard deviation"-like value
        errors1 = []
        errors2 = []
        for i in range(rotations.shape[2]):
            # These next two lines are strictly equivalent
            # u, s, v_T = np.linalg.svd(np.eye(3) - rotations[:, :, i].T @ average_rotation)
            u, s, v_T = np.linalg.svd(rotations[:, :, i] - average_rotation)
            errors1.append(s.T @ s)

            error = np.abs(np.arccos((np.trace(rotations[:, :, i].T @ average_rotation) - 1) / 2))
            errors2.append(error)

        std = np.sqrt(np.mean(errors1))
        print(f"Standard deviation (method 1) = {std} (unit?)")

        std = np.sqrt(np.mean(np.array(errors2) ** 2))
        print(f"Standard deviation (method 2) = {std} radian")
        print(f"Standard deviation (method 2) = {std * 180 / np.pi} degrees")

        # Compute the average of the translation vectors
        translations = np.array([np.array(mat[:3, 3]).T for mat in matrices]).T
        average_translation = np.mean(translations, axis=1)

        # Create the average matrix
        out = np.eye(4)
        out[:3, :3] = average_rotation
        out[:3, 3] = average_translation

        return out

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
