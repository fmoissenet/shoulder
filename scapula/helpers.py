import copy

import numpy as np
import matplotlib.pyplot as plt


class DataHelpers:
    @staticmethod
    def rough_normalize(data: np.ndarray) -> np.ndarray:
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
        out = np.eye(4)
        out[:3, :3] = homogenous_matrix[:3, :3].transpose()
        out[:3, 3] = -out[:3, :3] @ homogenous_matrix[:3, 3]
        return out

    @staticmethod
    def subtract_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
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

        def nearest_neighbor(src, dst):
            """
            Find the nearest (Euclidean distance) neighbor in dst for each point in src.

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

        def compute_transformation(points1, points2):
            """
            Compute the transformation (R, t) that aligns points1 to points2.

            Args:
            points1: numpy array of shape (3, N) representing the source point cloud
            points2: numpy array of shape (3, N) representing the destination point cloud

            Returns:
            R: 3x3 rotation matrix
            t: 3x1 translation vector
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
            R = np.dot(Vt.T, U.T)

            # Translation vector
            t = centroid2 - np.dot(R, centroid1)

            return R, t

        # Initial transformation (identity)
        r = np.eye(3)
        t = np.zeros((3, 1))

        # Copy the points
        pts1_mean = np.concatenate([np.mean(points1[:3, :], axis=1, keepdims=True), [[1]]])
        pts1_zeroed = MatrixHelpers.subtract_vectors(points1, pts1_mean)

        # Iterate
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
                __, indices = nearest_neighbor(pts1, pts2)

            # Compute the transformation
            r, t = compute_transformation(pts1, pts2[:, indices])
            rt_step = np.concatenate([np.concatenate([r, t], axis=1), [[0, 0, 0, 1]]])
            rt = rt_step @ rt

            # Check convergence
            squared_error = np.sum((np.eye(4) - rt_step) ** 2)
            if np.abs(prev_error - squared_error) < tolerance:
                break
            prev_error = squared_error

        # Reproject to the initial pose of the first point cloud
        rt[:3, 3] -= (rt[:3, :3] @ pts1_mean[:3, :])[:3, 0]
        return rt


class PlotHelpers:
    @staticmethod
    def show_axes(ax, axes: np.ndarray):
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
        ax.quiver(*origin, *x, color="r")
        ax.quiver(*origin, *y, color="g")
        ax.quiver(*origin, *z, color="b")

    @staticmethod
    def show():
        plt.show()
