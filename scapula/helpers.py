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
    def icp(points1, points2, max_iterations=100, tolerance=1e-4):
        """
        Perform ICP to align points1 to points2.

        Args:
        points1: numpy array of shape (3, N) representing the first point cloud
        points2: numpy array of shape (3, M) representing the second point cloud
        max_iterations: maximum number of iterations for ICP
        tolerance: convergence threshold

        Returns:
        R: 3x3 rotation matrix
        t: 3x1 translation vector
        """
        points2 = points2[:3, ::5]

        # Initial transformation (identity)
        r = np.eye(3)
        t = np.zeros((3, 1))
        print(points1.shape)
        print(points2.shape)
        print("")

        # Copy the points
        points1_transformed = points1[:3, :] - np.mean(points1[:3, :], axis=1, keepdims=True)
        pts2 = points2[:3, :]
        prev_error = np.inf
        final_r = np.eye(3)
        final_t = np.zeros((3, 1))

        for _ in range(max_iterations):
            # Find the nearest neighbors
            # if points1.shape[1] == points2.shape[1]:
            #     indices = np.arange(points1.shape[1])
            # else:
            __, indices = MatrixHelpers._nearest_neighbor(points1_transformed, pts2)
            # squared_error = np.mean(distance_squared)

            # Compute the transformation
            r, t = MatrixHelpers._compute_transformation(points1_transformed, pts2[:, indices])
            squared_error = np.sum((np.eye(3) - r) ** 2) + np.sum(t**2)
            final_r = final_r @ r
            final_t = final_t + t

            # Apply the transformation
            points1_transformed = r @ points1_transformed + t

            # Check convergence
            if np.abs(prev_error - squared_error) < tolerance:
                break
            prev_error = squared_error

        rt = np.eye(4)
        rt[:3, :3] = final_r
        rt[:3, 3] = final_t[:, 0]
        # TODO Find the RT
        return rt @ points1

    @staticmethod
    def _nearest_neighbor(src, dst):
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

    @staticmethod
    def _compute_transformation(points1, points2):
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


class PlotHelpers:
    @staticmethod
    def show_axes(ax, lcs):
        origin = lcs[:3, 3]
        x = lcs[:3, 0]
        y = lcs[:3, 1]
        z = lcs[:3, 2]
        ax.quiver(*origin, *x, color="r")
        ax.quiver(*origin, *y, color="g")
        ax.quiver(*origin, *z, color="b")

    @staticmethod
    def show():
        plt.show()
