import copy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button


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
        out = np.zeros((4, 4))
        out[:3, :3] = homogenous_matrix[:3, :3].transpose()
        out[:3, 3] = -out[:3, :3] @ homogenous_matrix[:3, 3]
        return out

    @staticmethod
    def icp(points1, points2, max_iterations=100, tolerance=1e-5):
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

        # Initial transformation (identity)
        R = np.eye(3)
        t = np.zeros((3, 1))

        # Copy the points
        points1_transformed = copy.deepcopy(points1)

        for _ in range(max_iterations):
            # Find the nearest neighbors
            distances, indices = MatrixHelpers._nearest_neighbor(points1_transformed, points2)

            # Compute the transformation
            R, t = MatrixHelpers._compute_transformation(points1_transformed, points2[:, indices])

            # Apply the transformation
            points1_transformed = np.dot(R, points1_transformed) + t

            # Check convergence
            if np.sum(distances) < tolerance:
                break

        return R, t

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
        distances = np.zeros(src.shape[1])
        indices = np.zeros(src.shape[1], dtype=int)

        for i in range(src.shape[1]):
            distances[i] = np.min(np.linalg.norm(dst - src[:, i][:, None], axis=0))
            indices[i] = np.argmin(np.linalg.norm(dst - src[:, i][:, None], axis=0))

        return distances, indices

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
    def pickable_scapula_geometry_plot(points_name: list[str], data: np.ndarray) -> dict[str, np.array]:
        # Prepare the figure
        fig = plt.figure(f"Pick the points")
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(data[0, :], data[1, :], data[2, :], ".", s=1, picker=5, alpha=0.3)
        scatter = ax.scatter(np.nan, np.nan, np.nan, ".", c="r", s=50)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_box_aspect([1, 1, 1])

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

    @staticmethod
    def show_axes(ax, lcs):
        origin = lcs[:3, 3]
        x = lcs[:3, 0]
        y = lcs[:3, 1]
        z = lcs[:3, 2]
        ax.quiver(*origin, *x, color="r")
        ax.quiver(*origin, *y, color="g")
        ax.quiver(*origin, *z, color="b")
