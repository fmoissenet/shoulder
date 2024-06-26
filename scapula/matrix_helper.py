import numpy as np


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
    def from_vectors(
        origin: np.ndarray, v1: np.ndarray, v2: np.ndarray, v1_name: str, v2_name: str, keep: str = "v1"
    ) -> np.array:
        """
        Compute the homogeneous matrix on the origin, the v1 vector, and the v2 vector, with v3 = v1 ^ v2 and either
        v1 or v2 recomputed (depending on the keep parameter). The v1 axis gives the name of all the axes, that is
        if v1_name is "x", then v2 will be "y" and v3 will be "z", if v1_name is "y", then v2 will be "z" and v3 will
        be "x", and if v1_name is "z", then v2 will be "x" and v3 will be "y".

        Args:
        origin: numpy array of shape (3,) representing the origin
        v1: numpy array of shape (3,) representing the first vector
        v2: numpy array of shape (3,) representing the second vector
        keep: string representing which vector to keep, either "v1" or "v2"

        Returns:
        out: 4x4 matrix representing the homogeneous matrix
        """
        v3 = np.cross(v1, v2)
        if keep == "v1":
            v2 = np.cross(v3, v1)
        elif keep == "v2":
            v1 = np.cross(v2, v3)
        else:
            raise ValueError("The keep parameter must be either 'v1' or 'v2'.")

        if v1_name == "x" and v2_name == "y":
            x, y, z = v1, v2, v3
        elif v1_name == "x" and v2_name == "z":
            x, z, y = v1, v2, v3
        elif v1_name == "y" and v2_name == "z":
            y, z, x = v1, v2, v3
        elif v1_name == "y" and v2_name == "x":
            y, x, z = v1, v2, v3
        elif v1_name == "z" and v2_name == "x":
            z, x, y = v1, v2, v3
        elif v1_name == "z" and v2_name == "y":
            z, y, x = v1, v2, v3
        else:
            raise ValueError("The v1_name parameter must be either 'x', 'y', or 'z'.")

        # Normalize the axes
        x /= np.linalg.norm(x)
        y /= np.linalg.norm(y)
        z /= np.linalg.norm(z)

        # Create the joint coordinate system
        jcs = np.eye(4)
        jcs[:3, 0] = x
        jcs[:3, 1] = y
        jcs[:3, 2] = z
        jcs[:3, 3] = origin

        return jcs

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
        initial_rt: np.ndarray = np.eye(4),
        return_points_error: bool = False,
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
        initial_rt: initial transformation matrix
        return_points_error: whether to return the points error or not

        Returns:
        aligned_points: numpy array of shape (3, N) representing the aligned first point cloud
        points_error: a float representing the norm of the error between the aligned points and the second point cloud
        """

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
        rt = initial_rt
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

        if return_points_error:
            pts1 = (rt @ pts1_zeroed[:, ::pts1_slice_jumps])[:3, :]
            points_error_squared, _ = MatrixHelpers.nearest_neighbor(pts1, pts2)
            points_error = np.sqrt(np.mean(points_error_squared))

        # Reproject to the initial pose of the first point cloud
        rt[:3, 3] -= (rt[:3, :3] @ pts1_mean[:3, :])[:3, 0]

        if return_points_error:
            return rt, points_error
        return rt

    @staticmethod
    def from_euler(angles, sequence, homogenous: bool = False):
        """
        Create a rotation matrix from Euler angles.

        Args:
        angles: numpy array of shape (3,) representing the Euler angles
        sequence: string representing the sequence of the Euler angles
        homogenous: whether to return a homogenous matrix (that is a 4x4 matrix) or a rotation matrix (3x3 matrix)

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

        if homogenous:
            out = np.eye(4)
            out[:3, :3] = rotation
        return out

    def average_matrices(matrices: list[np.ndarray]) -> np.ndarray:
        """
        Compute the average of a list of matrices. If compute_std is True, also compute the standard deviation of the
        rotation matrices. This is done by computing the standard deviation of the angle of the rotation matrices. To
        compute the angle, we use the formula:
        angle = arccos((trace(R1.T @ R2) - 1) / 2)
        where "R1.T @ R2" is the rotation matrix between a matrix and the average matrix.

        Args:
        matrices: list of numpy arrays of shape (4, 4) representing the homogenous matrices to average

        Returns:
        average: numpy array of shape (4, 4) representing the average matrix
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

        return out

    def angle_between_rotations(rotations: list[np.ndarray], rotation_reference: np.ndarray) -> np.ndarray:
        """
        Compute the angle between two rotations, following the formula:
        angle = arccos((trace(R1.T @ R2) - 1) / 2)

        Args:
        rotations: list of numpy arrays of shape (3, 3) representing the first rotation matrices
        rotation_reference: numpy array of shape (3, 3) representing the second rotation matrix

        Returns:
        angles: numpy array of shape (N,) representing the angles between the rotations to the second rotation matrix
        """
        return np.array([np.arccos((np.trace(rt[:3, :3].T @ rotation_reference[:3, :3]) - 1) / 2) for rt in rotations])

    def distance_between_origins(homogeneous: list[np.ndarray], homogeneous_reference: np.ndarray) -> np.ndarray:
        """
        Compute the euclidean distance between the origins of a list of homogeneous matrices and a reference homogeneous

        Args:
        homogeneous: list of numpy arrays of shape (4, 4) representing the homogeneous matrices
        homogeneous_reference: numpy array of shape (4, 4) representing the homogeneous matrix

        Returns:
        distances: numpy array of shape (N,) representing the distances between the origins
        """
        return np.array([np.linalg.norm(rt[:3, 3] - homogeneous_reference[:3, 3]) for rt in homogeneous])

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
        h = centered1 @ centered2.T

        # Singular Value Decomposition
        u, _, v_T = np.linalg.svd(h)

        # Rotation matrix
        r = (u @ v_T).T

        # Translation vector
        t = centroid2 - r @ centroid1

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
