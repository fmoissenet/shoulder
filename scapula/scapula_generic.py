import numpy as np

from .matrix_helper import MatrixHelpers


type AxisGeneric = tuple[list[str], list[str]]


class ScapulaJcsGeneric:
    def __init__(self, origin: list[str], axis: AxisGeneric, plane: tuple[AxisGeneric, AxisGeneric]):
        """
        Generic definition of a scapula joint coordinate system. The coordinate system is defined by the origin, the x-axis
        and the z-axis. The y-axis is computed as the cross product of the z-axis and the x-axis (assuming a right-handed
        coordinate system). Then the z-axis is recomputed as the cross product of the x-axis and the y-axis to ensure
        orthogonality. The x-axis is assumed to be medial to lateral, the y-axis is assumed to be posterior to anterior,
        and the z-axis is assumed to be inferior to superior.

        Args:
        origin: the origin of the coordinate system. The mean of all the elements in the list will be taken.
        x: the x-axis of the coordinate system. The mean of all the elements of the first list will be taken as the
        start point, and the mean of all the elements of the second list will be taken as the end point.
        z: the z-axis of the coordinate system. The mean of all the elements of the first list will be taken as the
        start point, and the mean of all the elements of the second list will be taken as the end point.
        """
        self.origin = origin
        self.axis = axis
        self.plane = plane

    def compute_coordinate_system(self, landmarks: dict[str, np.array]) -> np.array:
        """
        Compute the joint coordinate system based on the origin, the x-axis, and the z-axis.

        Returns:
        jcs: the joint coordinate system as a 4x4 matrix
        """

        # Compute the axes and origin
        origin = np.mean([landmarks[name] for name in self.origin], axis=0)[:3]

        axis_start = np.mean([landmarks[name] for name in self.axis[0]], axis=0)[:3]
        axis_end = np.mean([landmarks[name] for name in self.axis[1]], axis=0)[:3]
        axis = axis_end - axis_start

        plane_first_axis_start = np.mean([landmarks[name] for name in self.plane[0][0]], axis=0)[:3]
        plane_first_axis_end = np.mean([landmarks[name] for name in self.plane[0][1]], axis=0)[:3]
        plane_first_axis = plane_first_axis_end - plane_first_axis_start

        plane_second_axis_start = np.mean([landmarks[name] for name in self.plane[1][0]], axis=0)[:3]
        plane_second_axis_end = np.mean([landmarks[name] for name in self.plane[1][1]], axis=0)[:3]
        plane_second_axis = plane_second_axis_end - plane_second_axis_start

        plane_normal = np.cross(plane_first_axis, plane_second_axis)

        return MatrixHelpers.from_vectors(origin=origin, v1=axis, v2=plane_normal, v1_name="x", keep="v1")
