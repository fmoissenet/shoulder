from circle_fitting_3d import Circle3D
import numpy as np

from .matrix_helper import MatrixHelpers


type AxisGeneric = tuple[list[str], list[str]]


class ScapulaJcsGeneric:
    def __init__(
        self,
        origin: list[str],
        axis: AxisGeneric,
        axis_name: str,
        plane: tuple[AxisGeneric, AxisGeneric],
        plane_name: str,
        keep: str,
    ):
        """
        Generic definition of a scapula joint coordinate system. The coordinate system is defined by the origin, the x-axis
        and the z-axis. The y-axis is computed as the cross product of the z-axis and the x-axis (assuming a right-handed
        coordinate system). Then the z-axis is recomputed as the cross product of the x-axis and the y-axis to ensure
        orthogonality. The x-axis is assumed to be medial to lateral, the y-axis is assumed to be posterior to anterior,
        and the z-axis is assumed to be inferior to superior.

        Args:
        origin: the origin of the coordinate system. The mean of all the elements in the list will be taken.
        axis: the axis of the coordinate system. The mean of all the elements in the list will be taken.
        axis_name: the name of the axis
        plane: the plane of the coordinate system. The mean of all the elements in the list will be taken.
        plane_name: the name of the plane
        keep: "axis" or "plane" to keep the specified axis or plane as is
        """
        self.origin = origin
        self.axis = axis
        self.axis_name = axis_name
        self.plane = plane
        self.plane_name = plane_name
        if keep not in ["axis", "plane"]:
            raise ValueError("keep must be 'axis' or 'plane'")
        self.keep = keep

    def compute_coordinate_system(self, landmarks: dict[str, np.array]) -> np.array:
        """
        Compute the joint coordinate system based on the origin, the x-axis, and the z-axis.

        Returns:
        jcs: the joint coordinate system as a 4x4 matrix
        """

        # Compute the axes and origin
        origin = np.mean([landmarks[name] for name in self.origin], axis=0)
        if len(origin.shape) == 2:
            origin = origin[:, 0]
        origin = origin[:3]

        axis_start = np.mean([landmarks[name] for name in self.axis[0]], axis=0)
        axis_end = np.mean([landmarks[name] for name in self.axis[1]], axis=0)
        axis = axis_end - axis_start
        if len(axis.shape) == 2:
            axis = axis[:, 0]
        axis = axis[:3]

        if self.plane == "GC_CONTOUR_NORMAL":
            plane_normal = landmarks["GC_CONTOUR_NORMAL"][:3, 0] - landmarks["GC_CONTOUR_CENTER"][:3, 0]
        else:
            plane_first_axis_start = np.mean([landmarks[name] for name in self.plane[0][0]], axis=0)
            plane_first_axis_end = np.mean([landmarks[name] for name in self.plane[0][1]], axis=0)
            plane_first_axis = plane_first_axis_end - plane_first_axis_start
            if len(plane_first_axis.shape) == 2:
                plane_first_axis = plane_first_axis[:, 0]
            plane_first_axis = plane_first_axis[:3]

            plane_second_axis_start = np.mean([landmarks[name] for name in self.plane[1][0]], axis=0)
            plane_second_axis_end = np.mean([landmarks[name] for name in self.plane[1][1]], axis=0)
            plane_second_axis = plane_second_axis_end - plane_second_axis_start
            if len(plane_second_axis.shape) == 2:
                plane_second_axis = plane_second_axis[:, 0]
            plane_second_axis = plane_second_axis[:3]

            plane_normal = np.cross(plane_first_axis, plane_second_axis)

        if (
            (self.axis_name == "x" and self.plane_name == "y")
            or (self.axis_name == "y" and self.plane_name == "z")
            or (self.axis_name == "z" and self.plane_name == "x")
        ):
            v1, v2 = axis, plane_normal
            v1_name, v2_name = self.axis_name, self.plane_name
            keep = "v1" if self.keep == "axis" else "v2"
        else:
            v1, v2 = plane_normal, axis
            v1_name, v2_name = self.plane_name, self.axis_name
            keep = "v2" if self.keep == "plane" else "v1"

        return MatrixHelpers.from_vectors(origin=origin, v1=v1, v2=v2, v1_name=v1_name, v2_name=v2_name, keep=keep)
