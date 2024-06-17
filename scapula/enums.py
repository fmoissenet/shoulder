from enum import Enum

import numpy as np

from .scapula_generic import ScapulaJcsGeneric


class ScapulaDataType(Enum):
    RAW = 1
    RAW_NORMALIZED = 2
    LOCAL = 3


class JointCoordinateSystem(Enum):
    """
    Enum that defines the joint coordinate systems for the scapula. The coordinate systems are defined by the origin,
    the x-axis and the y-plane.
    """

    SCS1 = ScapulaJcsGeneric(
        origin=["AA"],
        axis=(["TS"], ["AA"]),
        axis_name="z",
        plane=((["AI"], ["TS"]), (["AI"], ["AA"])),
        plane_name="x",
        keep="axis",
    )
    SCS2 = ScapulaJcsGeneric(
        origin=["AA"],
        axis=(["TS"], ["AA"]),
        axis_name="x",
        plane=((["AI"], ["TS"]), (["AI"], ["AA"])),
        plane_name="y",
        keep="axis",
    )
    SCS3 = ScapulaJcsGeneric(
        origin=["AC"],
        axis=(["TS"], ["AA"]),
        axis_name="x",
        plane=((["AI"], ["TS"]), (["AI"], ["AA"])),
        plane_name="y",
        keep="axis",
    )
    SCS4 = ScapulaJcsGeneric(
        origin=["AC"],
        axis=(["TS"], ["AC"]),
        axis_name="z",
        plane=((["AI"], ["TS"]), (["AI"], ["AC"])),
        plane_name="x",
        keep="axis",
    )
    SCS5 = ScapulaJcsGeneric(
        origin=["GC_CIRCLE_CENTER"],
        axis=(["TS"], ["GC_CIRCLE_CENTER"]),
        axis_name="z",
        plane=((["AI"], ["TS"]), (["AI"], ["GC_CIRCLE_CENTER"])),
        plane_name="x",
        keep="axis",
    )
    SCS6 = ScapulaJcsGeneric(
        origin=["TS"],
        axis=(["TS"], ["AC"]),
        axis_name="x",
        plane=((["AI"], ["TS"]), (["AI"], ["AC"])),
        plane_name="y",
        keep="axis",
    )
    SCS7 = ScapulaJcsGeneric(
        origin=["AC"],
        axis=(["TS"], ["AC"]),
        axis_name="x",
        plane=((["AI"], ["TS"]), (["AI"], ["AC"])),
        plane_name="y",
        keep="axis",
    )
    SCS8 = ScapulaJcsGeneric(
        origin=["GC_MID"],
        axis=(["IE"], ["SE"]),
        axis_name="y",
        plane="GC_NORMAL",
        plane_name="z",
        keep="axis",
    )
    SCS9 = ScapulaJcsGeneric(
        origin=["GC_ELLIPSE_CENTER"],
        axis="GC_ELLIPSE_MAJOR",
        axis_name="z",
        plane="GC_NORMAL",
        plane_name="y",
        keep="axis",
    )
    SCS10 = ScapulaJcsGeneric(
        origin=["GC_MID"],
        axis=[["IE"], ["SE"]],
        axis_name="z",
        plane="GC_NORMAL",
        plane_name="x",
        keep="axis",
    )
    SCS11 = ScapulaJcsGeneric(
        origin=["GC_MID"],
        axis=(["IE"], ["SE"]),
        axis_name="x",
        plane="GC_NORMAL",
        plane_name="z",
        keep="plane",
    )

    def __call__(self, landmarks: dict[str, np.array]) -> np.array:
        return self.value.compute_coordinate_system(landmarks)
