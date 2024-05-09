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

    O_AA__X_TS_AA__Y_IA_TS_AA = ScapulaJcsGeneric(origin=["AA"], x=(["TS"], ["AA"]), z=(["IA"], ["TS"]))
    O_GC__X_TS_AA__Y_IA_TS_AA = ScapulaJcsGeneric(origin=["GC"], x=(["TS"], ["AA"]), z=(["IA"], ["TS"]))
    DUMMY = ScapulaJcsGeneric(origin=["GC"], x=(["IA"], ["AA"]), z=(["TS"], ["SA"]))
    ISB = O_AA__X_TS_AA__Y_IA_TS_AA

    def __call__(self, landmarks: dict[str, np.array]) -> np.array:
        return self.value.compute_coordinate_system(landmarks)
