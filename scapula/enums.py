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

    # O_AA__X_TS_AA__Y_IA_TS_AA = ScapulaJcsGeneric(origin=["AA"], x=(["TS"], ["AA"]), z=(["IA"], ["TS"]))
    # O_GC__X_TS_AA__Y_IA_TS_AA = ScapulaJcsGeneric(origin=["GC"], x=(["TS"], ["AA"]), z=(["IA"], ["TS"]))
    # DUMMY = ScapulaJcsGeneric(origin=["GC"], x=(["IA"], ["AA"]), z=(["TS"], ["SA"]))
    ISB = ScapulaJcsGeneric(origin=["AA"], x=(["TS"], ["AA"]), z=(["IA"], ["TS"]))
    FIGURE_A = ScapulaJcsGeneric(origin=["AA"], x=(["TS"], ["AA"]), z=(["IA"], ["TS"]))
    FIGURE_B1 = ScapulaJcsGeneric(origin=["AA"], x=(["TS"], ["AC"]), z=(["IA"], ["AC"]))
    FIGURE_B2 = ScapulaJcsGeneric(origin=["GC"], x=(["TS"], ["GC"]), z=(["IA"], ["GC"]))
    FIGURE_C1 = ScapulaJcsGeneric(origin=["TS"], x=(["TS"], ["AC"]), z=(["IA"], ["TS"]))
    FIGURE_C2 = ScapulaJcsGeneric(origin=["AC"], x=(["TS"], ["AC"]), z=(["IA"], ["AC"]))
    FIGURE_D = ScapulaJcsGeneric(origin=["GC"], x=(["TS"], ["AA"]), z=(["IE"], ["SE"]))

    def __call__(self, landmarks: dict[str, np.array]) -> np.array:
        return self.value.compute_coordinate_system(landmarks)
