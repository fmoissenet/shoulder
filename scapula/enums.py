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
    ISB = ScapulaJcsGeneric(origin=["AA"], axis=(["TS"], ["AA"]), plane=((["AI"], ["TS"]), (["AI"], ["AA"])))
    SCS2 = ScapulaJcsGeneric(origin=["AA"], axis=(["TS"], ["AA"]), plane=((["AI"], ["TS"]), (["AI"], ["AA"])))
    SCS3 = ScapulaJcsGeneric(origin=["AC"], axis=(["TS"], ["AC"]), plane=((["AI"], ["TS"]), (["AI"], ["AC"])))
    SCS4 = ScapulaJcsGeneric(origin=["GC"], axis=(["TS"], ["GC"]), plane=((["AI"], ["TS"]), (["AI"], ["GC"])))
    SCS5 = ScapulaJcsGeneric(origin=["TS"], axis=(["TS"], ["AC"]), plane=((["AI"], ["TS"]), (["AI"], ["AC"])))
    SCS6 = ScapulaJcsGeneric(origin=["AC"], axis=(["TS"], ["AC"]), plane=((["AI"], ["TS"]), (["AI"], ["AC"])))
    # SCS7 = # TODO
    # SCS8 = # TODO
    # SCS9 = # TODO
    # SCS10 = # TODO

    def __call__(self, landmarks: dict[str, np.array]) -> np.array:
        return self.value.compute_coordinate_system(landmarks)
