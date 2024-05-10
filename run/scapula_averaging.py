import os

import numpy as np
from scapula import Scapula, ScapulaDataType, PlotHelpers, JointCoordinateSystem, MatrixHelpers


def get_reference_scapula(filepath: str, use_precomputed_values: bool):
    # Get some remarkable points
    if use_precomputed_values:
        landmarks = {}
        landmarks["IA"] = np.array([-0.42450786, 0.12748057, 5.66849068, 1.0])
        landmarks["TS"] = np.array([-0.2877714118750495, 0.20422405338645436, 6.176624433088216, 1.0])
        landmarks["AA"] = np.array([-0.3332845410297929, -0.3215975587141159, 6.231563695676402, 1.0])
        landmarks["GC"] = np.array([-0.18614066393693082, -0.24647324980998891, 6.13033391435741, 1.0])
        landmarks["CP"] = np.array([-0.017074350751475963, -0.2542251571836168, 6.177195252266086, 1.0])
        landmarks["SA"] = np.array([-0.1021418924890583, 0.10170073318175565, 6.2920059467986755, 1.0])
        landmarks["AT"] = np.array([-0.1266968011420527, -0.3688454755778767, 6.265601393258909, 1.0])
    else:
        landmarks = None

    # Load the geometry data
    return Scapula.from_landmarks(filepath=filepath, predefined_landmarks=landmarks)


def average_reference_systems(
    reference_scapula: Scapula, scapulas: list[Scapula], jcs_type: JointCoordinateSystem, show: bool = True
):
    # Do not care about translation, so set the origin of all the reference frames to the same point
    reference_origin = reference_scapula.get_joint_coordinates_system(jcs_type)[:3, 3]
    all_rt = []
    for scapula in scapulas:
        # scapula = reference_scapula
        rt = scapula.get_joint_coordinates_system(jcs_type)
        rt[:3, 3] = reference_origin
        all_rt.append(rt)

    average_matrix = MatrixHelpers.average_matrices(all_rt)

    if show:
        # Plot the average scapula with all the axes
        ax = reference_scapula.plot_geometry(show_now=False, marker="o", color="b", s=5, alpha=0.1, show_jcs=[jcs_type])
        for rt in all_rt:
            PlotHelpers.show_axes(rt, ax=ax)
        PlotHelpers.show_axes(axes=average_matrix, ax=ax, linewidth=10)
        PlotHelpers.show()

    return average_matrix


def main():
    # Load the reference scapula
    reference_scapula = get_reference_scapula(
        filepath="models/scapula/reference/PJ151-M001-scapula.ply", use_precomputed_values=True
    )

    # # Plot for debug
    # for key, value in reference_scapula.landmarks().items():
    #     print(f'landmarks["{key}"] = np.array({value.tolist()})')
    # ax = reference_scapula.plot_geometry(show_now=True, marker="o", color="b", s=5, alpha=0.1)

    # Sequentially analyse all the scapulas
    scapula_folder = "models/scapula/Scapula-BD-EOS/asymptomatiques/"
    scapula_files = os.listdir(scapula_folder)

    left_scapula_files = ["PJ151-M021-scapula.ply", "PJ151-M015-scapula.ply"]

    scapulas: list[Scapula] = []
    for file in scapula_files:
        print(f"Processing {file}")

        if file in [
            "PJ151-M008-scapula.ply",
            "PJ151-M027-scapula.ply",
            "PJ151-M010-scapula.ply",
            "PJ151-M023-scapula.ply",
        ]:
            print("To validate")

        # Load the scapula data
        is_left = file in left_scapula_files
        filepath = os.path.join(scapula_folder, file)
        scapula = Scapula.from_reference_scapula(
            filepath=filepath, reference_scapula=reference_scapula, shared_indices_with_reference=True, is_left=is_left
        )

        # scapula.plot_geometry(
        #     data_type=ScapulaDataType.LOCAL,
        #     show_jcs=[JointCoordinateSystem.ISB, JointCoordinateSystem.O_GC__X_TS_AA__Y_IA_TS_AA],
        #     show_now=True,
        #     color="r",
        # )

        # TODO Automatically find the distance between corresponding indices to see if they match

        scapulas.append(scapula)

    # Compute the average reference system
    average_matrix = average_reference_systems(reference_scapula, scapulas, JointCoordinateSystem.DUMMY, show=True)

    # TODO Compute the "standard deviation" to the average matrices (variability)
    # TODO Show the results in a table


if __name__ == "__main__":
    main()
