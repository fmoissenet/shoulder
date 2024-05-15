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


def main():
    # Load the reference scapula
    reference_scapula = get_reference_scapula(
        filepath="models/scapula/reference/PJ151-M001-scapula.ply",
        use_precomputed_values=True,
        # filepath="models/scapula/Scapula-BD-FHOrtho/Scapula/PJ151-M081-scapula.stl",
        # use_precomputed_values=False,
    )

    # # Plot for debug
    # for key, value in reference_scapula.landmarks().items():
    #     print(f'landmarks["{key}"] = np.array({value.tolist()})')
    # ax = reference_scapula.plot_geometry(show_now=True, marker="o", color="b", s=5, alpha=0.1)

    # Sequentially analyse all the scapulas
    scapula_folders = [
        # "models/scapula/Scapula-BD-EOS/asymptomatiques/",
        # "models/scapula/Scapula-BD-EOS/pathologiques/",
        "models/scapula/Scapula-BD-FHOrtho/Scapula/",
    ]

    left_scapula_files = [
        "PJ151-M021-scapula.ply",
        "PJ151-M015-scapula.ply",
        "PJ151-M079-scapula.ply",
        "PJ151-M047-scapula.ply",
        "PJ151-M070-scapula.ply",
        "PJ151-M077-scapula.ply",
        "PJ151-M046-scapula.ply",
        "PJ151-M031-scapula.ply",
        "PJ151-M072-scapula.ply",
        "PJ151-M068-scapula.ply",
        "PJ151-M041-scapula.ply",
        "PJ151-M054-scapula.ply",
        "PJ151-M058-scapula.ply",
        "PJ151-M067-scapula.ply",
        "PJ151-M051-scapula.ply",
        "PJ151-M040-scapula.ply",
        "PJ151-M049-scapula.ply",
        "PJ151-M033-scapula.ply",
        "PJ151-M059-scapula.ply",
        "PJ151-M078-scapula.ply",
        "PJ151-M084-scapula.stl",
        "PJ151-M090-scapula.stl",
    ]

    failing_files = [
        "PJ151-M082-scapula.stl",
        "PJ151-M086-scapula.stl",
    ]

    scapulas: list[Scapula] = []
    for scapula_folder in scapula_folders:
        scapula_files = os.listdir(scapula_folder)
        for file in scapula_files:
            print(f"Processing {file}")

            # Load the scapula data
            is_left = file in left_scapula_files
            filepath = os.path.join(scapula_folder, file)
            scapula = Scapula.from_reference_scapula(
                filepath=filepath,
                reference_scapula=reference_scapula,
                shared_indices_with_reference="Scapula-BD-EOS" in scapula_folder,
                is_left=is_left,
            )

            scapula.plot_geometry(
                ax=reference_scapula.plot_geometry(show_now=False, marker="o", color="b", s=5, alpha=0.1),
                data_type=ScapulaDataType.LOCAL,
                show_jcs=[JointCoordinateSystem.ISB, JointCoordinateSystem.O_GC__X_TS_AA__Y_IA_TS_AA],
                show_now=True,
                color="r",
            )

            scapulas.append(scapula)

    # Compute the average reference system
    rts = {}
    for type in JointCoordinateSystem:
        rts[type.name] = Scapula.compute_average_reference_system_from_reference(
            scapulas, type, reference_system=JointCoordinateSystem.ISB
        )
    Scapula.plot_systems_in_reference_scapula(reference_scapula, scapulas, JointCoordinateSystem.DUMMY)

    MatrixHelpers.export_average_to_latex(rts, reference_system=JointCoordinateSystem.ISB)


if __name__ == "__main__":
    main()
