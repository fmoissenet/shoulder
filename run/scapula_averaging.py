import os

import numpy as np
from scapula import Scapula, ScapulaDataType, PlotHelpers, JointCoordinateSystem, MatrixHelpers

# ID | Definition | Positioning strategy
# AA | Acromial angle | First angle higher than 45° from the lateral edge of the acromion
# IA | Inferior angle | Most distal point of the scapula along the medial edge of the scapula
# TS | Root of the scapula spine | Apex of the root triangle, closest to the scapula spine


def get_reference_scapula(filepath: str, use_precomputed_values: bool):
    # Get some remarkable points
    if use_precomputed_values:
        landmarks = {}
        filename = os.path.basename(filepath)

        if filename == "PJ151-M001-scapula.ply":
            landmarks["IA"] = np.array([-0.42450786, 0.12748057, 5.66849068, 1.0])
            landmarks["TS"] = np.array([-0.2877714118750495, 0.20422405338645436, 6.176624433088216, 1.0])
            landmarks["AA"] = np.array([-0.3332845410297929, -0.3215975587141159, 6.231563695676402, 1.0])
            landmarks["GC"] = np.array([-0.18614066393693082, -0.24647324980998891, 6.13033391435741, 1.0])
            landmarks["CP"] = np.array([-0.017074350751475963, -0.2542251571836168, 6.177195252266086, 1.0])
            landmarks["SA"] = np.array([-0.1021418924890583, 0.10170073318175565, 6.2920059467986755, 1.0])
            landmarks["AT"] = np.array([-0.1266968011420527, -0.3688454755778767, 6.265601393258909, 1.0])
        elif filename == "PJ116_scapula_A_avg.ply":
            landmarks["IA"] = np.array([-0.43178419, 0.14275525, 5.68034491, 1.0])
            landmarks["TS"] = np.array([-0.30306839, 0.18366926, 6.15101467, 1.0])
            landmarks["AA"] = np.array([-0.3151479, -0.34203657, 6.25132842, 1.0])
            landmarks["GC"] = np.array([-0.20194872, -0.24786309, 6.10602707, 1.0])
            landmarks["CP"] = np.array([-0.02247475, -0.28800937, 6.14408965, 1.0])
            landmarks["SA"] = np.array([-0.13111922, 0.09036657, 6.28140869, 1.0])
            landmarks["AT"] = np.array([-0.13147937, -0.37142136, 6.28457016, 1.0])
        else:
            raise ValueError(f"Precomputed values for {filename} are not available.")
    else:
        landmarks = None

    # Load the geometry data
    return Scapula.from_landmarks(geometry=filepath, predefined_landmarks=landmarks)


def main():
    # Load the reference scapula
    reference_scapulas = {
        "Statistics": get_reference_scapula(
            filepath="models/scapula/reference/PJ116_scapula_A_avg.ply",
            use_precomputed_values=True,
        ),
        "EOS": get_reference_scapula(
            filepath="models/scapula/reference/PJ151-M001-scapula.ply",
            use_precomputed_values=True,
        ),
    }

    # Sequentially analyse all the scapulas
    scapula_folders = [
        "models/scapula/Scapula-BD-EOS/asymptomatiques/",
        "models/scapula/Scapula-BD-EOS/pathologiques/",
        # "models/scapula/Scapula-BD-FHOrtho/Scapula/",
        50,  # Generate 50 random scapulas
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

    scapulas: dict[str, list[Scapula]] = {}
    for scapula_folder in scapula_folders:
        if isinstance(scapula_folder, str):
            scapula_geometries = os.listdir(scapula_folder)

        elif isinstance(scapula_folder, int):
            scapula_geometries = Scapula.generator(
                models_folder="./models/scapula/Modele_stat/data/",
                number_to_generate=scapula_folder,
                model="P",
                reference_scapula=reference_scapulas["Statistics"],
            )

        else:
            raise ValueError(f"Invalid scapula_folder type: {type(scapula_folder)}")

        for index, geometry in enumerate(scapula_geometries):
            if isinstance(geometry, str):
                print(f"Processing {geometry}")

                # Load the scapula data
                scapula_type = "EOS"
                is_left = geometry in left_scapula_files
                filepath = os.path.join(scapula_folder, geometry)
                scapula = Scapula.from_reference_scapula(
                    geometry=filepath,
                    reference_scapula=reference_scapulas[scapula_type],
                    shared_indices_with_reference="Scapula-BD-EOS" in scapula_folder,
                    is_left=is_left,
                )

            elif isinstance(geometry, Scapula):
                print(f"Generating random scapula {index}")
                scapula_type = "Statistics"
                scapula = geometry

            else:
                raise ValueError(f"Invalid geometry type: {type(geometry)}")

            # scapula.plot_geometry(
            #     ax=reference_scapulas[scapula_type].plot_geometry(
            #         show_now=False, marker="o", color="b", s=5, alpha=0.1
            #     ),
            #     data_type=ScapulaDataType.LOCAL,
            #     show_jcs=[JointCoordinateSystem.ISB, JointCoordinateSystem.O_GC__X_TS_AA__Y_IA_TS_AA],
            #     show_now=True,
            #     color="r",
            # )

            if scapula_type not in scapulas:
                scapulas[scapula_type] = []

            scapulas[scapula_type].append(scapula)
            # TODO Use pointing method if STL are used (not morphing the reference scapula)
            # TODO Save the values so they can be reused based on their respective file_path

    # Compute the average reference system
    average_rts = {}
    for key in scapulas.keys():
        average_rts[key] = {}
        for type in JointCoordinateSystem:
            average_rts[key][type.name] = Scapula.compute_average_reference_system_from_reference(
                scapulas[key], type, reference_system=JointCoordinateSystem.ISB
            )

    # Export to LaTeX
    for key in average_rts.keys():
        PlotHelpers.export_average_matrix_to_latex(
            f"average_transformations_{key}.tex", average_rts[key], reference_system=JointCoordinateSystem.ISB
        )

        rt_reference = reference_scapulas[key].get_joint_coordinates_system(JointCoordinateSystem.DUMMY)
        PlotHelpers.export_average_matrix_to_latex(
            f"average_transformations_{key}_reference.tex",
            {"Reference": [rt_reference, [0, 0]]},
            reference_system=JointCoordinateSystem.ISB,
        )

    # Plot the results
    reference_scapula = reference_scapulas["Statistics"]

    # Print the angles between the reference scapula and the average scapula to the console
    ref_rt = reference_scapula.get_joint_coordinates_system(JointCoordinateSystem.DUMMY)
    angles = {
        key: MatrixHelpers.angle_between_rotations(ref_rt, average_rts[key]["DUMMY"][0]) for key in average_rts.keys()
    }
    PlotHelpers.export_error_angles_to_latex("angles.tex", angles, reference_name="Statistics")

    # Plot all the scapula rt
    # for key in scapulas.keys():
    #     Scapula.plot_systems_in_reference_scapula(reference_scapula, scapulas[key], JointCoordinateSystem.DUMMY)

    # Plot the scapulas in the reference scapula
    ax = reference_scapula.plot_geometry(show_now=False, marker="o", color="b", s=5, alpha=0.1)
    origin = reference_scapula.get_joint_coordinates_system(JointCoordinateSystem.DUMMY)[:3, 3]

    linestyles = {"Statistics": "--", "EOS": "-"}
    for key in average_rts.keys():
        PlotHelpers.show_axes(
            reference_scapulas[key].get_joint_coordinates_system(JointCoordinateSystem.DUMMY),
            ax=ax,
            translate_to=origin,
            linestyle=linestyles[key],
        )
        PlotHelpers.show_axes(
            average_rts[key]["DUMMY"][0], ax=ax, linewidth=5, translate_to=origin, linestyle=linestyles[key]
        )

    ax.set_xlim(-0.8, 1.1)
    ax.set_ylim(-0.8, 1.1)
    ax.set_zlim(-0.8, 1.1)
    PlotHelpers.show()


if __name__ == "__main__":
    main()
