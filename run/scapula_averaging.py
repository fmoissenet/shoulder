import os

import numpy as np
from scapula import Scapula, ScapulaDataType, PlotHelpers, JointCoordinateSystem, MatrixHelpers

from models.scapula.reference_scapula import get_reference_scapula


# TODO: The SCS8 and SCS10 don't fully conform to the definition as "Glenoid posteroanterior axis" cannot be defined yet
# TODO: The defition of SCS9 is defined differently, but should give the same results as the plane and the major/minor axis are all perpendicular


def main():
    #### OPTIONS ####
    skip = []  # ["EOS"]
    base_folder = "models/scapula/"
    reference_for_output = "Statistics"
    reference_jcs_type = JointCoordinateSystem.SCS1
    plot_individual_scapulas = False
    plot_reference_scapula = False
    plot_all_scapulas = False
    plot_average_scapulas = True
    plot_histograms = True
    generate_latex = True
    angle_in_degrees = True
    scapulas_to_use = {
        "EOS": {
            "to_use": ["A", "P"],
            "A": {"folder": f"{base_folder}/Scapula-BD-EOS/asymptomatiques/"},
            "P": {"folder": f"{base_folder}/Scapula-BD-EOS/pathologiques/"},
            "shared_indices_with_reference": True,
            "reference": {
                "path": f"{base_folder}/Scapula-BD-EOS/asymptomatiques/PJ151-M001-scapula.ply",
                "use_precomputed_values": True,
            },
            "is_left": [
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
            ],
        },
        "Statistics": {
            "to_use": ["A", "P"],
            "A": {"folder": f"{base_folder}/Modele_stat/data/", "generate": 500},
            "P": {"folder": f"{base_folder}/Modele_stat/data/", "generate": 500},
            "shared_indices_with_reference": True,
            "reference": {
                "path": f"{base_folder}/Modele_stat/data/PJ116_scapula_A_avg.ply",
                "use_precomputed_values": True,
            },
        },
    }
    latex_save_folder = "latex/"
    #################

    # TODO Use pointing method if STL are used (not morphing the reference scapula)
    # TODO Save the values so they can be reused based on their respective file_path
    scapulas: dict[str, list[Scapula]] = {}
    reference_scapulas: dict[str, Scapula] = {}
    for scapula_type in scapulas_to_use.keys():
        if scapula_type in skip:
            continue

        scapulas[scapula_type] = []
        reference_scapulas[scapula_type] = get_reference_scapula(
            filepath=scapulas_to_use[scapula_type]["reference"]["path"],
            use_precomputed_values=scapulas_to_use[scapula_type]["reference"]["use_precomputed_values"],
            reference_jcs_type=reference_jcs_type,
        )
        if plot_reference_scapula:
            reference_scapulas[scapula_type].plot_geometry(show_now=True, show_glenoid=True, show_landmarks=True)

        for scapula_subtype in scapulas_to_use[scapula_type]["to_use"]:
            print(f"Processing {scapula_type} - {scapula_subtype}")
            if scapula_type == "EOS":
                for name in os.listdir(scapulas_to_use[scapula_type][scapula_subtype]["folder"]):
                    print(f"\t{name}...")
                    scapula = Scapula.from_reference_scapula(
                        geometry=os.path.join(scapulas_to_use[scapula_type][scapula_subtype]["folder"], name),
                        reference_scapula=reference_scapulas[scapula_type],
                        reference_jcs_type=reference_jcs_type,
                        shared_indices_with_reference=scapulas_to_use[scapula_type]["shared_indices_with_reference"],
                        is_left=name in scapulas_to_use[scapula_type]["is_left"],
                    )

                    scapulas[scapula_type].append(scapula)

            elif scapula_type == "Statistics":
                print(f"\tgenerating {scapulas_to_use[scapula_type][scapula_subtype]['generate']} scapulas...")
                scapula = Scapula.generator(
                    models_folder=scapulas_to_use[scapula_type][scapula_subtype]["folder"],
                    number_to_generate=scapulas_to_use[scapula_type][scapula_subtype]["generate"],
                    model=scapula_subtype,
                    reference_scapula=reference_scapulas[scapula_type],
                    reference_jcs_type=reference_jcs_type,
                )

                scapulas[scapula_type].extend(list(scapula))

            else:
                raise ValueError(f"Invalid scapula_folder type: {scapula_type}")

        if plot_individual_scapulas:
            for scapula in scapulas[scapula_type]:
                scapula.plot_geometry(
                    ax=reference_scapulas[scapula_type].plot_geometry(
                        show_now=False, marker="o", color="b", s=5, alpha=0.1, landmarks_color="b"
                    ),
                    data_type=ScapulaDataType.LOCAL,
                    show_jcs=[JointCoordinateSystem.SCS10],
                    show_now=True,
                    color="r",
                )

    # Compute the average reference system
    reference_scapula = reference_scapulas[reference_for_output]
    average_rts: dict[str, dict[JointCoordinateSystem, np.array]] = {}
    reference_rts: dict[str, dict[JointCoordinateSystem, np.array]] = {}
    average_rotation_errors: dict[str, dict[JointCoordinateSystem, np.array]] = {}
    average_translation_errors: dict[str, dict[JointCoordinateSystem, np.array]] = {}
    for key in scapulas.keys():
        average_rts[key] = {}
        average_rotation_errors[key] = {}
        average_translation_errors[key] = {}
        reference_rts[key] = {}
        for target in JointCoordinateSystem:
            all_rt = Scapula.get_frame_of_reference(scapulas[key], target, reference_system=reference_jcs_type)
            # Modify the translation so it is in distance (as opposed to normalized)
            for i, rt in enumerate(all_rt):
                rt[:3, 3] *= scapulas[key][i].scale_factor

            average_rts[key][target] = MatrixHelpers.average_matrices(all_rt)
            average_rotation_errors[key][target] = MatrixHelpers.angle_between_rotations(
                all_rt, average_rts[key][target]
            )
            average_translation_errors[key][target] = MatrixHelpers.distance_between_origins(
                all_rt, average_rts[key][target]
            )

            reference_rt = Scapula.get_frame_of_reference(
                [reference_scapula], target, reference_system=reference_jcs_type
            )
            reference_rt[0][:3, 3] *= reference_scapula.scale_factor
            reference_rts[key][target] = MatrixHelpers.average_matrices(reference_rt)

    # Export to LaTeX
    if generate_latex:
        for key in average_rts.keys():
            PlotHelpers.export_average_matrix_to_latex(
                f"{latex_save_folder}/average_transformations_{key}.tex",
                average_rts[key],
                reference_rts[key],
                angle_name=key,
                reference_system=reference_jcs_type,
            )

            PlotHelpers.export_errors_to_latex(
                f"{latex_save_folder}/errors_{key}.tex",
                angle_name=key,
                average_angles=average_rotation_errors[key],
                average_translations=average_translation_errors[key],
                reference_system=reference_jcs_type,
                angle_in_degrees=angle_in_degrees,
            )

    if plot_histograms:
        Scapula.plot_error_histogram(
            average_rts=average_rts, average_errors=average_rotation_errors, angle_in_degrees=angle_in_degrees
        )

    # Plot all the scapula rt
    if plot_all_scapulas:
        for key in scapulas.keys():
            for gcs_to_show in JointCoordinateSystem:
                Scapula.plot_systems_in_reference_scapula(reference_scapula, scapulas[key], gcs_to_show)

    # Plot the scapulas in the reference scapula
    if plot_average_scapulas:
        for gcs_to_show in JointCoordinateSystem:
            title = f"Scapula averages of {gcs_to_show.name}"
            ax = reference_scapula.plot_geometry(
                show_now=False,
                marker="o",
                color="b",
                s=5,
                alpha=0.1,
                figure_title=title,
                show_landmarks=True,
            )
            ax.set_title(title)
            origin = reference_scapula.get_joint_coordinates_system(gcs_to_show)[:3, 3]

            linestyles = {"Statistics": "--", "EOS": "-"}
            for key in average_rts.keys():
                PlotHelpers.show_axes(
                    reference_scapulas[key].get_joint_coordinates_system(gcs_to_show),
                    ax=ax,
                    translate_to=origin,
                    linestyle=linestyles[key],
                )
                PlotHelpers.show_axes(
                    average_rts[key][gcs_to_show], ax=ax, linewidth=5, translate_to=origin, linestyle=linestyles[key]
                )

            ax.set_xlim(-0.8, 1.1)
            ax.set_ylim(-0.8, 1.1)
            ax.set_zlim(-0.8, 1.1)
        PlotHelpers.show()


if __name__ == "__main__":
    main()
