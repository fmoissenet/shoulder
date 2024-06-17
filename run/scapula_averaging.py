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
            landmarks["AA"] = np.array([-78.4296, -77.6471, 1461.666])
            landmarks["AC"] = np.array([-42.9741, -75.9749, 1478.648])
            landmarks["AI"] = np.array([-101.0337, 30.4742, 1328.427])
            landmarks["GC_CONTOURS"] = [
                # np.array([-44.5196, -58.2721, 1434.744]),
                np.array([-55.5574, -61.6306, 1438.7978]),
                np.array([-56.1480, -62.1223, 1437.2054]),
                np.array([-56.5098, -62.3411, 1435.3087]),
                np.array([-56.4796, -62.8190, 1433.3354]),
                np.array([-56.4067, -62.9497, 1431.4057]),
                np.array([-56.3423, -63.2794, 1429.9996]),
                np.array([-55.8402, -63.5005, 1428.4134]),
                np.array([-55.1354, -63.5717, 1426.8645]),
                np.array([-54.4079, -63.5274, 1425.2611]),
                np.array([-53.0645, -63.4687, 1423.6015]),
                np.array([-51.4247, -63.5307, 1422.3685]),
                np.array([-49.5969, -63.3878, 1421.1534]),
                np.array([-47.7149, -63.1350, 1420.5169]),
                np.array([-45.7338, -62.7885, 1419.9713]),
                np.array([-43.6296, -62.3171, 1419.5781]),
                np.array([-42.1541, -61.9004, 1419.7362]),
                np.array([-40.4246, -61.6258, 1420.1055]),
                np.array([-38.5735, -60.9826, 1420.7021]),
                np.array([-36.9161, -60.4591, 1421.5690]),
                np.array([-35.5175, -59.8560, 1422.5805]),
                np.array([-34.1758, -59.3923, 1424.0521]),
                np.array([-33.2812, -58.6889, 1425.3764]),
                np.array([-32.4931, -58.3901, 1426.9992]),
                np.array([-31.9843, -58.1107, 1428.4809]),
                np.array([-31.6268, -57.9430, 1429.9537]),
                np.array([-31.5374, -57.5556, 1431.3334]),
                np.array([-31.3656, -57.5842, 1432.9304]),
                np.array([-31.4772, -57.5244, 1434.6384]),
                np.array([-31.4743, -57.5618, 1436.3835]),
                np.array([-31.3694, -57.5914, 1437.9992]),
                np.array([-31.1427, -58.0074, 1439.8841]),
                np.array([-31.0159, -58.5420, 1441.4790]),
                np.array([-30.7118, -58.8409, 1443.1997]),
                np.array([-30.3949, -59.2316, 1444.8914]),
                np.array([-30.4515, -59.8141, 1446.3438]),
                np.array([-30.5311, -60.2860, 1447.9426]),
                np.array([-30.8790, -60.6002, 1449.3629]),
                np.array([-31.8224, -60.8694, 1450.6899]),
                np.array([-33.3906, -61.0832, 1451.7214]),
                np.array([-35.2567, -60.9241, 1452.4091]),
                np.array([-36.9942, -60.9872, 1452.2508]),
                np.array([-38.8440, -60.9385, 1452.1469]),
                np.array([-40.5193, -60.8740, 1451.9503]),
                np.array([-42.2359, -60.6887, 1451.3040]),
                np.array([-43.5085, -60.8957, 1450.7296]),
                np.array([-45.3559, -60.5407, 1450.0366]),
                np.array([-46.8346, -60.7522, 1449.0709]),
                np.array([-48.4158, -60.8076, 1448.0628]),
                np.array([-49.9428, -60.4611, 1447.0750]),
                np.array([-50.8484, -60.8381, 1445.9136]),
                np.array([-52.0131, -60.5703, 1444.7331]),
                np.array([-52.8918, -60.9948, 1443.4036]),
                np.array([-54.0063, -61.4301, 1442.1071]),
                np.array([-55.0000, -61.1420, 1440.4970]),
            ]
            landmarks["IE"] = np.array([-50.103, -63.4075, 1421.618])
            landmarks["SE"] = np.array([-33.5696, -61.1017, 1451.804])
            landmarks["TS"] = np.array([-64.0712, 52.9972, 1447.156])
        elif filename == "PJ116_scapula_A_avg.ply":
            landmarks["AA"] = np.array([-76.36315, -76.65424, 1465.118])
            landmarks["AC"] = np.array([-42.40293, -76.51300, 1481.305])
            landmarks["AI"] = np.array([-100.1907, 34.60924, 1329.708])
            landmarks["GC_CONTOURS"] = [
                np.array([-60.19635, -61.33104, 1438.422]),
                np.array([-60.39573, -61.95986, 1435.552]),
                np.array([-61.25107, -62.02857, 1433.651]),
                np.array([-61.5977, -62.13195, 1431.84]),
                np.array([-60.44757, -62.71045, 1429.9]),
                np.array([-59.2497, -63.01856, 1427.533]),
                np.array([-59.2497, -63.01856, 1427.533]),
                np.array([-58.00447, -63.21724, 1424.741]),
                np.array([-57.30323, -63.24438, 1422.351]),
                np.array([-57.30323, -63.24438, 1422.351]),
                np.array([-54.54115, -62.95175, 1422.478]),
                np.array([-53.05796, -63.15958, 1420.221]),
                np.array([-49.71919, -62.85508, 1418.852]),
                np.array([-49.71919, -62.85508, 1418.852]),
                np.array([-47.28054, -62.39348, 1419.175]),
                np.array([-44.55117, -61.76605, 1419.421]),
                np.array([-43.57395, -61.41884, 1418.432]),
                np.array([-41.83871, -61.17418, 1419.535]),
                np.array([-39.90199, -60.51387, 1420.847]),
                np.array([-37.95419, -59.81199, 1422.035]),
                np.array([-37.28038, -59.20134, 1424.714]),
                np.array([-35.33654, -58.59534, 1425.788]),
                np.array([-35.9418, -58.51763, 1427.404]),
                np.array([-35.9418, -58.51763, 1427.404]),
                np.array([-34.4885, -57.99295, 1429.481]),
                np.array([-35.5196, -57.76676, 1431.87]),
                np.array([-35.5196, -57.76676, 1431.87]),
                np.array([-36.34355, -57.50804, 1434.157]),
                np.array([-35.74775, -57.71066, 1436.666]),
                np.array([-35.5744, -57.97537, 1438.643]),
                np.array([-34.62261, -58.24261, 1440.375]),
                np.array([-34.62261, -58.24261, 1440.375]),
                np.array([-34.3879, -58.94827, 1443.051]),
                np.array([-35.59828, -59.31228, 1444.296]),
                np.array([-34.06554, -59.84186, 1446.433]),
                np.array([-34.40521, -60.7108, 1449.246]),
                np.array([-34.40521, -60.7108, 1449.246]),
                np.array([-35.35465, -61.09417, 1451.724]),
                np.array([-37.18041, -61.22071, 1450.973]),
                np.array([-37.72746, -61.32121, 1452.458]),
                np.array([-40.99765, -61.14626, 1452.715]),
                np.array([-40.99765, -61.14626, 1452.715]),
                np.array([-43.73926, -61.01735, 1452.293]),
                np.array([-45.51012, -60.61703, 1452.793]),
                np.array([-46.73182, -60.90506, 1450.342]),
                np.array([-48.83723, -60.19616, 1451.696]),
                np.array([-49.93012, -60.63281, 1449.283]),
                np.array([-51.94585, -60.24407, 1449.275]),
                np.array([-52.4436, -60.65552, 1447.118]),
                np.array([-54.73639, -60.66155, 1445.768]),
                np.array([-56.64297, -60.16588, 1445.096]),
                np.array([-56.58953, -60.91004, 1442.557]),
                np.array([-58.68475, -60.73643, 1441.695]),
                np.array([-57.84651, -61.10421, 1440.172]),
            ]
            landmarks["IE"] = np.array([-53.05796, -63.15958, 1420.221])
            landmarks["SE"] = np.array([-37.72746, -61.32121, 1452.458])
            landmarks["TS"] = np.array([-64.70759, 51.27537, 1445.679])
        else:
            raise ValueError(f"Precomputed values for {filename} are not available.")
    else:
        landmarks = None

    # Load the geometry data
    return Scapula.from_landmarks(geometry=filepath, predefined_landmarks=landmarks)


def main():
    #### OPTIONS ####
    skip = []  # ["Statistics"]  # ["EOS"]
    base_folder = "models/scapula/"
    reference_for_output = "Statistics"
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
            all_rt = Scapula.get_frame_of_reference(scapulas[key], target, reference_system=JointCoordinateSystem.ISB)
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
                [reference_scapula], target, reference_system=JointCoordinateSystem.ISB
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
                reference_system=JointCoordinateSystem.ISB,
            )

            PlotHelpers.export_errors_to_latex(
                f"{latex_save_folder}/errors_{key}.tex",
                angle_name=key,
                average_angles=average_rotation_errors[key],
                average_translations=average_translation_errors[key],
                reference_system=JointCoordinateSystem.ISB,
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
