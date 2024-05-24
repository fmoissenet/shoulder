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
            landmarks["SA"] = np.array([-0.1021418924890583, 0.10170073318175565, 6.2920059467986755, 1.0])
            landmarks["AA"] = np.array([-0.3332845410297929, -0.3215975587141159, 6.231563695676402, 1.0])
            landmarks["AC"] = np.array([-0.18210408, -0.30156264, 6.26945651, 1.0])
            landmarks["AT"] = np.array([-0.1266968011420527, -0.3688454755778767, 6.265601393258909, 1.0])
            landmarks["CP"] = np.array([-0.017074350751475963, -0.2542251571836168, 6.177195252266086, 1.0])
            landmarks["GC"] = np.array([-0.18614066393693082, -0.24647324980998891, 6.13033391435741, 1.0])
            landmarks["IE"] = np.array([-0.24505717, -0.25463561, 6.0552915, 1.0])
            landmarks["SE"] = np.array([-0.12792344, -0.25661786, 6.19245894, 1.0])
        elif filename == "PJ116_scapula_A_avg.ply":
            landmarks["IA"] = np.array([-0.43178419, 0.14275525, 5.68034491, 1.0])
            landmarks["TS"] = np.array([-0.30306839, 0.18366926, 6.15101467, 1.0])
            landmarks["SA"] = np.array([-0.13111922, 0.09036657, 6.28140869, 1.0])
            landmarks["AA"] = np.array([-0.33216135, -0.30433252, 6.25798885, 1.0])
            landmarks["AC"] = np.array([-0.17491792, -0.30715463, 6.28388734, 1.0])
            landmarks["AT"] = np.array([-0.13147937, -0.37142136, 6.28457016, 1.0])
            landmarks["CP"] = np.array([-0.02247475, -0.28800937, 6.14408965, 1.0])
            landmarks["GC"] = np.array([-0.20194872, -0.24786309, 6.10602707, 1.0])
            landmarks["IE"] = np.array([-0.26225041, -0.26192896, 6.07919785, 1.0])
            landmarks["SE"] = np.array([-0.14490825, -0.25775035, 6.19629289, 1.0])
        else:
            raise ValueError(f"Precomputed values for {filename} are not available.")
    else:
        landmarks = None

    # Load the geometry data
    return Scapula.from_landmarks(geometry=filepath, predefined_landmarks=landmarks)


def main():
    #### OPTIONS ####
    skip = ["EOS"]
    base_folder = "models/scapula/"
    reference_for_output = "Statistics"
    plot_individual_scapulas = True
    plot_all_scapulas = False
    plot_average_scapulas = True
    generate_latex = True
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
            "A": {"folder": f"{base_folder}/Modele_stat/data/", "generate": 10},
            "P": {"folder": f"{base_folder}/Modele_stat/data/", "generate": 10},
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
    reference_scapulas = {}
    for scapula_type in scapulas_to_use.keys():
        if scapula_type in skip:
            continue

        scapulas[scapula_type] = []
        reference_scapulas[scapula_type] = get_reference_scapula(
            filepath=scapulas_to_use[scapula_type]["reference"]["path"],
            use_precomputed_values=scapulas_to_use[scapula_type]["reference"]["use_precomputed_values"],
        )

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
                        show_now=False, marker="o", color="b", s=5, alpha=0.1
                    ),
                    data_type=ScapulaDataType.LOCAL,
                    show_jcs=[JointCoordinateSystem.ISB],
                    show_now=True,
                    color="r",
                )

    # Compute the average reference system
    reference_scapula = reference_scapulas[reference_for_output]
    average_rts = {}
    reference_rts = {}
    average_errors = {}
    for key in scapulas.keys():
        average_rts[key] = {}
        average_errors[key] = {}
        reference_rts[key] = {}
        for type in JointCoordinateSystem:
            all_rt = Scapula.change_frame_of_reference(scapulas[key], type, reference_system=JointCoordinateSystem.ISB)
            average_rts[key][type] = MatrixHelpers.average_matrices(all_rt)
            average_errors[key][type] = MatrixHelpers.angle_between_rotations(all_rt, average_rts[key][type])

            reference_rt = Scapula.change_frame_of_reference(
                [reference_scapula], type, reference_system=JointCoordinateSystem.ISB
            )
            reference_rts[key][type] = MatrixHelpers.average_matrices(reference_rt)

    # Export to LaTeX
    if generate_latex:
        for key in average_rts.keys():
            PlotHelpers.export_average_matrix_to_latex(
                f"{latex_save_folder}/average_transformations_{key}.tex",
                average_rts[key],
                average_errors[key],
                angle_name=key,
                reference_system=JointCoordinateSystem.ISB,
            )
            PlotHelpers.export_average_matrix_to_latex(
                f"{latex_save_folder}/reference_transformations_{key}.tex",
                reference_rts[key],
                None,
                angle_name=key,
                reference_system=JointCoordinateSystem.ISB,
            )

    from matplotlib import pyplot as plt

    x_lim = -np.inf
    for key in average_rts.keys():
        for type in JointCoordinateSystem:
            x_lim = max(x_lim, max(average_errors[key][type] * 180 / np.pi))
    x_lim = int(x_lim) + 1

    for key in average_rts.keys():
        n_graphs = len(JointCoordinateSystem)
        _, axs = plt.subplots(n_graphs, 1, tight_layout=True, num=f"Error distribution for {key}")
        for i in range(n_graphs):
            type = list(JointCoordinateSystem)[i]
            axs[i].set_title(f"Error distribution for {key} - {type.name}")
            axs[i].hist(average_errors[key][type] * 180 / np.pi, bins=range(x_lim))
            axs[i].set_xlim(0, x_lim)
        axs[n_graphs - 1].set_xlabel(f"Error (°)")
        plt.show()

    # Plot all the scapula rt
    if plot_all_scapulas:
        for key in scapulas.keys():
            for gcs_to_show in JointCoordinateSystem:
                Scapula.plot_systems_in_reference_scapula(reference_scapula, scapulas[key], gcs_to_show)

    # Plot the scapulas in the reference scapula
    if plot_average_scapulas:
        for gcs_to_show in JointCoordinateSystem:
            ax = reference_scapula.plot_geometry(show_now=False, marker="o", color="b", s=5, alpha=0.1)
            ax.set_title(f"Scapula averages of {gcs_to_show.name}")
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
