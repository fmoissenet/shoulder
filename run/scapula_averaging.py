import os

import numpy as np
from scapula import Scapula, ScapulaDataType, PlotHelpers


def get_reference_scapula(filepath: str, use_precomputed_values: bool):
    # Get some remarkable points
    if use_precomputed_values:
        landmarks = {}
        landmarks["IA"] = np.array([-0.42450786, 0.12748057, 5.66849068, 1.0])
        landmarks["TS"] = np.array([-0.27999221, 0.22328151, 6.13702906, 1.0])
        landmarks["AA"] = np.array([-0.34284121, -0.29284564, 6.23839738, 1.0])
        landmarks["AC"] = np.array([-0.19040381, -0.29713313, 6.27516834, 1.0])
    else:
        landmarks = None

    # Load the geometry data
    return Scapula.from_landmarks(filepath=filepath, predefined_landmarks=landmarks)


def main():
    # Load the reference scapula
    reference_scapula = get_reference_scapula(
        filepath="models/scapula/reference/PJ151-M001-scapula.ply", use_precomputed_values=True
    )

    # Plot for debug
    # from matplotlib import pyplot as plt

    # fig = plt.figure(f"Scapula")
    # ax_with_ref = fig.add_subplot(121, projection="3d")
    ax = reference_scapula.plot_geometry(show_now=False, marker="o", color="b", s=10)

    # Sequentially analyse all the scapulas
    scapula_folder = "models/scapula/Scapula-BD-EOS/asymptomatiques/"
    scapula_files = os.listdir(scapula_folder)

    left_scapula_files = ["PJ151-M021-scapula.ply", "PJ151-M015-scapula.ply"]

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

        ax = reference_scapula.plot_geometry(show_now=False, marker="o", color="b", s=10)
        scapula.plot_geometry(
            ax=ax,
            data_type=ScapulaDataType.LOCAL,
            show_now=True,
            color="r",
        )

        # TODO Automatically find the distance between corresponding indices to see if they match
        # TODO Compute all the different reference frames
    PlotHelpers.show()

    # TODO Compute the average matrices from ISB reference frame to other local coordinate systems
    # TODO Compute the "standard deviation" to the average matrices (variability)
    # TODO Show the results in a table


if __name__ == "__main__":
    main()
