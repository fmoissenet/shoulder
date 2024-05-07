import os

import numpy as np
from scapula import Scapula, PlotHelpers


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
    return Scapula(filepath=filepath, predefined_landmarks=landmarks)


def main():
    # Load the reference scapula
    reference_scapula = get_reference_scapula(
        filepath="models/scapula/reference/PJ151-M001-scapula.ply", use_precomputed_values=False
    )

    # Plot for debug
    ax = reference_scapula.plot_geometry(show_now=False, marker="o", color="b", s=10)

    # Sequentially analyse all the scapulas
    scapula_folder = "models/scapula/Scapula-BD-EOS/asymptomatiques/"
    scapula_files = os.listdir(scapula_folder)

    for file in scapula_files:
        # Load the scapula data
        filepath = os.path.join(scapula_folder, file)
        scapula = Scapula(filepath=filepath, reference_scapula=reference_scapula)
        scapula.plot_geometry(ax=ax, show_now=False, color="r")
        # TODO Find the optimal transformation to get to the reference scapula
        # TODO Automatically find the distance between corresponding indices to see if they match
        # TODO or automatically label all the scapula bony landmarks based on their proximity with the reference
        # TODO Get all the reference frames
        # TODO Project the scapula in the local reference frame
        # TODO Compute all the difference reference frames

    PlotHelpers.show()

    # TODO Compute the average matrices from ISB reference frame to other local coordinate systems
    # TODO Compute the "standard deviation" to the average matrices (variability)
    # TODO Show the results in a table


if __name__ == "__main__":
    main()
