import os

import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData
from matplotlib.widgets import Button


def load_scapula_geometry(file_path: str) -> np.ndarray:
    extension = file_path.split(".")[-1]
    if extension == "ply":
        data = None
        with open(file_path, "rb") as f:
            plydata = PlyData.read(f)
            tp = np.asarray(plydata["vertex"])
            data = np.array((tp["x"], tp["y"], tp["z"]))
            data = np.concatenate((data, np.ones((1, data.shape[1]))))
        return data

    else:
        raise NotImplementedError(f"The file extension {extension} is not supported yet.")


def plot_scapula_geometry(data: np.ndarray) -> None:
    fig = plt.figure(f"Scapula")
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(data[0, :], data[1, :], data[2, :], ".", s=1, picker=5, alpha=0.3)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])
    plt.show()


def pickable_scapula_geometry_plot(points_name: list[str], data: np.ndarray) -> dict[str, np.array]:
    # Prepare the figure
    fig = plt.figure(f"Pick the points")
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(data[0, :], data[1, :], data[2, :], ".", s=1, picker=5, alpha=0.3)
    scatter = ax.scatter(np.nan, np.nan, np.nan, ".", c="r", s=50)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])

    # Add the next button
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, "Next")

    def on_pick_point(event):
        picked_index = int(event.ind[0])

        picked_point[0] = data[:, picked_index]
        scatter._offsets3d = (
            picked_point[0][np.newaxis, 0],
            picked_point[0][np.newaxis, 1],
            picked_point[0][np.newaxis, 2],
        )
        scatter.set_sizes([50])
        fig.canvas.draw_idle()

    def on_confirmed_point(event):
        # Save the previous point
        if event is not None:
            picked_points[points_name[current_point[0]]] = picked_point[0]

        current_point[0] += 1

        if current_point[0] == len(points_name):
            plt.close(fig)
            return

        picked_point[0] = np.array([np.nan, np.nan, np.nan])
        scatter._offsets3d = (
            picked_point[0][np.newaxis, 0],
            picked_point[0][np.newaxis, 1],
            picked_point[0][np.newaxis, 2],
        )

        point_name = points_name[current_point[0]]
        ax.title.set_text(f"Pick the {point_name} then close the window")
        fig.canvas.draw_idle()

    # Setup the connection
    picked_points = {}
    current_point = [-1]
    picked_point = [np.array([np.nan, np.nan, np.nan])]
    on_confirmed_point(None)
    fig.canvas.mpl_connect("pick_event", on_pick_point)
    bnext.on_clicked(on_confirmed_point)
    plt.show()

    return picked_points


def transpose_homogenous_matrix(homogenous_matrix: np.ndarray) -> np.ndarray:
    out = np.zeros((4, 4))
    out[:3, :3] = homogenous_matrix[:3, :3].transpose()
    out[:3, 3] = -out[:3, :3] @ homogenous_matrix[:3, 3]
    return out


def compute_scapula_isb_coordinate_system(points: dict[str, np.array]) -> np.array:
    origin = points["AC"][:3]
    x = points["AA"][:3] - points["TS"][:3]
    z = points["TS"][:3] - points["IA"][:3]
    y = np.cross(z, x)
    z = np.cross(x, y)
    x /= np.linalg.norm(x)
    y /= np.linalg.norm(y)
    z /= np.linalg.norm(z)
    lcs = np.eye(4)
    lcs[:3, 0] = x
    lcs[:3, 1] = y
    lcs[:3, 2] = z
    lcs[:3, 3] = origin
    return lcs


def get_reference_scapula_data(filepath: str, use_precomputed_values: bool):
    # Load the geometry data
    reference_scapula = load_scapula_geometry(filepath)

    # Get some remarkable points
    if use_precomputed_values:
        points = {}
        points["IA"] = np.array([-99.4965, 29.879, 1328.5854, 1.0])
        points["TS"] = np.array([-63.5616, 53.0593, 1444.6829, 1.0])
        points["AA"] = np.array([-80.0813, -71.6938, 1461.5833, 1.0])
        points["AC"] = np.array([-46.3822, -69.0688, 1471.2103, 1.0])
    else:
        reference_frame_marker_names = ["IA", "TS", "AA", "AC"]
        points = pickable_scapula_geometry_plot(points_name=reference_frame_marker_names, data=reference_scapula)
        if False in [name in points.keys() for name in reference_frame_marker_names]:
            raise RuntimeError("Not all required points were selected")

    # Project the scapula in its local reference frame based on ISB
    reference_isb_gcs = compute_scapula_isb_coordinate_system(points)
    reference_scapula = transpose_homogenous_matrix(reference_isb_gcs) @ reference_scapula

    return reference_scapula, reference_isb_gcs


def main():
    # Load the reference scapula
    reference_scapula, reference_isb_gcs = get_reference_scapula_data(
        filepath="models/scapula/reference/PJ151-M001-scapula.ply", use_precomputed_values=True
    )

    # Plot for debug
    plot_scapula_geometry(reference_scapula)

    # Sequentially analyse all the scapulas
    scapula_folder = "models/scapula/Scapula-BD-EOS/asymptomatiques/"
    scapula_files = os.listdir(scapula_folder)

    for file in scapula_files:
        # TODO load the scapula data
        # TODO Find the optimal transformation to get to the reference scapula
        # TODO Automatically find the distance between corresponding indices to see if they match
        # TODO or automatically label all the scapula bony landmarks based on their proximity with the reference
        # TODO Get all the reference frames
        # TODO Project the scapula in the local reference frame
        # TODO Compute all the difference reference frames
        ...

    # TODO Compute the average matrices from ISB reference frame to other local coordinate systems
    # TODO Compute the "standard deviation" to the average matrices (variability)
    # TODO Show the results in a table


if __name__ == "__main__":
    main()
