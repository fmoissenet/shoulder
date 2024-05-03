from functools import partial
import os

import matplotlib.pyplot as plt
import numpy as np
from plyfile import PlyData


def load_scapula(file_path: str) -> np.ndarray:
    extension = file_path.split(".")[-1]
    if extension == "ply":
        data = None
        with open(file_path, "rb") as f:
            plydata = PlyData.read(f)
            tp = np.asarray(plydata["vertex"])
            data = np.array((tp["x"], tp["y"], tp["z"]))
        return data

    else:
        raise NotImplementedError(f"The file extension {extension} is not supported yet.")


def on_pick(fig, ax, scatter, data, event):
    ind = int(event.ind[0])

    pt = data[:, ind]
    scatter._offsets3d = pt[np.newaxis, 0], pt[np.newaxis, 1], pt[np.newaxis, 2]
    fig.canvas.draw_idle()
    print(f"Point picked: {pt[0]:.3f}, {pt[1]:.3f}, {pt[2]:.3f}")


def main():
    reference_scapula = load_scapula(f"models/scapula/Scapula-BD-EOS/asymptomatiques/PJ151-M001-scapula.ply")

    scapula_folder = "models/scapula/Scapula-BD-EOS/asymptomatiques/"
    scapula_files = os.listdir(scapula_folder)

    # TEMPORARY
    scapula_files = scapula_files[:1]

    scapulas = []
    for file in scapula_files:
        scapulas.append(load_scapula(f"{scapula_folder}/{file}"))

    # Show the scapulas
    fig = plt.figure("")
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(reference_scapula[0, :], reference_scapula[1, :], reference_scapula[2, :], ".", s=1, picker=5, alpha=0.3)
    scatter = ax.scatter(-60, 0, 1325, ".", c="r", s=10)
    # for scapula in scapulas:
    #   ax.scatter(scapula[0, :], scapula[1, :], scapula[2, :], ".", s=1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])
    fig.canvas.mpl_connect("pick_event", partial(on_pick, fig, ax, scatter, reference_scapula))
    plt.show()


if __name__ == "__main__":
    main()
