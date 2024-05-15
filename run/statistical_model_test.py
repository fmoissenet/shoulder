import copy
import math

import numpy as np
from plyfile import PlyData
import os

dir_models = "./models/scapula/Modele_stat/data/"  # Repertoire ou sont situées les données
dir_new_models = "save/"  # Repertoire ou sont sauvées les modeles créés

model_stat_type = "A"  # 'A', 'P' ou 'H'
mode = [0, 3, 5]  # Liste des valeurs et vecteurs propres qui participent a la contruction du modele
number_of_models = 1  # Indiquer le nombre de modele a produire
is_display = True  # True : affichage des modeles produits si False non
debug = False

model_stat_A = {
    "mean_model_path": os.path.join(dir_models, "PJ116_scapula_A_avg.ply"),
    "pca_eigen_vectors": os.path.join(dir_models, "PJ116_eigen_vectors_scapula_A.csv"),
    "pca_eigen_values": os.path.join(dir_models, "PJ116_eigen_values_scapula_A.csv"),
}

model_stat_P = {
    "mean_model_path": os.path.join(dir_models, "PJ116_scapula_P_avg.ply"),
    "pca_eigen_vectors": os.path.join(dir_models, "PJ116_eigen_vectors_scapula_P.csv"),
    "pca_eigen_values": os.path.join(dir_models, "PJ116_eigen_values_scapula_P.csv"),
}

model_stat_FHOrtho = {
    "mean_model_path": os.path.join(dir_models, "FHOrtho_scapula_avg.ply"),
    "pca_eigen_vectors": os.path.join(dir_models, "PJ116_eigen_vectors_scapula_FHOrtho.csv"),
    "pca_eigen_values": os.path.join(dir_models, "PJ116_eigen_values_scapula_FHOrtho.csv"),
}


class Open3dMock:
    def __init__(self, vertices: np.ndarray = None, triangles: np.ndarray = None):
        self.vertices = vertices
        self.triangles = triangles

    @classmethod
    def from_ply(cls, path):
        ply_data = PlyData.read(path)

        vertices_tp = np.asarray(ply_data["vertex"])
        vertices = np.array((vertices_tp["x"], vertices_tp["y"], vertices_tp["z"]))

        triangles = np.asarray(ply_data["face"])["vertex_indices"]

        return cls(vertices, triangles)

    def compute_vertex_normals(self):
        pass

    def paint_uniform_color(self, color):
        pass


class o3d:
    class io:
        @staticmethod
        def read_triangle_mesh(path):
            return Open3dMock.from_ply(path)

        @staticmethod
        def write_triangle_mesh(path, mesh):
            pass

    class utility:
        @staticmethod
        def Vector3dVector(vertices):
            return vertices

    class visualization:
        @staticmethod
        def draw_geometries(geometries):
            from matplotlib import pyplot as plt

            fig = plt.figure(f"Scapula")
            ax = fig.add_subplot(111, projection="3d")

            colors = ["b", "r", "g", "y", "c", "m"]
            for i, geometry in enumerate(geometries):
                data = geometry.vertices
                kwargs = {}
                kwargs["s"] = 1
                # kwargs["alpha"] = 0.3
                kwargs["color"] = colors[i]
                kwargs["marker"] = "."
                ax.scatter(data[0, :], data[1, :], data[2, :], picker=5, **kwargs)

                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_zlabel("Z")

                ax.set_box_aspect([1, 1, 1])

            plt.show()


def load_data():
    mean_model = None
    pca_eigen_vectors = None
    pca_eigen_values = None

    # Lecture de modele moyen et des parametres PCA (vecteurs + valeurs) selon le modele statistique choisi
    if model_stat_type == "A":
        # Lecture du fichier ply
        mean_model = o3d.io.read_triangle_mesh(model_stat_A["mean_model_path"])

        # Lecture du fichier csv contenant les vecteurs propres estimes par PCA
        pca_eigen_vectors = np.loadtxt(model_stat_A["pca_eigen_vectors"], delimiter=";", dtype=float)

        # Lecture du fichier csv contenant les valeurs propres estimees par PCA
        pca_eigen_values = np.loadtxt(model_stat_A["pca_eigen_values"], delimiter=";", dtype=float)
    elif model_stat_type == "P":
        # Lecture du fichier ply
        mean_model = o3d.io.read_triangle_mesh(model_stat_P["mean_model_path"])

        # Lecture du fichier csv contenant les vecteurs propres estimes par PCA
        pca_eigen_vectors = np.loadtxt(model_stat_P["pca_eigen_vectors"], delimiter=";", dtype=float)

        # Lecture du fichier csv contenant les valeurs propres estimees par PCA
        pca_eigen_values = np.loadtxt(model_stat_P["pca_eigen_values"], delimiter=";", dtype=float)
    else:
        # Lecture du fichier ply
        mean_model = o3d.io.read_triangle_mesh(model_stat_FHOrtho["mean_model_path"])

        # Lecture du fichier csv contenant les vecteurs propres estimes par PCA
        pca_eigen_vectors = np.loadtxt(model_stat_FHOrtho["pca_eigen_vectors"], delimiter=";", dtype=float)

        # Lecture du fichier csv contenant les valeurs propres estimees par PCA
        pca_eigen_values = np.loadtxt(model_stat_FHOrtho["pca_eigen_values"], delimiter=";", dtype=float)

    return mean_model, pca_eigen_vectors, pca_eigen_values


def build_model(mean_model, mean_model_vertices, pca_eigen_vectors, pca_eigen_values):
    """
    Fonction pour la creation d'un modele surfacique a partir d'un modele statistique PCA
    :param mean_model: modele moyen
    :param mean_model_vertices: sommets du modele moyen
    :param pca_eigen_vectors: vecteurs propres estimes par PCA
    :param pca_eigen_values: valeurs propres estimees par PCA
    :return: le modele surfacique produit
    """
    # b is a vector of floats to apply to each mode of variation
    full_b = np.zeros(pca_eigen_values.shape)
    for i in mode:
        # Une valeur randomisee uniforme entre les bornes statistiquement acceptable +/-3 * sqrt(eigen_value)
        new_value = np.random.uniform(-3 * math.sqrt(pca_eigen_values[i]), 3 * math.sqrt(pca_eigen_values[i]))
        full_b[i] = new_value
    # Calcul de P * b
    pca_applied = pca_eigen_vectors.dot(full_b)

    # Creation d'un nouveau modele : mean_model + pca_applied
    example = mean_model_vertices.flatten().T + pca_applied
    new_model = copy.deepcopy(mean_model)
    new_model_vertices = np.reshape(example, (mean_model_vertices.shape[0], mean_model_vertices.shape[1]))
    new_model.vertices = o3d.utility.Vector3dVector(new_model_vertices)

    return new_model


def build_models():
    """
    Fonction pour la production de modeles surfaciques a l'aide du modele statistique.
         modele moyen + les parametres de la PCA :
         new_model = mean_model + P * b, P : matrice contenant les vecteurs propres, b parametres

         Les nouveaux modeles sont sauvés dans des fichiers stl
    """
    # Lecture de modele moyen et des parametres PCA (vecteurs + valeurs) selon le modele statistique choisi
    mean_model, pca_eigen_vectors, pca_eigen_values = load_data()

    # Recuperation des sommets et des faces du modele 3D moyen
    mean_model_vertices = np.asarray(mean_model.vertices)
    mean_model_faces = np.asarray(mean_model.triangles)

    # Debug pour les donnees chargees
    if debug:
        print("Vertices - shape:")
        print(mean_model_vertices.shape)
        print("Triangles - shape:")
        print(mean_model_faces.shape)

        print("Eigen vectors - shape:")
        print(pca_eigen_vectors.shape)

        # Verification entre le nombre de sommets du modele moyen et le nombre de composantes qui forment un vecteur propre
        if mean_model_vertices.shape[0] * 3 != pca_eigen_vectors.shape[0]:
            print("Attention, erreur entre le model moyen et les vecteurs propres")

        print("Eigen values - shape:")
        print(pca_eigen_values.shape)

    # Generation de nouveaux modeles : new_model = mean_model + P * b
    new_models = []
    for i in range(number_of_models):
        # Creation d'un nouveau modele
        new_model = build_model(mean_model, mean_model_vertices, pca_eigen_vectors, pca_eigen_values)

        # Sauvegarde du modele dans un fichier .ply
        name_model = "model_" + str(i) + ".ply"
        path_save_model = os.path.join(dir_new_models, name_model)
        o3d.io.write_triangle_mesh(path_save_model, new_model)

        new_models.append(new_model)

    # Affichage du modele moyen et du nouveau modele
    if is_display:
        all_models = [mean_model]
        all_models.extend(new_models)
        o3d.visualization.draw_geometries(all_models)


if __name__ == "__main__":
    build_models()
