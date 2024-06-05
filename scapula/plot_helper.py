import os

import numpy as np
import matplotlib.pyplot as plt

from .enums import JointCoordinateSystem


class PlotHelpers:
    @staticmethod
    def show_axes(axes: np.ndarray, ax: plt.Axes = None, translate_to: np.ndarray = None, **kwargs):
        """
        Show the axes in the plot.

        Args:
        ax: matplotlib axis
        axes: 4x4 matrix representing the axes
        translate_to: numpy array of shape (3,) representing the translation to apply to the axes
        kwargs: additional arguments to pass to the quiver function
        """

        origin = axes[:3, 3] if translate_to is None else translate_to
        x = axes[:3, 0]
        y = axes[:3, 1]
        z = axes[:3, 2]
        ax.quiver(*origin, *x, color="r", **kwargs)
        ax.quiver(*origin, *y, color="g", **kwargs)
        ax.quiver(*origin, *z, color="b", **kwargs)

    @staticmethod
    def show():
        """
        Easy way to show the plot.
        """
        plt.show()

    @staticmethod
    def export_average_matrix_to_latex(
        file_path: str,
        average_matrix: dict[str, np.ndarray],
        average_angles: dict[str, np.ndarray] | None,
        average_translations: dict[str, np.ndarray] | None,
        angle_name: str,
        reference_system: JointCoordinateSystem,
        angle_in_degrees: bool = True,
    ):
        """
        Export the average reference system to a LaTeX table. The values are expected to be in the format output by
        the average_matrices function with the compute_std flag set to True and put in a dictionary with the key being
        the name of the reference system.

        Args:
        file_path (str): The path to the LaTeX file
        average_matrix (dict[str, np.ndarray]): The average matrices of the form {key: average_matrix}
        average_angles (dict[str, np.ndarray]): The average angles of the form {key: average_angles}
        average_translations (dict[str, np.ndarray]): The average translations of the form {key: average_translations}
        angle_names (str): The names of the angles
        reference_system (JointCoordinateSystem): The reference system
        angle_in_degrees (bool): Whether the angles are in degrees or radians
        """

        ncols = 2
        if average_angles is not None:
            ncols += 1
        if average_translations is not None:
            ncols += 1

        all_average_str = []
        for key in average_matrix.keys():
            matrix = average_matrix[key]
            key_string = key.name.replace("_", "\\_")

            row = f"{key_string} & \\begin{{tabular}}{{cccc}}\n"
            row += f"\\\\\n".join("&".join(f"{value:.4f}" for value in row) for row in matrix)
            row += f"\\end{{tabular}}"
            if average_angles is not None:
                angles = average_angles[key] * (180 / np.pi if angle_in_degrees else 1)
                row += f" & {np.mean(angles):.4f} ({np.std(angles):.4f})"
            if average_translations is not None:
                translations = average_translations[key]
                row += f" & {np.mean(translations):.4f} ({np.std(translations):.4f})"
            row += "\\\\\n"

            all_average_str.append(row)

        all_average_str = f"\\cmidrule(lr){{1-{ncols}}}\n".join(all_average_str)

        table_header = (
            f"\\makecell{{\\textbf{{From {reference_system.name}}} \\\\ \\textbf{{to}} }} & " 
            f"\\makecell{{\\textbf{{Average transformation}} \\\\ \\textbf{{matrix}}}}" 
            f"{" & \\makecell{\\textbf{Angle to mean (" f"{"rad" if not angle_in_degrees else "\\textdegree"}" ")} \\\\ \\textbf{Mean (SD)}}" if average_angles else ""}"
            f"{" & \\makecell{\\textbf{Translation to mean (mm)} \\\\ \\textbf{Mean (SD)}}" if average_translations else ""}"
        )
        latex_content = f"""
\\documentclass{{article}}
\\usepackage{{amsmath}}
\\usepackage{{array}}
\\usepackage{{booktabs}}
\\usepackage{{geometry}}
\\usepackage{{makecell}}

\\begin{{document}}

\\begin{{table}}[ht!]
\\centering
\\begin{{tabular}}{{l{"c" * (ncols - 1)}}}
\\toprule
{table_header} \\\\
\\midrule
{all_average_str}
\\bottomrule
\\end{{tabular}}
\\caption{{Average transformation matrix from {reference_system.name} for the {angle_name} matrices.}}
\\label{{tab:summary{reference_system.name}{angle_name}}}
\\end{{table}}

\\end{{document}}
        """

        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))

        with open(file_path, "w") as file:
            file.write(latex_content)
