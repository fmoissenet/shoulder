import casadi
import numpy as np
from shoulder import optimize_muscle_parameters, ModelBiorbd


# Add muscletendon equilibrium constraint
# Multivariate normal => center + noise , returns covariance matrix (Robust optimization)
# TODO Does the model target the same pose as the astronaut?


def main():
    # Aliases
    cx = casadi.SX
    use_predefined_muscle_ratio_values = True
    robust_optimization = True
    models = (
        ModelBiorbd("models/Wu_DeGroote.bioMod", use_casadi=True),
        # ModelBiorbd("models/Wu_Thelen.bioMod", use_casadi=True),
    )
    expand = True

    results = []
    for model in models:
        np.random.seed(42)
        # Optimize
        results.append(
            optimize_muscle_parameters(
                cx, model, use_predefined_muscle_ratio_values, robust_optimization, expand, verbose=True
            )
        )

    for result in results:
        print(result)


if __name__ == "__main__":
    main()
