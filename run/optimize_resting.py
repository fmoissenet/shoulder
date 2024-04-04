import casadi
import numpy as np
from shoulder import ModelBiorbd, ControlsTypes, MuscleParameter


# Add muscletendon equilibrium constraint
# Multivariate normal => center + noise , returns covariance matrix


def compute_qdot(x: casadi.MX | casadi.SX, model: ModelBiorbd, q: np.ndarray, qdot: np.ndarray, emg: np.ndarray):
    n_q = model.n_q
    n_muscles = model.n_muscles

    # Compute the dynamics using the MX as biorbd is MX based
    tendon_slack_lengths_mx = casadi.MX.sym("tendon_slack_lengths", n_muscles, 1)
    q_mx = casadi.MX.sym("q_mx", n_q, 1)
    qdot_mx = casadi.MX.sym("qdot_mx", n_q, 1)
    emg_mx = casadi.MX.sym("emg_mx", n_muscles, 1)
    for i in range(n_muscles):
        model.set_muscle_parameters(index=i, tendon_slack_length=tendon_slack_lengths_mx[i])
    xdot = model.forward_dynamics(q=q_mx, qdot=qdot_mx, controls=emg_mx, controls_type=ControlsTypes.EMG)

    # Convert the outputs to the type corresponding to the x vector and collapsing the graph at q, qdot and emg
    xdot = casadi.Function(
        "xdot",
        [q_mx, qdot_mx, emg_mx, tendon_slack_lengths_mx],
        xdot,
        ["q_in", "qdot_in", "emg_in", "tendon_slack_lengths_in"],
        ["qdot", "qddot"],
    )
    qddot = xdot(q_in=q, qdot_in=qdot, emg_in=emg, tendon_slack_lengths_in=x)["qddot"]

    return qddot


# TODO: Add optimization of the optimal lengths based on strongest position of the muscles (via q at qdot=0)


def find_minimal_tendon_slack_lengths(model: ModelBiorbd, emg: np.ndarray, q: np.array, qdot: np.array) -> np.array:
    """
    Find values for the tendon slack lengths where the muscle starts to produce passive muscle forces
    """

    # Declare some aliases
    n_muscles = model.n_muscles
    threshold = 1e-8

    # Get initial guesses from the model
    x = np.ones(n_muscles) * 0.0001

    # For each muscle, test if the muscle is producing passive forces
    tendon_slack_lengths_mx = casadi.MX.sym("tendon_slack_lengths", n_muscles, 1)
    emg_mx = casadi.MX.sym("emg", *emg.shape)
    q_mx = casadi.MX.sym("q", *q.shape)
    qdot_mx = casadi.MX.sym("qdot", *qdot.shape)

    for i in range(n_muscles):
        model.set_muscle_parameters(index=i, tendon_slack_length=tendon_slack_lengths_mx[i])

    muscle_forces = casadi.Function(
        "muscle_forces",
        [tendon_slack_lengths_mx, emg_mx, q_mx, qdot_mx],
        [model.muscle_force(emg_mx, q_mx, qdot_mx)],
        ["tendon_slack_lengths", "emg", "q", "qdot"],
        ["forces"],
    ).expand()

    # TODO: Add target force to the optimization problem
    for i in range(n_muscles):
        max_so_far = None
        min_so_far = None

        while True:
            # Update the muscle parameters
            model.set_muscle_parameters(index=i, tendon_slack_length=float(x[i]))

            # Get the muscle force
            force = muscle_forces(x, emg, q, qdot)[i]

            # If the muscle is producing passive forces, optimize the tendon slack length
            if force > 0 and force < threshold:
                break
            elif force > 0:
                min_so_far = x[i]
                if max_so_far is not None:
                    x[i] = (x[i] + max_so_far) / 2
                else:
                    x[i] *= 2
            else:
                max_so_far = x[i]
                x[i] = (x[i] + min_so_far) / 2

    return x


def optimize_tendon_slack_lengths(
    cx, model: ModelBiorbd, emg: np.ndarray, q: np.ndarray, qdot: np.ndarray
) -> np.ndarray:
    """
    Find values for the tendon slack lengths that do not produce any muscle force
    """

    # Declare some aliases
    n_muscles = model.n_muscles

    # Prepare the decision variables
    x = cx.sym("tendon_slack_lengths", n_muscles, 1)

    # Get initial guesses and bounds based on the model
    lbx = find_minimal_tendon_slack_lengths(model, emg, q, qdot)
    ubx = lbx * 1.25

    # Set arbitrary initial guess
    x0 = np.mean([lbx, ubx], axis=0)

    # Compute the cost functions
    obj = casadi.sum1(compute_qdot(x, model, q, qdot, emg) ** 2)

    # Compute some non-linear constraints
    g = cx()
    lbg = np.array([])
    ubg = np.array([])

    # Solve the program
    solver = casadi.nlpsol("solver", "ipopt", {"x": x, "f": obj, "g": g})
    sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

    return sol["x"]


def main():
    # Aliases
    cx = casadi.SX
    models = (ModelBiorbd("models/Wu_DeGroote.bioMod", use_casadi=True),)

    for model in models:
        n_q = model.n_q
        n_muscles = model.n_muscles
        q = np.zeros(n_q)
        qdot = np.zeros(n_q)
        emg = np.zeros(n_muscles)

        opimized_tendon_slack_lengths = optimize_tendon_slack_lengths(cx, model, emg, q, qdot)

        # Print the results
        print(f"The optimal tendon slack lengths are: {opimized_tendon_slack_lengths}")


if __name__ == "__main__":
    main()
