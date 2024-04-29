from functools import partial
from packaging.version import Version

import biorbd
import biorbd_casadi
import casadi
import numpy as np
from scipy import integrate


from .enums import ControlsTypes, IntegrationMethods, MuscleParameter
from .model_abstract import ModelAbstract
from .parameters_optimization import optimize_muscle_parameters
from ..helpers import Vector, Scalar, VectorHelpers


assert Version(biorbd.__version__) == Version("1.11.2")
assert Version(biorbd_casadi.__version__) == Version("1.11.2")


class ModelBiorbd(ModelAbstract):
    def __init__(self, model: str | ModelAbstract, use_casadi: bool = False):
        self._use_casadi = use_casadi
        self.brbd = biorbd_casadi if self._use_casadi else biorbd
        if isinstance(model, str):
            self._model = self.brbd.Model(model)
        elif isinstance(model, ModelBiorbd):
            self._model = self.brbd.Model(model._model.path().absolutePath().to_string())
        else:
            raise ValueError("model_path must be a string or a ModelBiorbd object")

    @property
    def biorbd_model(self) -> biorbd.Model | biorbd_casadi.Model:
        return self._model

    @property
    def name(self) -> str:
        return self._model.path().absolutePath().to_string().split("/")[-1].split(".bioMod")[0]

    @property
    def n_q(self) -> int:
        return self._model.nbQ()

    @property
    def q_ranges(self) -> np.ndarray:
        q_ranges = []
        for segment in self._model.segments():
            for range in segment.QRanges():
                q_ranges.append([range.min(), range.max()])
        return np.array(q_ranges)

    @property
    def n_muscles(self) -> int:
        return self._model.nbMuscleTotal()

    @property
    def muscle_names(self) -> list[str]:
        return [name.to_string() for name in self._model.muscleNames()]

    def optimize_muscle_parameters(
        self, use_predefined_muscle_ratio_values: bool = True, robust_optimization: bool = False, expand: bool = True
    ) -> None:
        results = optimize_muscle_parameters(
            cx=casadi.SX,
            model=self._model if self._use_casadi else ModelBiorbd(self, use_casadi=True),
            use_predefined_muscle_ratio_values=use_predefined_muscle_ratio_values,
            robust_optimization=robust_optimization,
            expand=expand,
        )
        for i in range(self.n_muscles):
            self.set_muscle_parameters(
                i,
                optimal_length=results.optimal_lengths.values[i],
                tendon_slack_length=results.tendon_slack_lengths.values[i],
            )

    @property
    def relaxed_pose(self) -> np.ndarray:
        return np.array([0, -0.01, 0, 0])

    def ranged_relaxed_poses(self, limit: float, n_elements: int) -> np.ndarray:
        if n_elements == 1:
            return self.relaxed_pose[np.newaxis, :].T

        poses = []
        for plane_of_elevation in np.random.normal(0, limit, n_elements):
            for elevation in np.random.normal(0, limit, n_elements):
                poses.append(self.relaxed_pose + np.array([plane_of_elevation, np.abs(elevation), 0, 0]))
        return np.concatenate((self.relaxed_pose[np.newaxis, :], poses[:-1])).T

    @property
    def strongest_poses(self) -> dict[str, np.ndarray]:
        return {
            "DELT1": np.array([0, -np.pi / 2, 0, 0]),  # [2.02282, -2.68809, 0.778794, 1.2031]
            "DELT2": np.array([0, -np.pi / 2, 0, 0]),
            "DELT3": np.array([0, -np.pi / 2, 0, 0]),
            "TRIlong": np.array([0, -0.01, 0, np.pi / 2]),
            "INFSP": np.array([0, -0.01, 0, 0]),
            "SUPSP": np.array([0, -0.01, 0, 0]),  # [0.94661, -1.00282, 1.56935, 1.2031]
            "SUBSC": np.array([1, -0.5, 0, 0]),  # [0.944205, -0.40017, -0.0495518, 1.2031]
            "TMIN": np.array([0, -0.01, 0, 0]),
            "TMAJ": np.array([0, -0.01, 0, 0]),
            "CORB": np.array([0, -0.01, 0, 0]),
            "PECM1": np.array([0, -np.pi / 2, 0, 0]),  # [2.35619, -1.92047, 0.302886, 1.2031]
            "PECM2": np.array([0, -np.pi / 2, 0, 0]),
            "PECM3": np.array([0, -np.pi / 2, 0, 0]),
            "LAT": np.array([0, -np.pi / 2, 0, 0]),
            "BIClong": np.array([0, -0.01, 0, np.pi / 2]),
            "BICshort": np.array([0, -0.01, 0, np.pi / 2]),
        }

    def muscles_kinematics(
        self, q: Vector, qdot: Vector = None, muscle_index: range | slice | int = None
    ) -> Vector | tuple[Vector, Vector]:
        data_type = type(q)
        if qdot is not None and not isinstance(qdot, data_type):
            raise ValueError("q and qdot must have the same type")

        muscle_index = VectorHelpers.index_to_slice(muscle_index, self.n_muscles)
        muscle_index = list(range(*muscle_index.indices(self.n_muscles)))

        if len(q.shape) == 1:
            q = q[:, np.newaxis]
        if qdot is not None and len(qdot.shape) == 1:
            qdot = qdot[:, np.newaxis]

        n_muscles = len(muscle_index)
        data_type = type(q)

        lengths = VectorHelpers.initialize(data_type, n_muscles, q.shape[1])
        velocities = VectorHelpers.initialize(data_type, n_muscles, q.shape[1])
        for i in range(q.shape[1]):
            if qdot is None:
                self._model.updateMuscles(q[:, i], True)
            else:
                self._model.updateMuscles(q[:, i], qdot[:, i], True)

            for j in range(n_muscles):
                mus = self._upcast_muscle(self._model.muscle(muscle_index[j]))
                length_tp = mus.length(self._model, q[:, i], False)
                if self._use_casadi:
                    length_tp = length_tp.to_mx()
                    if data_type == np.ndarray:
                        length_tp = casadi.Function("length", [], [length_tp], [], ["length"])()["length"]
                lengths[j, i] = length_tp

                if qdot is not None:
                    velocity_tp = mus.velocity(self._model, q[:, i], qdot[:, i], False)
                    if self._use_casadi:
                        velocity_tp = velocity_tp.to_mx()
                        if data_type == np.ndarray:
                            velocity_tp = casadi.Function("velocity", [], [velocity_tp], [], ["velocity"])()["velocity"]
                    velocities[j, i] = velocity_tp

        if qdot is None:
            return lengths
        else:
            return lengths, velocities

    def muscle_force_coefficients(
        self,
        emg: Vector,
        q: Vector,
        qdot: Vector = None,
        muscle_index: int | range | slice | None = None,
    ) -> Vector | tuple[Vector, Vector, Vector]:
        data_type = type(emg)
        if not isinstance(q, data_type) or (qdot is not None and not isinstance(qdot, data_type)):
            raise ValueError("emg, q and qdot must have the same type")

        muscle_index = VectorHelpers.index_to_slice(muscle_index, self.n_muscles)
        muscle_index = list(range(*muscle_index.indices(self.n_muscles)))

        if len(q.shape) == 1:
            q = q[:, np.newaxis]
        if qdot is not None and len(qdot.shape) == 1:
            qdot = qdot[:, np.newaxis]

        n_muscles = len(muscle_index)

        out_flpe = VectorHelpers.initialize(data_type, n_muscles, q.shape[1])
        out_flce = VectorHelpers.initialize(data_type, n_muscles, q.shape[1])
        out_fvce = VectorHelpers.initialize(data_type, n_muscles, q.shape[1])
        for i in range(q.shape[1]):
            if qdot is None:
                self._model.updateMuscles(q[:, i], True)
            else:
                self._model.updateMuscles(q[:, i], qdot[:, i], True)

            for j in range(n_muscles):
                mus = self._upcast_muscle(self._model.muscle(muscle_index[j]))
                activation = self.brbd.State(emg[j, i], emg[j, i])

                flpe_tp = mus.FlPE()
                if self._use_casadi:
                    flpe_tp = flpe_tp.to_mx()
                    if data_type == np.ndarray:
                        flpe_tp = casadi.Function("flpe", [], [flpe_tp], [], ["flpe"])()["flpe"]
                out_flpe[j, i] = flpe_tp

                flce_tp = mus.FlCE(activation)
                if self._use_casadi:
                    flce_tp = flce_tp.to_mx()
                    if data_type == np.ndarray:
                        flce_tp = casadi.Function("flce", [], [flce_tp], [], ["flce"])()["flce"]
                out_flce[j, i] = flce_tp

                if qdot is not None:
                    flve_tp = mus.FvCE()
                    if self._use_casadi:
                        flve_tp = flve_tp.to_mx()
                        if data_type == np.ndarray:
                            flve_tp = casadi.Function("flve", [], [flve_tp], [], ["flve"])()["flve"]
                    out_fvce[j, i] = flve_tp

        if qdot is None:
            return out_flpe, out_flce
        else:
            return out_flpe, out_flce, out_fvce

    def muscle_force(
        self, emg: Vector, q: Vector, qdot: Vector, muscle_index: int | range | slice | None = None
    ) -> Vector:
        data_type = type(emg)
        if not isinstance(q, data_type) or not isinstance(qdot, data_type):
            raise ValueError("emg, q and qdot must have the same type")
        if len(emg.shape) == 1:
            emg = emg[:, np.newaxis]

        muscle_index = VectorHelpers.index_to_slice(muscle_index, self.n_muscles)
        muscle_index = list(range(*muscle_index.indices(self.n_muscles)))

        if len(q.shape) == 1:
            q = q[:, np.newaxis]
        if len(qdot.shape) == 1:
            qdot = qdot[:, np.newaxis]

        n_muscles = len(muscle_index)

        out_force = VectorHelpers.initialize(data_type, n_muscles, q.shape[1])
        for i in range(q.shape[1]):
            if qdot is None:
                self._model.updateMuscles(q[:, i], True)
            else:
                self._model.updateMuscles(q[:, i], qdot[:, i], True)

            for j in range(n_muscles):
                mus = self._upcast_muscle(self._model.muscle(muscle_index[j]))
                activation = self.brbd.State(emg[j, i], emg[j, i])
                force_tp = mus.force(activation)
                if self._use_casadi:
                    force_tp = force_tp.to_mx()
                    if data_type == np.ndarray:
                        force_tp = casadi.Function("force", [], [force_tp], [], ["force"])()["force"]
                out_force[j, i] = force_tp

        return out_force

    def set_muscle_parameters(
        self, index: int, optimal_length: Scalar = None, tendon_slack_length: Scalar = None
    ) -> None:
        if optimal_length is not None:
            self._model.muscle(index).characteristics().setOptimalLength(optimal_length)

        if tendon_slack_length is not None:
            self._model.muscle(index).characteristics().setTendonSlackLength(tendon_slack_length)

    def get_muscle_parameter(self, index: int, parameter_to_get: MuscleParameter) -> Scalar:
        if parameter_to_get == MuscleParameter.OPTIMAL_LENGTH:
            if self._use_casadi:
                return self._model.muscle(index).characteristics().optimalLength().to_mx()
            else:
                return self._model.muscle(index).characteristics().optimalLength()
        elif parameter_to_get == MuscleParameter.TENDON_SLACK_LENGTH:
            if self._use_casadi:
                return self._model.muscle(index).characteristics().tendonSlackLength().to_mx()
            else:
                return self._model.muscle(index).characteristics().tendonSlackLength()
        elif parameter_to_get == MuscleParameter.PENNATION_ANGLE:
            if self._use_casadi:
                return self._model.muscle(index).characteristics().pennationAngle().to_mx()
            else:
                return self._model.muscle(index).characteristics().pennationAngle()
        elif parameter_to_get == MuscleParameter.MAXIMAL_FORCE:
            if self._use_casadi:
                return self._model.muscle(index).characteristics().forceIsoMax().to_mx()
            else:
                return self._model.muscle(index).characteristics().forceIsoMax()
        else:
            raise NotImplementedError(f"Parameter {parameter_to_get} not implemented")

    def integrate(
        self,
        t: Vector,
        states: Vector,
        controls: Vector,
        controls_type: ControlsTypes = ControlsTypes.TORQUE,
        integration_method: IntegrationMethods = IntegrationMethods.RK45,
    ) -> tuple[Vector, Vector]:
        data_type = type(states)
        if not isinstance(controls, data_type):
            raise ValueError("states and controls must have the same type")

        func = partial(self.forward_dynamics, tau=controls)

        if controls_type == ControlsTypes.EMG:
            if controls.shape[0] != self.n_muscles:
                raise ValueError(f"EMG controls should have {self.n_muscles} muscles, but got {controls.shape[0]}")
            func = partial(self._forward_dynamics_muscles, emg=controls)
        elif controls_type == ControlsTypes.TORQUE:
            if controls.shape[0] != self.n_q:
                raise ValueError(
                    f"Torque controls should have {self.n_q} generalized coordinates, but got {controls.shape[0]}"
                )
            func = partial(self._forward_dynamics, tau=controls)
        else:
            raise NotImplementedError(f"Control {controls_type} not implemented")

        t_span = (t[0], t[-1])
        results = integrate.solve_ivp(fun=func, t_span=t_span, y0=states, method=integration_method.value, t_eval=t)

        q = results.y[: self.n_q, :]
        qdot = results.y[self.n_q :, :]
        return q, qdot

    def forward_dynamics(
        self,
        q: Vector,
        qdot: Vector,
        controls: Vector,
        controls_type: ControlsTypes = ControlsTypes.TORQUE,
    ) -> tuple[Vector, Vector]:
        data_type = type(q)
        if not isinstance(qdot, data_type) or not isinstance(controls, data_type):
            raise ValueError("q, qdot and controls must have the same type")

        if controls_type == ControlsTypes.EMG:
            if controls.shape[0] != self.n_muscles:
                raise ValueError(f"EMG controls should have {self.n_muscles} muscles, but got {controls.shape[0]}")
            func = partial(self._forward_dynamics_muscles, t=[], emg=controls)
        elif controls_type == ControlsTypes.TORQUE:
            if controls.shape[0] != self.n_q:
                raise ValueError(
                    f"Torque controls should have {self.n_q} generalized coordinates, but got {controls.shape[0]}"
                )
            func = partial(self._forward_dynamics, t=[], tau=controls)
        else:
            raise NotImplementedError(f"Control {controls_type} not implemented")

        results = func(x=VectorHelpers.concatenate(q, qdot))
        q = results[: self.n_q]
        qdot = results[self.n_q :]
        return q, qdot

    def _forward_dynamics_muscles(self, t: float, x: Vector, emg: Vector) -> Vector:
        q = x[: self.n_q]
        qdot = x[self.n_q :]
        states = self._model.stateSet()
        for k in range(self._model.nbMuscles()):
            states[k].setExcitation(emg[k])
            states[k].setActivation(emg[k])
        tau = self._model.muscularJointTorque(states, q, qdot)
        tau = tau.to_mx() if self._use_casadi else tau.to_array()

        return self._forward_dynamics(t, x, tau)

    def _forward_dynamics(self, t: float, x: Vector, tau: Vector) -> Vector:
        q = x[: self.n_q]
        qdot = x[self.n_q :]
        qddot = self._model.ForwardDynamics(q, qdot, tau)
        qddot = qddot.to_mx() if self._use_casadi else qddot.to_array()

        return VectorHelpers.concatenate(qdot, qddot)

    def animate(self, states: list[Vector]) -> None:
        import bioviz

        viz = bioviz.Viz(
            loaded_model=self._model,
            show_local_ref_frame=False,
            show_segments_center_of_mass=False,
            show_global_center_of_mass=False,
            show_gravity_vector=False,
        )
        viz.load_movement(states[0])
        viz.set_camera_roll(np.pi / 2)
        viz.exec()

    def _upcast_muscle(
        self, muscle: biorbd.Muscle | biorbd_casadi.Muscle
    ) -> (
        biorbd.HillType
        | biorbd_casadi.HillType
        | biorbd.HillThelenType
        | biorbd_casadi.HillThelenType
        | biorbd.HillDeGrooteType
        | biorbd_casadi.HillDeGrooteType
    ):
        muscle_type_id = muscle.type()
        if muscle_type_id == self.brbd.IDEALIZED_ACTUATOR:
            return self.brbd.IdealizedActuator(muscle)
        elif muscle_type_id == self.brbd.HILL:
            return self.brbd.HillType(muscle)
        elif muscle_type_id == self.brbd.HILL_THELEN:
            return self.brbd.HillThelenType(muscle)
        elif muscle_type_id == self.brbd.HILL_DE_GROOTE:
            return self.brbd.HillDeGrooteType(muscle)
        else:
            raise ValueError(f"Muscle type {muscle_type_id} not supported")
