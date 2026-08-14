from abc import ABC, abstractmethod
from typing import Any, Union
import yaml # type: ignore


PANOC_LIGHT_SUFFIX = "_light"


def panoc_light_optimizer_name(base_optimizer_name: str) -> str:
    """Derive the panoc_light build's optimizer name from the base PANOC one.

    `panoc` and `panoc_light` are built from the same MPC config file (same problem
    dimensions, just two different builder implementations), so the only thing that has
    to differ for them to coexist under the same `build_directory` is this name. Both
    `run_mpc.resolve_mpc_backend` and `build_solver.py` call this so the two scripts can
    never compute a different name for the same build.
    """
    if base_optimizer_name.endswith(PANOC_LIGHT_SUFFIX):
        return base_optimizer_name
    return base_optimizer_name + PANOC_LIGHT_SUFFIX


class Configurator:
    FIRST_LOAD = False
    def __init__(self, yaml_fp: str, with_partition=False) -> None:
        if Configurator.FIRST_LOAD:
            print(f'{self.__class__.__name__} Loading configuration from "{yaml_fp}".')
            Configurator.FIRST_LOAD = False
        if with_partition:
            yaml_load = self.from_yaml_all(yaml_fp)
        else:
            yaml_load = self.from_yaml(yaml_fp)
        for key in yaml_load:
            setattr(self, key, yaml_load[key])
            # getattr(self, key).__set_name__(self, key)

    @staticmethod
    def from_yaml(load_path) -> Union[dict, Any]:
        with open(load_path, 'r') as stream:
            try:
                parsed_yaml = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        return parsed_yaml
    
    @staticmethod
    def from_yaml_all(load_path) -> Union[dict, Any]:
        parsed_yaml = {}
        with open(load_path, 'r') as stream:
            try:
                for data in yaml.load_all(stream, Loader=yaml.FullLoader):
                    parsed_yaml.update(data)
            except yaml.YAMLError as exc:
                print(exc)
        return parsed_yaml


class _Configuration(ABC):
    """Base class for configuration/specification classes."""
    def __init__(self, config: Configurator) -> None:
        self._config = config
        self._load_config()

    @abstractmethod
    def _load_config(self):
        pass

    @classmethod
    def from_yaml(cls, yaml_fp: str, with_partition=False):
        config = Configurator(yaml_fp, with_partition)
        return cls(config)


class CircularRobotSpecification(_Configuration):
    """Specification class for circular robots."""
    def __init__(self, config: Configurator):
        super().__init__(config)

    def _load_config(self):
        config = self._config
        self.ts = config.ts     # sampling time

        self.vehicle_width = config.vehicle_width
        self.vehicle_margin = config.vehicle_margin
        self.social_margin = config.social_margin
        self.lin_vel_min = config.lin_vel_min
        self.lin_vel_max = config.lin_vel_max
        self.lin_acc_min = config.lin_acc_min
        self.lin_acc_max = config.lin_acc_max
        self.ang_vel_max = config.ang_vel_max
        self.ang_acc_max = config.ang_acc_max


class MpcConfiguration(_Configuration):
    """Configuration class for MPC Trajectory Generation Module."""
    def __init__(self, config: Configurator) -> None:
        super().__init__(config)

    def _load_config(self):
        config = self._config
        self.ts = config.ts        # sampling time

        self.N_hor = config.N_hor  # control/pred horizon
        self.action_steps = config.action_steps # number of action steps (normally 1)

        self.ns = config.ns        # number of states
        self.nu = config.nu        # number of inputs
        self.nq = config.nq        # number of penalties
        self.Nother = config.Nother   # number of other robots
        self.nstcobs = config.nstcobs # dimension of a static-obstacle description
        self.Nstcobs = config.Nstcobs # number of static obstacles
        self.ndynobs = config.ndynobs # dimension of a dynamic-obstacle description
        self.Ndynobs = config.Ndynobs # number of dynamic obstacles

        self.solver_type = config.solver_type           # Determines which solver to use ('PANOC', 'PANOC_LIGHT', or 'Casadi')

        self.max_solver_time = config.max_solver_time   # [P] maximum time for the solver to run
        self.build_directory = config.build_directory   # [P] directory to store the generated solver
        self.build_type = config.build_type             # [P] type of the generated solver
        self.bad_exit_codes = config.bad_exit_codes     # [P] bad exit codes of the solver
        self.optimizer_name = config.optimizer_name     # [P] name of the generated solver

        self.lin_vel_penalty = config.lin_vel_penalty   # Cost for linear velocity control action
        self.lin_acc_penalty = config.lin_acc_penalty   # Cost for linear acceleration
        self.ang_vel_penalty = config.ang_vel_penalty   # Cost for angular velocity control action
        self.ang_acc_penalty = config.ang_acc_penalty   # Cost for angular acceleration
        self.qrpd = config.qrpd                         # Cost for reference path deviation
        self.qpos = config.qpos                         # Cost for position deviation each time step to the reference
        self.qvel = config.qvel                         # Cost for speed    deviation each time step to the reference
        self.qtheta = config.qtheta                     # Cost for heading  deviation each time step to the reference
        self.qstcobs = config.qstcobs                   # Cost for static obstacle avoidance
        self.qdynobs = config.qdynobs                   # Cost for dynamic obstacle avoidance
        self.qpN = config.qpN                           # Terminal cost; error relative to final reference position
        self.qthetaN = config.qthetaN                   # Terminal cost; error relative to final reference heading

        # [C] Casadi-only knobs. These were hard-coded literals in casadi_impl.py; PANOC keeps
        # its own built-in values and ignores all of them.
        self.qfleet = config.qfleet                     # [C] Cost for collision with other robots, current step
        self.qfleet_pred = config.qfleet_pred           # [C] Cost for collision with other robots, over the horizon
        self.fleet_safe_distance = config.fleet_safe_distance          # [C] meters, or None to derive from robot spec
        self.fleet_critical_distance = config.fleet_critical_distance  # [C] meters, or None to derive from robot spec
        self.critical_step = config.critical_step       # [C] Horizon step past which current-step fleet/dyn terms stop
        self.obstacle_beta = config.obstacle_beta       # [C] Sharpness of the smooth obstacle costs
        self.rho_init = config.rho_init                 # [C] Penalty homotopy, initial weight
        self.rho_factor = config.rho_factor             # [C] Penalty homotopy, weight multiplier per outer iteration
        self.max_outer_iter = config.max_outer_iter     # [C] Penalty homotopy, outer solves per control step
        self.max_solver_iter = config.max_solver_iter   # [C] ipopt.max_iter

