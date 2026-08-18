import os
import pathlib

from configs import MpcConfiguration, CircularRobotSpecification, panoc_light_optimizer_name

from basic_motion_model import motion_model
from pkg_mpc_tracker.casadi_build import builder_panoc
from pkg_mpc_tracker.casadi_build import builder_panoc_light

def return_cfg_path(fname: str) -> str:
    root_dir = pathlib.Path(__file__).resolve().parents[1]
    cfg_path = os.path.join(root_dir, "config", fname)
    return cfg_path

def load_mpc_config(fname: str) -> MpcConfiguration:
    """Load the MPC configuration."""
    return MpcConfiguration.from_yaml(return_cfg_path(fname))

def load_robot_spec(fname: str) -> CircularRobotSpecification:
    """Load the robot specification."""
    return CircularRobotSpecification.from_yaml(return_cfg_path(fname))

if __name__ == "__main__":
    cfg_fname = "mpc_default.yaml"
    robot_spec = "robot_spec.yaml"
    # "panoc" builds builder_panoc.PanocBuilder (the project's own, more developed PANOC
    # formulation); "panoc_light" builds builder_panoc_light.PanocLightBuilder (ported from
    # the MPC_light reference project: no critical-step cutoff on the fleet terms, dynamic
    # obstacles not wired in). Both variants share `cfg_fname`'s problem dimensions and only
    # need distinct optimizer names to land in separate `mpc_solver/<name>/` directories --
    # see `configs.panoc_light_optimizer_name`. This must match `mpc_backend` at run time
    # (run_mpc.py's `MPC_BACKENDS`), and `cfg_fname` here must match run_mpc.py's `CFG_FNAME`.
    panoc_builder = "panoc"  # "panoc" or "panoc_light"

    config_mpc = load_mpc_config(cfg_fname)
    config_robot = load_robot_spec(robot_spec)

    if panoc_builder == "panoc_light":
        config_mpc.optimizer_name = panoc_light_optimizer_name(config_mpc.optimizer_name)
        mpc_module = builder_panoc_light.PanocLightBuilder(config_mpc, config_robot)
        mpc_module.build(motion_model.unicycle_model, test=False)
    else:
        mpc_module = builder_panoc.PanocBuilder(config_mpc, config_robot)
        mpc_module.load_motion_model(motion_model.unicycle_model)
        mpc_module.build(test=False)
