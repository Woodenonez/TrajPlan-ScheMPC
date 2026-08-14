from typing import Callable

import casadi as ca # type: ignore
from opengen import opengen as og # type: ignore # or "import opengen as og"

from . import mpc_helper as mh
from . import mpc_cost as mc

from configs import MpcConfiguration, CircularRobotSpecification


class PanocLightBuilder:
    """Build the MPC module via OPEN, using the simpler single-pass formulation from the
    `MPC_light` reference project rather than `builder_panoc.PanocBuilder`.

    Differences from `PanocBuilder`, kept intentionally rather than reconciled:
        - Fleet collision costs (current-step and predictive) are always active every
          horizon step; there is no `critical_step` cutoff.
        - Dynamic-obstacle costs are not wired in (upstream ships them commented out;
          the equivalent code is kept commented here too, for parity).
        - `build` takes the motion model directly instead of a separate
          `load_motion_model` call.

    Methods:
        build: Build the MPC problem and solver.
    """

    def __init__(self, mpc_config: MpcConfiguration, robot_config: CircularRobotSpecification):
        self._cfg = mpc_config
        self._spec = robot_config
        ### Frequently used
        self.ts = self._cfg.ts        # sampling time
        self.ns = self._cfg.ns        # number of states
        self.nu = self._cfg.nu        # number of inputs
        self.N_hor = self._cfg.N_hor  # control/pred horizon

    def build(self, motion_model: Callable[[ca.SX, ca.SX, float], ca.SX], use_tcp:bool=False, test:bool=False):
        """Build the MPC problem and solver, including states, inputs, cost, and constraints.

        Args:
            motion_model: Callable function `s'=f(s,u,ts)` that generates next state given the current state and action.
            use_tcp : If the solver will be called directly or via TCP.
            test : If the function is called for testing purposes, i.e. without building the solver.

        Notes:
            Inputs (u): speed, angular speed
            states (s): x, y, theta

        References:
            Ellipse definition: [https://math.stackexchange.com/questions/426150/what-is-the-general-equation-of-the-ellipse-that-is-not-in-the-origin-and-rotate]
        """
        print(f'[{self.__class__.__name__}] Building MPC module...')

        u = ca.SX.sym('u', self.nu*self.N_hor)      # 0. Inputs from 0 to N_hor-1
        u_m1 = ca.SX.sym('u_m1', self.nu)           # 1. Input at kt=-1
        s_0 = ca.SX.sym('s_0', self.ns)             # 2. State at kt=0
        s_N = ca.SX.sym('s_N', self.ns)             # 3. State of goal at kt=N_hor
        q = ca.SX.sym('q', self._cfg.nq)            # 4. Penalty for terms related to states/inputs

        r_s = ca.SX.sym('r_s', self.ns*self.N_hor)  # 5. Reference states
        r_v = ca.SX.sym('r_v', self.N_hor)          # 6. Reference speed

        c_0 = ca.SX.sym('c_0', self.ns*self._cfg.Nother)                 # 7. States of other robots at kt=0
        c = ca.SX.sym('c', self.ns*self.N_hor*self._cfg.Nother)          # 8. Predicted states of other robots

        o_s = ca.SX.sym('os', self._cfg.Nstcobs*self._cfg.nstcobs)                        # 9. Static obstacles
        o_d = ca.SX.sym('od', self._cfg.Ndynobs*self._cfg.ndynobs*(self.N_hor+1))         # 10. Dynamic obstacles
        q_stc = ca.SX.sym('qstc', self.N_hor)       # 11. Static obstacle weights
        q_dyn = ca.SX.sym('qdyn', self.N_hor)       # 12. Dynamic obstacle weights

        z = ca.vertcat(u_m1, s_0, s_N, q, r_s, r_v, c_0, c, o_s, o_d, q_stc, q_dyn)

        (x, y, theta) = (s_0[0], s_0[1], s_0[2])
        (x_goal, y_goal, theta_goal) = (s_N[0], s_N[1], s_N[2])
        (v_init, w_init) = (u_m1[0], u_m1[1])
        (qpos, qvel, qtheta, rv, rw) = (q[0], q[1], q[2], q[3], q[4])
        (qN, qthetaN, qrpd, acc_penalty, w_acc_penalty) = (q[5], q[6], q[7], q[8], q[9])

        ref_states = ca.reshape(r_s, (self.ns, self.N_hor)) # each column is a state
        ref_states = ca.horzcat(ref_states, ref_states[:, [-1]])[:2, :]

        cost = 0
        penalty_constraints = 0
        state = ca.vcat([x, y, theta])
        for kt in range(0, self.N_hor): # LOOP OVER PREDICTIVE HORIZON

            # Run step with motion model
            u_t = u[kt*self.nu:(kt+1)*self.nu]
            state = motion_model(state, u_t, self.ts)

            ### Reference deviation costs
            cost += mc.cost_refpath_deviation(state, ref_states[:, kt:], weight=qrpd)
            cost += qvel * (u_t[0]-r_v[kt])**2
            cost += ca.sum1(ca.vertcat(rv, rw) * u_t**2)

            ### Fleet collision avoidance (always active -- no critical-step cutoff)
            other_x_0 = c_0[ ::self.ns] # first  state
            other_y_0 = c_0[1::self.ns] # second state
            other_robots_0 = ca.hcat([other_x_0, other_y_0]).T # every column is a state of a robot
            cost += mc.cost_fleet_collision(state[:2], other_robots_0,
                                            safe_distance=2*(self._spec.vehicle_width+self._spec.vehicle_margin), weight=1000)

            ## Fleet collision avoidance [Predictive]
            other_robots_x = c[kt*self.ns  ::self.ns*self.N_hor] # first  state
            other_robots_y = c[kt*self.ns+1::self.ns*self.N_hor] # second state
            other_robots = ca.hcat([other_robots_x, other_robots_y]).T # every column is a state of a robot
            cost += mc.cost_fleet_collision(state[:2], other_robots,
                                            safe_distance=2*(self._spec.vehicle_width+self._spec.vehicle_margin), weight=10)

            ### Static obstacles
            for i in range(self._cfg.Nstcobs):
                eq_param = o_s[i*self._cfg.nstcobs : (i+1)*self._cfg.nstcobs]
                n_edges = int(self._cfg.nstcobs / 3) # 3 means b, a0, a1
                b, a0, a1 = eq_param[:n_edges], eq_param[n_edges:2*n_edges], eq_param[2*n_edges:]

                inside_stc_obstacle = mh.inside_cvx_polygon(state, b.T, a0.T, a1.T)
                penalty_constraints += ca.fmax(0, ca.vertcat(inside_stc_obstacle))
                cost += mc.cost_inside_cvx_polygon(state, b.T, a0.T, a1.T, weight=q_stc[kt])

            ### Dynamic obstacles -- disabled upstream (kept commented for parity with MPC_light)
            # x_dyn     = o_d[0::self._cfg.ndynobs*(self.N_hor+1)]
            # y_dyn     = o_d[1::self._cfg.ndynobs*(self.N_hor+1)]
            # rx_dyn    = o_d[2::self._cfg.ndynobs*(self.N_hor+1)]
            # ry_dyn    = o_d[3::self._cfg.ndynobs*(self.N_hor+1)]
            # As        = o_d[4::self._cfg.ndynobs*(self.N_hor+1)]
            # alpha_dyn = o_d[5::self._cfg.ndynobs*(self.N_hor+1)]
            #
            # inside_dyn_obstacle = mh.inside_ellipses(state.T, [x_dyn, y_dyn, rx_dyn, ry_dyn, As])
            # penalty_constraints += ca.fmax(0, inside_dyn_obstacle)
            #
            # ellipse_param = [x_dyn, y_dyn,
            #                  rx_dyn+self._spec.vehicle_margin+self._spec.social_margin,
            #                  ry_dyn+self._spec.vehicle_margin+self._spec.social_margin,
            #                  As, alpha_dyn]
            # cost += mc.cost_inside_ellipses(state.T, ellipse_param, weight=1000)

            ### Dynamic obstacles [Predictive] -- disabled upstream (kept commented for parity)
            # x_dyn     = o_d[(kt+1)*self._cfg.ndynobs  ::self._cfg.ndynobs*(self.N_hor+1)]
            # y_dyn     = o_d[(kt+1)*self._cfg.ndynobs+1::self._cfg.ndynobs*(self.N_hor+1)]
            # rx_dyn    = o_d[(kt+1)*self._cfg.ndynobs+2::self._cfg.ndynobs*(self.N_hor+1)]
            # ry_dyn    = o_d[(kt+1)*self._cfg.ndynobs+3::self._cfg.ndynobs*(self.N_hor+1)]
            # As        = o_d[(kt+1)*self._cfg.ndynobs+4::self._cfg.ndynobs*(self.N_hor+1)]
            # alpha_dyn = o_d[(kt+1)*self._cfg.ndynobs+5::self._cfg.ndynobs*(self.N_hor+1)]
            #
            # inside_dyn_obstacle = mh.inside_ellipses(state.T, [x_dyn, y_dyn, rx_dyn, ry_dyn, As])
            # penalty_constraints += ca.fmax(0, inside_dyn_obstacle)
            #
            # ellipse_param = [x_dyn, y_dyn,
            #                  rx_dyn+self._spec.vehicle_margin,
            #                  ry_dyn+self._spec.vehicle_margin,
            #                  As, alpha_dyn]
            # cost += mc.cost_inside_ellipses(state.T, ellipse_param, weight=q_dyn[kt])

        ### Terminal cost
        cost += qN*((state[0]-x_goal)**2 + (state[1]-y_goal)**2) + qthetaN*(state[2]-theta_goal)**2

        ### Max speed bound
        umin = [self._spec.lin_vel_min, -self._spec.ang_vel_max] * self.N_hor
        umax = [self._spec.lin_vel_max,  self._spec.ang_vel_max] * self.N_hor
        bounds = og.constraints.Rectangle(umin, umax)

        ### Acceleration bounds and cost
        v = u[0::2] # velocity
        w = u[1::2] # angular velocity
        acc   = (v-ca.vertcat(v_init, v[0:-1]))/self.ts
        w_acc = (w-ca.vertcat(w_init, w[0:-1]))/self.ts
        acc_constraints = ca.vertcat(acc, w_acc)
        # Acceleration bounds
        acc_min   = [ self._spec.lin_acc_min] * self.N_hor
        w_acc_min = [-self._spec.ang_acc_max] * self.N_hor
        acc_max   = [ self._spec.lin_acc_max] * self.N_hor
        w_acc_max = [ self._spec.ang_acc_max] * self.N_hor
        acc_bounds = og.constraints.Rectangle(acc_min + w_acc_min, acc_max + w_acc_max)
        # Accelerations cost
        cost += ca.mtimes(acc.T, acc)*acc_penalty
        cost += ca.mtimes(w_acc.T, w_acc)*w_acc_penalty

        problem = og.builder.Problem(u, z, cost) \
            .with_constraints(bounds) \
            .with_aug_lagrangian_constraints(acc_constraints, acc_bounds)
        problem.with_penalty_constraints(penalty_constraints)

        build_config = og.config.BuildConfiguration() \
            .with_build_directory(self._cfg.build_directory) \
            .with_build_mode(self._cfg.build_type)
        if not use_tcp:
            build_config.with_build_python_bindings()
        else:
            build_config.with_tcp_interface_config()

        meta = og.config.OptimizerMeta() \
            .with_optimizer_name(self._cfg.optimizer_name)

        solver_config = og.config.SolverConfiguration() \
            .with_initial_penalty(10) \
            .with_max_duration_micros(self._cfg.max_solver_time)

        builder = og.builder.OpEnOptimizerBuilder(problem, meta, build_config, solver_config) \
            .with_verbosity_level(1)
        if test:
            print(f"[{self.__class__.__name__}] MPC builder is tested without building.")
            return 1
        else:
            builder.build()

        print(f'[{self.__class__.__name__}] MPC module built.')
