from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, TypedDict, cast

import casadi as ca  # type: ignore

from configs import CircularRobotSpecification, MpcConfiguration

from . import mpc_cost as mc
from . import mpc_helper as mh


class PenaltyTerms(TypedDict):
    pos: ca.SX
    vel: ca.SX
    theta: ca.SX
    v: ca.SX
    w: ca.SX
    posN: ca.SX
    thetaN: ca.SX
    rpd: ca.SX
    acc_penalty: ca.SX
    w_acc_penalty: ca.SX


@dataclass(frozen=True)
class CasadiProblem:
    """Container returned by build() to keep symbolic objects together."""

    solver: ca.Function
    w: ca.SX # decision vector [x,u]
    p: ca.SX # Parameter vector
    g: ca.SX # constraint expression vector
    lbw: list[float]
    ubw: list[float]
    lbg: list[float]
    ubg: list[float]


class CasadiNMPC:
    """Learning-oriented CasADi NMPC scaffold aligned with PanocBuilder.

    Important:
    - Parameter vector layout matches `builder_panoc.py` exactly.
    - Decision variables use direct multiple shooting: `[x_0, u_0, x_1, ..., u_{N-1}, x_N]`.
    - Dynamics and acceleration constraints are implemented.
    - Obstacle terms are included as soft costs; hard constraints can be added as TODO.
    """

    _large_weight = 1.0 # 1000
    _small_weight = 0.5 # 10
    _critical_step = 100
    _penalty_weight = 10.0 # Modelling parameter, can be changed!

    def __init__(self, mpc_config: MpcConfiguration, robot_config: CircularRobotSpecification):
        self._cfg = mpc_config
        self._spec = robot_config
        self.ts = self._cfg.ts # sampling time
        self.ns = self._cfg.ns # number of states
        self.nu = self._cfg.nu # number of inputs
        self.N_hor = self._cfg.N_hor #control/pred horizon

        self._motion_model: Callable[[ca.SX, ca.SX, float], ca.SX] | None = None
        self._load_parameters()

    @classmethod
    def from_yaml(cls, mpc_cfg_fpath: str, robot_cfg_fpath: str) -> "CasadiNMPC":
        mpc_config = MpcConfiguration.from_yaml(mpc_cfg_fpath)
        robot_config = CircularRobotSpecification.from_yaml(robot_cfg_fpath)
        return cls(mpc_config, robot_config)

    def load_motion_model(self, motion_model: Callable[[ca.SX, ca.SX, float], ca.SX]) -> None:
        self._motion_model = motion_model


        # One parameter less, since multiple shooting. u is not dedcision varaible.
    def _load_parameters(self) -> None:
        """Define parameter blocks with the same ordering as PANOC builder."""
        nu, ns, N = self.nu, self.ns, self.N_hor

        # One parameter less, since multiple shooting. u is not dedcision varaible.

        self._u_m1 = ca.SX.sym("u_m1", nu)  # 1. previous input
        self._s_0 = ca.SX.sym("s_0", ns)  # 2. current state
        self._s_N = ca.SX.sym("s_N", ns)  # 3. terminal reference state
        self._q = ca.SX.sym("q", self._cfg.nq)  # 4. penalty weights
        self._r_s = ca.SX.sym("r_s", ns * N)  # 5. reference states
        self._r_v = ca.SX.sym("r_v", N)  # 6. reference speed
        self._c_0 = ca.SX.sym("c_0", ns * self._cfg.Nother)  # 7. other robots @ current step
        self._c = ca.SX.sym("c", ns * N * self._cfg.Nother)  # 8. other robots predicted
        self._o_s = ca.SX.sym("o_s", self._cfg.Nstcobs * self._cfg.nstcobs)  # 9. static obstacles
        self._o_d = ca.SX.sym("o_d", self._cfg.Ndynobs * self._cfg.ndynobs * (N + 1))  # 10. dynamic obstacles
        self._q_stc = ca.SX.sym("q_stc", N)  # 11. static obstacle weights
        self._q_dyn = ca.SX.sym("q_dyn", N)  # 12. dynamic obstacle weights
        # adding penalty parameter
        self._rho_pen = ca.SX.sym("rho_pen", 1)
        # Parameter vector
        self._p = ca.vertcat(
            self._u_m1,
            self._s_0,
            self._s_N,
            self._q,
            self._r_s,
            self._r_v,
            self._c_0,
            self._c,
            self._o_s,
            self._o_d,
            self._q_stc,
            self._q_dyn,
            self._rho_pen # added
        )
        self._p = cast(ca.SX, self._p)

        self._q_terms: PenaltyTerms = {
            "pos": self._q[0],
            "vel": self._q[1],
            "theta": self._q[2],
            "v": self._q[3],
            "w": self._q[4],
            "posN": self._q[5],
            "thetaN": self._q[6],
            "rpd": self._q[7],
            "acc_penalty": self._q[8],
            "w_acc_penalty": self._q[9],
        }

    def _other_robots_at_step(self, k: int) -> ca.SX:
        x = self._c[k * self.ns :: self.ns * self.N_hor]
        y = self._c[k * self.ns + 1 :: self.ns * self.N_hor]
        return ca.hcat([x, y]).T

    def _other_robots_current(self) -> ca.SX:
        x0 = self._c_0[:: self.ns]
        y0 = self._c_0[1:: self.ns]
        return ca.hcat([x0, y0]).T

    @staticmethod
    def _wrapped_angle_error(theta: ca.SX, theta_ref: ca.SX) -> ca.SX:
        """Return the shortest signed angular difference in [-pi, pi]."""

        delta = theta - theta_ref
        return ca.atan2(ca.sin(delta), ca.cos(delta))

    def _stage_cost(self, k: int, x_next: ca.SX, u_k: ca.SX, ref_states: ca.SX) -> ca.SX:
        """Per-step cost. Mirrors PanocBuilder terms as soft penalties."""
        cts = mc.CostTerms()
        theta_err = mh.angle_error(x_next[2], ref_states[2, 0])

        ### Reference deviation costs J_R =||s_k- s(tilde)_k ||*Qs + ||u_k - u(tilde)_k||
        ### the term ||u_k - u_k-1 || is performed in build().
        cts.cost_pos = self._q_terms["pos"] * (
            (x_next[0] - ref_states[0, 0]) ** 2 + (x_next[1] - ref_states[1, 0]) ** 2
        )
        cts.cost_rpd = mc.cost_refpath_deviation(x_next, ref_states[:2, :], weight=self._q_terms["rpd"]) # state in x,y
        cts.cost_rvd = self._q_terms["vel"] * (u_k[0] - self._r_v[k]) ** 2 # control action term
        theta_error = self._wrapped_angle_error(x_next[2], ref_states[2, 0])
        cts.cost_rtd = self._q_terms["theta"] * theta_error**2 # state in theta
        cts.cost_input = ca.sum1(ca.vertcat(self._q_terms["v"], self._q_terms["w"]) * u_k**2) # ||u_k||Q_u

        ### Fleet collision avoidance: J_f =  max(0,Q_f * (d_fleet - distance))**2
        ### used from mpc_cost, cost_fleet_collision.
        safe_distance = 0.1 #2 * (self._spec.vehicle_width + self._spec.vehicle_margin)
        critical_distance = 0.05 #2 * self._spec.vehicle_width + self._spec.vehicle_margin
        if k < self._critical_step:
            cts.cost_fleet = mc.cost_fleet_collision(
                x_next[:2],
                self._other_robots_current(),
                safe_distance=critical_distance,
                weight=self._large_weight,
            )

        # ## Fleet collision avoidance [Predictive]
        # cts.cost_fleet_pred = mc.cost_fleet_collision(
        #     x_next[:2],
        #     self._other_robots_at_step(k),
        #     safe_distance=safe_distance,
        #     weight=self._small_weight,
        # )
        ### J_O dynamic/static obstacle costs, similar to PANOC implementation.
        #cts.cost_dynobs = self._dynamic_obstacle_current_cost(k, x_next)
        #cts.cost_stcobs = self._static_obstacle_cost(x_next, self._q_stc[k])
        #cts.cost_dynobs_pred = self._dynamic_obstacle_cost(k, x_next, self._q_dyn[k])
        # penalty_constraints_stcobs = self._penalty_weight * self._static_obstacle_intrusion(x_next)
        # penalty_constraints_dynobs = self._penalty_weight * self._dynamic_obstacle_intrusion(k, x_next)
        return cts.sum() # + penalty_constraints_stcobs + penalty_constraints_dynobs

    def _static_obstacle_cost(self, state: ca.SX, weight: ca.SX) -> ca.SX:
        cost = ca.SX(0.0)
        for i in range(self._cfg.Nstcobs):
            eq_param = self._o_s[i * self._cfg.nstcobs : (i + 1) * self._cfg.nstcobs]
            n_edges = int(self._cfg.nstcobs / 3)
            b = eq_param[:n_edges]
            a0 = eq_param[n_edges : 2 * n_edges]
            a1 = eq_param[2 * n_edges :]
            # cost += mc.cost_inside_cvx_polygon(state, b.T, a0.T, a1.T, weight=weight)
            cost += mc.cost_inside_cvx_polygon(state, b.T, a0.T, a1.T, weight=weight)
        return cost

    def _dynamic_obstacle_cost(self, k: int, state: ca.SX, weight: ca.SX) -> ca.SX:
        # Predictive terms (k+1) use the same indexing layout as PanocBuilder.
        cost = ca.SX(0.0)
        x_dyn = self._o_d[(k + 1) * self._cfg.ndynobs :: self._cfg.ndynobs * (self.N_hor + 1)]
        y_dyn = self._o_d[(k + 1) * self._cfg.ndynobs + 1 :: self._cfg.ndynobs * (self.N_hor + 1)]
        rx_dyn = self._o_d[(k + 1) * self._cfg.ndynobs + 2 :: self._cfg.ndynobs * (self.N_hor + 1)]
        ry_dyn = self._o_d[(k + 1) * self._cfg.ndynobs + 3 :: self._cfg.ndynobs * (self.N_hor + 1)]
        ang_dyn = self._o_d[(k + 1) * self._cfg.ndynobs + 4 :: self._cfg.ndynobs * (self.N_hor + 1)]
        alpha_dyn = self._o_d[(k + 1) * self._cfg.ndynobs + 5 :: self._cfg.ndynobs * (self.N_hor + 1)]
        ellipse_param = [
            x_dyn,
            y_dyn,
            rx_dyn + self._spec.vehicle_margin,
            ry_dyn + self._spec.vehicle_margin,
            ang_dyn,
            alpha_dyn,
        ]
        cost += mc.cost_inside_ellipses(state.T, ellipse_param, weight=weight) 
        return cost #mc.cost_inside_ellipses_smooth(state.T, ellipse_param, weight=weight) # mc.cost_inside_ellipses(state.T, ellipse_param, weight=weight)

    def _dynamic_obstacle_current_cost(self, k: int, state: ca.SX) -> ca.SX:
        cost = ca.SX(0.0)
        if k >= self._critical_step:
            return ca.SX(0.0)
        x_dyn = self._o_d[0 :: self._cfg.ndynobs * (self.N_hor + 1)]
        y_dyn = self._o_d[1 :: self._cfg.ndynobs * (self.N_hor + 1)]
        rx_dyn = self._o_d[2 :: self._cfg.ndynobs * (self.N_hor + 1)]
        ry_dyn = self._o_d[3 :: self._cfg.ndynobs * (self.N_hor + 1)]
        ang_dyn = self._o_d[4 :: self._cfg.ndynobs * (self.N_hor + 1)]
        alpha_dyn = self._o_d[5 :: self._cfg.ndynobs * (self.N_hor + 1)]
        ellipse_param = [
            x_dyn,
            y_dyn,
            rx_dyn + self._spec.vehicle_margin + self._spec.social_margin,
            ry_dyn + self._spec.vehicle_margin + self._spec.social_margin,
            ang_dyn,
            alpha_dyn,
        ]
        cost += mc.cost_inside_ellipses(state.T, ellipse_param, weight=self._large_weight) 
        return cost # mc.cost_inside_ellipses_smooth(state.T, ellipse_param, weight=self._large_weight)  #mc.cost_inside_ellipses(state.T, ellipse_param, weight=self._large_weight)

    def _static_obstacle_intrusion(self, state: ca.SX) -> ca.SX:
        inside_stc: list[ca.SX] = []
        for i in range(self._cfg.Nstcobs):
            eq_param = self._o_s[i * self._cfg.nstcobs : (i + 1) * self._cfg.nstcobs]
            n_edges = int(self._cfg.nstcobs / 3)
            b = eq_param[:n_edges]
            a0 = eq_param[n_edges : 2 * n_edges]
            a1 = eq_param[2 * n_edges :]
            #inside_stc.append(mh.smooth_cvx_intrusion(state, b.T, a0.T, a1.T))
            inside_stc.append(mh.inside_cvx_polygon(state, b.T, a0.T, a1.T))
        if len(inside_stc) == 0:
            return ca.SX.zeros(0, 1)
        return ca.vertcat(*inside_stc)

    def _dynamic_obstacle_intrusion(self, k: int, state: ca.SX) -> ca.SX:
        x_dyn = self._o_d[0 :: self._cfg.ndynobs * (self.N_hor + 1)]
        y_dyn = self._o_d[1 :: self._cfg.ndynobs * (self.N_hor + 1)]
        rx_dyn = self._o_d[2 :: self._cfg.ndynobs * (self.N_hor + 1)]
        ry_dyn = self._o_d[3 :: self._cfg.ndynobs * (self.N_hor + 1)]
        ang_dyn = self._o_d[4 :: self._cfg.ndynobs * (self.N_hor + 1)]
        if k >= self._critical_step:
            return ca.SX.zeros(x_dyn.shape[0], x_dyn.shape[1])
        inside_dyn = mh.inside_ellipses(state, [x_dyn, y_dyn, rx_dyn, ry_dyn, ang_dyn])# inside_dyn = mh.inside_ellipses(state, [x_dyn, y_dyn, rx_dyn, ry_dyn, ang_dyn])
        return inside_dyn 

    def build(
        self,
        solver_type: str = "ipopt",
        solver_options: dict | None = None,
    ) -> CasadiProblem:
        """Build NLP solver and return symbolic problem artifacts.

        TODO for you:
        1. Add/adjust hard obstacle constraints in `g` if needed.
        2. Validate obstacle indexing against your runtime packed parameter vector.
        3. Compare objective term-by-term against paper Eqs + PANOC behavior.
        """
        if self._motion_model is None:
            raise RuntimeError("Call `load_motion_model(...)` before `build()`.")

        # Decision variable
        X = ca.SX.sym("X", self.ns * (self.N_hor + 1))
        U = ca.SX.sym("U", self.nu * self.N_hor)
        w = ca.vertcat(X, U)
        w = cast(ca.SX, w) 

        # Slack variable
        # epsilon_static = ca.SX.sym("epsilon_1", self.N_hor * self._cfg.Nstcobs)
        # epsilon_dynamic = ca.SX.sym("epsilon_2", self.N_hor * self._cfg.Ndynobs)
        # w = ca.vertcat(X, U, epsilon_static, epsilon_dynamic)
        # w = cast(ca.SX, w)

        
        # lbw = [-ca.inf] * (self.ns * (self.N_hor + 1))
        # ubw = [ca.inf] * (self.ns * (self.N_hor + 1))

        lbw = [-ca.inf] * w.size1()
        ubw = [ca.inf] * w.size1()

        # Add bound for new input constraint
        # X_offset = 0
        # U_offset = self.ns * (self.N_hor + 1) # size of x-blcok
        # e_static_offset = U_offset + self.nu * self.N_hor
        # e_dynamic_offset = e_static_offset + self.N_hor * self._cfg.Nstcobs

        # for k in range(self.N_hor * self._cfg.Nstcobs):
        #     lbw[e_static_offset + k] = 0.0
        #     ubw[e_static_offset + k] = ca.inf

        # for k in range(self.N_hor * self._cfg.Ndynobs):
        #     lbw[e_dynamic_offset + k] = 0.0
        #     ubw[e_dynamic_offset + k] = ca.inf



        g: list[ca.SX] = []
        lbg: list[float] = []
        ubg: list[float] = []

        # Initial state equality: X_0 = s_0 (provided in parameters)
        x0_dec = X[: self.ns]
        g.append(x0_dec - self._s_0)
        lbg.extend([0.0] * self.ns)
        ubg.extend([0.0] * self.ns)

        ref_states = ca.reshape(self._r_s, (self.ns, self.N_hor))
        ref_states = ca.horzcat(ref_states, ref_states[:, [-1]])

        total_cost = ca.SX(0.0)
        prev_v = self._u_m1[0]
        prev_w = self._u_m1[1]

        for k in range(self.N_hor):
            x_k = X[k * self.ns : (k + 1) * self.ns]
            x_kp1 = X[(k + 1) * self.ns : (k + 2) * self.ns]
            u_k = U[k * self.nu : (k + 1) * self.nu]
            #e_static_k = epsilon_static[k * self._cfg.Nstcobs : (k+1) * self._cfg.Nstcobs]
            #e_dynamic_k = epsilon_dynamic[k * self._cfg.Ndynobs : (k+1) * self._cfg.Ndynobs]

            x_hat = self._motion_model(x_k, u_k, self.ts)
            g.append(x_kp1 - x_hat)
            lbg.extend([0.0] * self.ns)
            ubg.extend([0.0] * self.ns)

            # Add slack to g static
            # g.append(self._static_obstacle_intrusion(x_kp1)-e_static_k)
            # lbg.extend([-ca.inf] * self._cfg.Nstcobs)
            # ubg.extend([0.0] * self._cfg.Nstcobs)

            # Add slack to g dynamic
            # g.append(self._dynamic_obstacle_intrusion(k, x_kp1) - e_dynamic_k)
            # lbg.extend([-ca.inf] * self._cfg.Ndynobs)
            # ubg.extend([0.0] * self._cfg.Ndynobs)


            # Generate objective function J = Jr + Jo + sum(Jf)
            total_cost += self._stage_cost(k, x_kp1, u_k, ref_states[:, k:])

            # Acceleration bounds (same role as PANOC ALM constraints).
            # ||u_k - u_k-1 || Qa 
            acc = (u_k[0] - prev_v) / self.ts
            w_acc = (u_k[1] - prev_w) / self.ts
            g.extend([acc, w_acc])
            lbg.extend([self._spec.lin_acc_min, -self._spec.ang_acc_max])
            ubg.extend([self._spec.lin_acc_max, self._spec.ang_acc_max])

            # Add acceleration penalty to the cost/objective function
            total_cost += self._q_terms["acc_penalty"] * acc**2 # ||u_k - u_k-1 || Qa
            total_cost += self._q_terms["w_acc_penalty"] * w_acc**2 # ||u_k - u_k-1 || Qa
            #total_cost +=  ca.sum1(e_static_k)#1e2 * ca.sum1(e_static_k) + 1e4 * ca.sum1(e_static_k**2)
            #total_cost +=  ca.sum1(e_dynamic_k)#1e2 * ca.sum1(e_dynamic_k) + 1e4 * ca.sum1(e_dynamic_k**2)
            # total_cost += rho_stc * ca.sum1(e_static_k**2)
            # total_cost += rho_dyn * ca.sum1(e_dynamic_k**2)
            # v_stc = ca.fmax(0,ca.vertcat(self._static_obstacle_intrusion(x_kp1)))
            # v_dyn = ca.fmax(0, self._dynamic_obstacle_intrusion(k, x_kp1))
            # v = ca.vertcat(v_stc, v_dyn)
            # total_cost += self._rho_pen * ca.dot(v,v)

            prev_v = u_k[0]
            prev_w = u_k[1]

        x_N = X[self.N_hor * self.ns : (self.N_hor + 1) * self.ns]
        theta_terminal_err = mh.angle_error(x_N[2], self._s_N[2])
        total_cost += self._q_terms["posN"] * ((x_N[0] - self._s_N[0]) ** 2 + (x_N[1] - self._s_N[1]) ** 2)
        theta_terminal_error = self._wrapped_angle_error(x_N[2], self._s_N[2])
        total_cost += self._q_terms["thetaN"] * theta_terminal_error**2

        for k in range(self.N_hor):
            uk_start = self.ns * (self.N_hor + 1) + k * self.nu
            lbw[uk_start] = self._spec.lin_vel_min
            ubw[uk_start] = self._spec.lin_vel_max
            lbw[uk_start + 1] = -self._spec.ang_vel_max
            ubw[uk_start + 1] = self._spec.ang_vel_max

        g_expr = ca.vertcat(*g)
        nlp = {"x": w, "p": self._p, "f": total_cost, "g": g_expr}

        if solver_options is None:
            solver_options = {"ipopt.print_level": 0, "print_time": 0, "ipopt.max_iter":500} 
        solver = ca.nlpsol("nmpc_solver", solver_type, nlp, solver_options)

        return CasadiProblem(
            solver=solver,
            w=w,
            p=self._p,
            g=g_expr,
            lbw=[float(v) for v in lbw],
            ubw=[float(v) for v in ubw],
            lbg=lbg,
            ubg=ubg,
        )
    
    @staticmethod
    def shift_warm_start(w_opt: list[float], ns: int, nu: int, N: int) -> list[float]:
        """Shift warm start for w = [X, U]."""
        x_size = ns * (N + 1)
        u_size = nu * N
        expected_size = x_size + u_size
        if len(w_opt) != expected_size:
            raise ValueError(
                f"Warm-start vector size mismatch: got {len(w_opt)}, expected {expected_size} "
                f"(ns={ns}, nu={nu}, N={N})."
            )

        x = w_opt[:x_size]
        u = w_opt[x_size:x_size + u_size]

        x_shift = x[ns:] + x[-ns:]
        u_shift = u[nu:] + u[-nu:]

        return x_shift + u_shift


    def pack_parameters(
        self,
        u_m1: Sequence[float],
        s_0: Sequence[float],
        s_N: Sequence[float],
        q: Sequence[float],
        r_s: Sequence[float],
        r_v: Sequence[float],
        c_0: Sequence[float],
        c: Sequence[float],
        o_s: Sequence[float],
        o_d: Sequence[float],
        q_stc: Sequence[float],
        q_dyn: Sequence[float],
        rho_pen: float,
    ) -> list[float]:
        """Pack parameter blocks in the exact order used by `self._p`."""
        p: list[float] = []
        extend = p.extend
        extend(u_m1)
        extend(s_0)
        extend(s_N)
        extend(q)
        extend(r_s)
        extend(r_v)
        extend(c_0)
        extend(c)
        extend(o_s)
        extend(o_d)
        extend(q_stc)
        extend(q_dyn)
        extend(([rho_pen]))
        return p
