import sys
from typing import Callable, Union, Optional, List, cast

import numpy as np
import casadi as cs # type: ignore


class MultipleShootingSolver:
    """Build a NLP problem with a horizon (i.e. MPC) in the form of multiple shooting.
    
    Important:
        ```
            min_{w} J(w),
            s.t.    g(w) =  0, 
                    h(w) <= 0.
        ```
        `w` - decision variables, `g` - equality constraints, `h` - inequality constraints

        The solver focuses on an agent with a motion model `x_next = f(x_now, u_now)`.

    Notes:
        A variable with a subscript `_i` means it is a symbol, not a value (e.g. `x_0`).
        A variable with a number `i` means it is a value, not a symbol (e.g. `x0`). 
    """
    def __init__(self, ns: int, nu: int, ts: float, horizon: int) -> None:
        """Initialize the solver.

        Args:
            ns: The number of states.
            nu: The number of inputs.
            ts: The sampling time.
            horizon: The prediction/control horizon.
        """
        self._ns = ns
        self._nu = nu
        self._ts = ts
        self._N = horizon

        self._create_placeholders()
        self._f_func:Optional[Callable] = None # motion model
        self._lbx = [[-cs.inf]*self._ns]*(self._N+1)
        self._ubx = [[ cs.inf]*self._ns]*(self._N+1)
        self._lbu = [[-cs.inf]*self._nu]*self._N
        self._ubu = [[ cs.inf]*self._nu]*self._N

        self._eq_func_list:list = []
        self._ineq_func_list:list = []

        self.problem: Optional[dict] = None
        self.solver: Optional[cs.Function] = None
        self._built = False
        self._with_params = False
        self._with_initial_state = False

    def _create_placeholders(self) -> None:
        """Create placeholders (variable and constraint lists) for the problem."""
        self._w_list: List[cs.SX] = [] # decision variables
        self._g_list: List[cs.SX] = [] # equality constraints
        self._h_list: List[cs.SX] = [] # inequality constraints

        self._lbw_list: List[float] = [] # lower bound of decision variables
        self._ubw_list: List[float] = [] # upper bound of decision variables
        self._lbg_list: List[float] = [] # lower bound of equality constraints
        self._ubg_list: List[float] = [] # upper bound of equality constraints
        self._lbh_list: List[float] = [] # lower bound of inequality constraints
        self._ubh_list: List[float] = [] # upper bound of inequality constraints

    def _set_bound(self, bound: list) -> List[List[float]]:
        """Set the lower and upper bounds of the states or controls.
        
        Notes:
            The bounds can be set in two ways:
            1: list[float], Each value is for each state, the whole list will repeat N_horizon times.
            2: list[list[float]], Each sub-list is for a time step in the N_horizon.
        """
        if not isinstance(bound[0], list):
            return [bound] * self._N
        else:
            return bound

    @property
    def ns(self) -> int:
        return self._ns
    
    @property
    def nu(self) -> int:
        return self._nu

    @property
    def ts(self) -> float:
        if not isinstance(self._ts, float):
            raise RuntimeError("The time step is not fixed.")
        return self._ts

    @property
    def N(self) -> int:
        return self._N

    def set_state_bound(self, lbx: list, ubx: list) -> None:
        """Set the lower and upper bounds of the states.
        
        Notes:
            The bounds can be set in two ways:
            1: list[float], Each value is for each state, the whole list will repeat N_horizon times.
            2: list[list[float]], Each sub-list is for a time step in the N_horizon.
        """
        self._lbx = self._set_bound(lbx)
        self._ubx = self._set_bound(ubx)
        
    def set_control_bound(self, lbu: list, ubu: list) -> None:
        """Set the lower and upper bounds of the controls.

        Notes:
            The bounds can be set in two ways:
            1: list[float], Each value is for each state, the whole list will repeat N_horizon times.
            2: list[list[float]], Each sub-list is for a time step in the N_horizon.
        """
        self._lbu = self._set_bound(lbu)
        self._ubu = self._set_bound(ubu)
            
    def set_motion_model(self, func: Union[cs.Function, Callable], c2d:bool=False, sub_sampling:int=0) -> None:
        """Set the motion model of the system, which should be a mapping shown below.
        
        Notes:
            ```
            (x_next, cost_now) = func(x_now, u_now)
            ```

        Args:
            func: the motion model of the system.
            c2d: whether the motion model is continuous or discrete. Default: False (discrete).
            sub_sampling: (if c2d is true) the number of sub-samples in each time interval.
        """
        if c2d:
            self._f_func = self.return_discrete_function(func, self._ns, self._nu, self._ts, sub_sampling=sub_sampling)
        else:
            self._f_func = func

    def set_parameters(self, params: cs.SX, with_initial_state:bool=False) -> None:
        """Set the parameters of the problem.

        Args:
            params: The parameters of the problem.
            with_initial_state: Whether the first ns parameters are the initial state. Default: False.

        Raises:
            RuntimeError: The initial state has already been set.
            
        Notes:
            If with_initial_state, the first ns parameters must be the initial state. Can have redundant parameters.
        """
        if with_initial_state and self._with_initial_state:
            raise RuntimeError("The initial state has already been set.")
        self._params = params
        self._n_params = params.shape[0]
        if with_initial_state:
            self._x0 = params[:self._ns]
        self._with_params = True

    def set_initial_state(self, x0: List[float]) -> None:
        """Set the initial state of the problem.

        Args:
            x0: The initial state of the problem.

        Raises:
            RuntimeError: The parameters including the initial state have already been set.
        """
        if self._with_params:
            raise RuntimeError("The parameters including the initial state have already been set.")
        self._x0 = cs.DM(x0)
        self._with_initial_state = True


    def add_equality_constraint(self, func: Union[cs.Function, Callable]) -> None:
        """Add an external equality constraint to the problem.

        Args:
            func: The equality constraint function, should equal to 0.

        Raises:
            RuntimeError: The problem has already been built.
        """
        if self._built:
            raise RuntimeError("The problem has already been built. Cannot add more constraints.")
        self._eq_func_list.append(func)

    def add_inequality_constraint(self, func: Union[cs.Function, Callable]) -> None:
        """Add an external inequality constraint to the problem.

        Args:
            func: The inequality constraint function, should be less than or equal to 0.

        Raises:
            RuntimeError: The problem has already been built.
        """
        if self._built:
            raise RuntimeError("The problem has already been built. Cannot add more constraints.")
        self._ineq_func_list.append(func)


    def build_problem(self) -> dict:
        """Build the NLP problem in the form of multiple shooting.

        Raises:
            RuntimeError: The motion model is not set.
            RuntimeError: Parameters or initial state must be set.
        
        Returns:
            problem: a dictionary (x, f, g, lbx, ubx, lbg, ubg) containing the NLP problem.

        Notes:
            Method `build_solver` automatically calls this function.
        """
        if self._f_func is None:
            raise RuntimeError("Motion model is not set.")
        if not self._with_params:
            if not self._with_initial_state:
                raise RuntimeError("Parameters or initial state must be set.")
            self._params = cs.SX.sym('p', 0)
            self._n_params = 0

        self._w_list.append(cs.SX.sym('x_0', self._ns))
        self._lbw_list += self._lbx[0]
        self._ubw_list += self._ubx[0]
        self._g_list.append(self._w_list[0] - self._x0)
        self._lbg_list += [0]*self._ns
        self._ubg_list += [0]*self._ns

        x_k = self._w_list[0]
        J = 0 # objective
        for k in range(self._N):
            x_next = cs.SX.sym('x_' + str(k+1), self._ns)
            u_k = cs.SX.sym('u_' + str(k), self._nu)
            x_next_hat, J_k = self._f_func(x_k, u_k)

            self._w_list += [u_k, x_next]
            self._lbw_list += self._lbu[k] + self._lbx[k+1]
            self._ubw_list += self._ubu[k] + self._ubx[k+1]

            self._g_list += [x_next - x_next_hat]
            self._lbg_list += [0]*self._ns
            self._ubg_list += [0]*self._ns

            for eq_func in self._eq_func_list:
                self._g_list += [eq_func(x_k, u_k)]
                self._lbg_list += [0]
                self._ubg_list += [0]

            for ineq_func in self._ineq_func_list:
                self._h_list += [ineq_func(x_k, u_k)]
                self._lbh_list += [-cs.inf]
                self._ubh_list += [0]

            J += J_k
            x_k = x_next

        constraint_list = self._g_list + self._h_list
        constraint_lb_list = self._lbg_list + self._lbh_list
        constraint_ub_list = self._ubg_list + self._ubh_list
        
        self.problem = {'f': J, 'g': cs.vertcat(*constraint_list),
                        'x': cs.vertcat(*self._w_list), 'p': self._params,
                        'lbx': self._lbw_list, 'ubx': self._ubw_list,
                        'lbg': constraint_lb_list, 'ubg': constraint_ub_list}
        self._built = True
        return self.problem
    
    def build_solver(self, solver_type:str='ipopt', 
                     solver_options:Optional[dict]={'ipopt.print_level':0, 'print_time':5},
                     build_kwargs:Optional[dict]=None):
        """_summary_

        Args:
            solver_type: The solving algorithm. Defaults to 'ipopt'.
            solver_options: Options for the solver. Defaults to {'ipopt.print_level':0, 'print_time':5}.
            build_kwargs: Keyword arguments for build the solver. Default: None.

        Returns:
            solver: The solver function.

        Notes:
            Solver option will supress printout from the solver if not None.
        """
        if not self._built:
            self.build_problem()

        self.problem = cast(dict, self.problem)
        problem = {k: self.problem[k] for k in ('f', 'x', 'g', 'p')}

        solver_options = {} if solver_options is None else solver_options
        build_kwargs = {} if build_kwargs is None else build_kwargs

        self.solver = cs.nlpsol('solver', solver_type, problem, solver_options, **build_kwargs)
        self.solver = cast(cs.Function, self.solver)
        return self.solver

    def run(self, 
            parameters:Optional[List[float]]=None, 
            initial_guess:Optional[List[float]]=None, 
            run_kwargs:Optional[dict]=None,
            self_check:bool=False):
        """Solve the NLP problem.
        
        Args:
            parameters: The parameters of the NLP problem. Default: None.
            initial_guess: The initial guess of the decision variables. Default: None.
            run_kwargs: Keyword arguments for run the solver. Default: None.

        Raises:
            RuntimeError: The problem is not built.
            RuntimeError: The solver is not built.
        
        Returns:
            sol: the solution of the NLP problem.

        Notes:
            Parameters can have less elements than the number of parameters in the problem,
            and the rest of the parameters will be set to 0.
        """
        if self.problem is None:
            raise RuntimeError("The problem is not built.")
        if self.solver is None:
            raise RuntimeError("The solver is not built.")

        if parameters is None:
            parameters = [0.0]*self._n_params
        elif len(parameters) < self._n_params:
            parameters += [0.0]*(self._n_params-len(parameters))
        if initial_guess is None:
            initial_guess = [0.0] * ((self._ns+self._nu)*self._N+self._ns)
        run_kwargs = {} if run_kwargs is None else run_kwargs

        if self_check:
            self.self_check(initial_guess, parameters)
        sol: dict = self.solver(x0=initial_guess, p=parameters, lbx=self.problem['lbx'], ubx=self.problem['ubx'],
                                lbg=self.problem['lbg'], ubg=self.problem['ubg'],**run_kwargs)
        sol_stats: dict = self.solver.stats()
        
        solving_time = 1000*sol_stats['t_wall_total']
        exit_status = sol_stats['return_status']
        sol_cost = float(sol['f'])
        return sol, solving_time, exit_status, sol_cost
    
    
    def get_initial_guess(self, sol: dict) -> List[float]: # TODO: validate this function
        """Get the next initial guess from the solution.
        
        Args:
            sol: The solution of the NLP problem.
        
        Returns:
            initial_guess: The next initial guess of the decision variables.
        """
        sol_x:cs.DM = sol['x']
        initial_guess = sol_x.full().flatten().tolist()
        return initial_guess
    
    def get_pred_states(self, sol: dict) -> List[List[float]]:
        """Get the predicted states from the solution.
        
        Args:
            sol: the solution of the NLP problem.
        
        Returns:
            x_pred: a list of predicted states. Each row is a state.
        """
        sol_x = sol['x']
        nu_factor = 1

        x_pred = []
        for i in range(self._ns):
            xi_pred:cs.DM = sol_x[i::(self._ns+self._nu*nu_factor)]
            xi_pred_np:np.ndarray = xi_pred.full()
            x_pred.append(xi_pred_np.flatten().tolist())
        return x_pred
    
    def get_opt_controls(self, sol: dict) -> List[List[float]]:
        """Get the optimal controls from the solution.
        
        Args:
            sol: the solution of the NLP problem.
        
        Returns:
            u_opt: a list of optimal controls. Each row is a control.
        """
        sol_x = sol['x']
        nu_factor = 1

        u_opt = []
        for i in range(self._nu*nu_factor):
            ui_opt:cs.DM = sol_x[self._ns+i::(self._ns+self._nu*nu_factor)]
            ui_opt_np:np.ndarray = ui_opt.full()
            u_opt.append(ui_opt_np.flatten().tolist())
        return u_opt

    def self_check(self, initial_guess:Optional[List[float]]=None, parameters:Optional[List[float]]=None):
        if (initial_guess is not None) and (self.problem is not None):
            if len(initial_guess) != self.problem['x'].shape[0]:
                raise RuntimeError(f"The length of initial guess ({len(initial_guess)}) does not match the number of decision variables ({self.problem['x'].shape[0]}).")
        if (parameters is not None) and (self.problem is not None):
            if len(parameters) != self.problem['p'].shape[0]:
                raise RuntimeError(f"The length of parameters ({len(parameters)}) does not match the number of parameters ({self.problem['p'].shape[0]}).")
        
    @staticmethod
    def return_discrete_function(func_c: cs.Function, ns: int, nu: int, ts: float, 
                                 method:str='rk4', sub_sampling:int=0) -> cs.Function:
        """Return a discrete function from a continuous function.
        
        Args:
            method: the method to use for discretization. Default/only: `rk4`.
            sub_sampling: the number of sub-sampling steps in each time interval. Default: 0.
        """
        x = cs.SX.sym('x', ns)
        u = cs.SX.sym('u', nu)
        M = sub_sampling + 1
        dt = ts/M # if sub_sampling == 0, dt = ts

        x_next = x
        J_next = 0
        for _ in range(M):
            k1, k1_q = func_c(x_next, u)
            k2, k2_q = func_c(x_next + dt/2*k1, u)
            k3, k3_q = func_c(x_next + dt/2*k2, u)
            k4, k4_q = func_c(x_next + dt*k3, u)
            x_next += dt/6*(k1 + 2*k2 + 2*k3 + k4)
            J_next += dt/6*(k1_q + 2*k2_q + 2*k3_q + k4_q)
        f = cs.Function('f', [x, u], [x_next, J_next])
        return f
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt # type: ignore

    ### Parameters
    T = 10 # total time
    N = 50 # number of control steps
    ts = T/N
    x0 = [0.0, 1.0]

    ### Continuous time
    def return_continuous_function(ns: int, nu: int) -> cs.Function:
        xc = cs.SX.sym('xc', ns)
        uc = cs.SX.sym('uc', nu)
        xc_dot = cs.vertcat((1-xc[1]**2)*xc[0] - xc[1] + uc, xc[0])
        Jc_obj = xc[0]**2 + xc[1]**2 + uc**2
        fc = cs.Function('fc', [xc, uc], [xc_dot, Jc_obj])
        return fc

    fc = return_continuous_function(2, 1)

    ms_solver = MultipleShootingSolver(2, 1, ts, N)
    ms_solver.set_motion_model(fc, c2d=True, sub_sampling=3)
    ms_solver.set_control_bound([-1.0], [1.0])
    ### Option 1: set the initial state as a parameter
    ms_solver.set_parameters(cs.SX.sym('p', 5), with_initial_state=True)
    ms_solver.build_solver()
    sol, *_ = ms_solver.run(parameters=x0)
    ### Option 2: set the initial state as fixed
    # ms_solver.set_initial_state(x0)
    # ms_solver.build_solver()
    # sol, *_ = ms_solver.run()

    # Plot the solution
    u_opt = ms_solver.get_opt_controls(sol)
    x_pred = ms_solver.get_pred_states(sol)

    tgrid = [T/N*k for k in range(N+1)]
    plt.figure(1)
    plt.clf()
    plt.plot(tgrid, x_pred[0], '--')
    plt.plot(tgrid, x_pred[1], '-')
    plt.step(tgrid, cs.vertcat(cs.DM.nan(1), cs.DM(u_opt[0])), '-.')
    plt.xlabel('t')
    plt.legend(['x1','x2','u'])
    plt.grid()
    plt.show()


