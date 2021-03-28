#!/usr/bin/env python3
from casadi import *
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


class MVE:

    def __init__(self, project_name='solver for the maximum volume ellipsoid'):
        self.project_name = project_name

    def initSearchRegion(self, x_lb, x_ub):
        self.n_x = len(x_lb)
        self.hyperplanes_a = np.identity(self.n_x)
        self.hyperplanes_b = np.array(x_ub).reshape((-1, 1))
        self.hyperplanes_a = np.vstack((self.hyperplanes_a, -np.identity(self.n_x)))
        self.hyperplanes_b = np.vstack((self.hyperplanes_b, -np.array(x_lb).reshape((-1, 1))))

    def mveSolver(self, tolerance=0.0001):
        try:
            # create a symmetric matrix variable
            C = cp.Variable((self.n_x, self.n_x), symmetric=True)
            d = cp.Variable(self.n_x)
            # create the constraints
            constraints = [C >> tolerance]
            for i in range(self.hyperplanes_a.shape[0]):
                constraints += [(cp.norm2(C @ self.hyperplanes_a[i].reshape(-1, 1)) + self.hyperplanes_a[i] @ d) <=
                                self.hyperplanes_b[i]]

            prob = cp.Problem(cp.Minimize(-cp.log_det(C)), constraints)
            prob.solve(solver=cp.MOSEK, verbose=False)
            return d.value, C.value
        except:
            return None, None

    def addHyperplane(self, a, b):
        self.hyperplanes_a = np.vstack((self.hyperplanes_a, a))
        self.hyperplanes_b = np.vstack((self.hyperplanes_b, b))

    # this is for internal test
    def draw(self, C=None, d=None):
        theta = np.arange(-pi,pi+0.1,0.1)
        if C is not None and d is not None:
            circle = np.vstack((np.cos(theta), np.sin(theta)))
            ellipsis = np.matmul(C,circle)+d.reshape(-1,1)
            plt.plot(ellipsis[0], ellipsis[1])

        # obtain the line data
        lines = []
        for i in range(self.hyperplanes_a.shape[0]):
            n = self.hyperplanes_a[i]
            b = self.hyperplanes_b[i]
            if abs(n[1]) < 1e-10:
                y = theta
                x = np.array([b/n[0]]*theta.size)
            else:
                x = theta
                y = (b-n[0]*x)/n[1]
            lines.append( (x,y) )

        for line in lines:
            plt.plot(line[0],line[1])
        plt.scatter([1],[1.])
        plt.axis([-pi, pi, -pi, pi])
        plt.show()


class OCSys:

    def __init__(self, project_name="my optimal control system"):
        self.project_name = project_name

    def setStateVariable(self, state, state_lb=[], state_ub=[]):
        self.state = state
        self.n_state = self.state.numel()
        if len(state_lb) == self.n_state:
            self.state_lb = state_lb
        else:
            self.state_lb = self.n_state * [-1e20]

        if len(state_ub) == self.n_state:
            self.state_ub = state_ub
        else:
            self.state_ub = self.n_state * [1e20]

    def setControlVariable(self, control, control_lb=[], control_ub=[]):
        self.control = control
        self.n_control = self.control.numel()

        if len(control_lb) == self.n_control:
            self.control_lb = control_lb
        else:
            self.control_lb = self.n_control * [-1e20]

        if len(control_ub) == self.n_control:
            self.control_ub = control_ub
        else:
            self.control_ub = self.n_control * [1e20]

    def setDyn(self, ode):

        self.dyn = ode
        self.dyn_fn = Function('dynamics', [self.state, self.control], [self.dyn])

    def setPathCost(self, features, weights):
        self.weights = weights
        self.features = features
        self.path_cost = dot(self.features, self.weights)
        self.path_cost_fn = Function('cost', [self.state, self.control, self.weights], [self.path_cost])

    def setFinalCost(self,final_cost):
        self.final_cost = final_cost
        self.final_cost_fn = Function('cost', [self.state], [self.final_cost])

    def ocSolver(self, ini_state, horizon, weights, print_level=0):
        assert hasattr(self, 'state'), "Define the state variable first!"
        assert hasattr(self, 'control'), "Define the control variable first!"
        assert hasattr(self, 'dyn'), "Define the system dynamics first!"
        assert hasattr(self, 'features'), "Define the  cost function first!"

        if type(ini_state) == numpy.ndarray:
            ini_state = ini_state.flatten().tolist()

        # Start with an empty NLP
        w = []
        w0 = []
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []

        # "Lift" initial conditions
        Xk = MX.sym('X0', self.n_state)
        w.append(Xk)
        lbw += ini_state
        ubw += ini_state
        w0 += ini_state

        # Formulate the NLP
        for k in range(horizon):
            # New NLP variable for the control
            Uk = MX.sym('U_' + str(k), self.n_control)
            w.append(Uk)
            lbw += self.control_lb
            ubw += self.control_ub
            w0 += [0.5 * (x + y) for x, y in zip(self.control_lb, self.control_ub)]

            # Integrate till the end of the interval
            Xnext = self.dyn_fn(Xk, Uk)
            Ck = self.path_cost_fn(Xk, Uk, weights)
            J = J + Ck

            # New NLP variable for state at end of interval
            Xk = MX.sym('X_' + str(k + 1), self.n_state)
            w.append(Xk)
            lbw += self.state_lb
            ubw += self.state_ub
            w0 += [0.5 * (x + y) for x, y in zip(self.state_lb, self.state_ub)]

            # Add equality constraint
            g.append(Xnext - Xk)
            lbg += self.n_state * [0]
            ubg += self.n_state * [0]

        # add the final cost
        J = J + self.final_cost_fn(Xk)

        # Create an NLP solver and solve it
        opts = {'ipopt.print_level': print_level, 'ipopt.sb': 'yes', 'print_time': print_level}
        prob = {'f': J, 'x': vertcat(*w), 'g': vertcat(*g)}
        solver = nlpsol('solver', 'ipopt', prob, opts)
        # Solve the NLP
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        w_opt = sol['x'].full().flatten()

        # take the optimal control and state
        sol_traj = numpy.concatenate((w_opt, self.n_control * [0]))
        sol_traj = numpy.reshape(sol_traj, (-1, self.n_state + self.n_control))
        state_traj_opt = sol_traj[:, 0:self.n_state]
        control_traj_opt = numpy.delete(sol_traj[:, self.n_state:], -1, 0)
        time = numpy.array([k for k in range(horizon + 1)])

        # output
        opt_sol = {"state_traj_opt": state_traj_opt,
                   "control_traj_opt": control_traj_opt,
                   "time": time,
                   "horizon": horizon,
                   "cost": sol['f'].full()}

        return opt_sol

    def getRecoveryMat(self, opt_sol):
        # validity check
        assert hasattr(self, 'state'), "Define the state variable first!"
        assert hasattr(self, 'control'), "Define the control variable first!"
        assert hasattr(self, 'dyn'), "Define the system dynamics first!"
        assert hasattr(self, 'features'), "Define the  cost function first!"

        # compute the Jacobian of the dynamics
        dfx_fn = Function('dfx_fn', [self.state, self.control], [jacobian(self.dyn, self.state)])
        dfu_fn = Function('dfu_fn', [self.state, self.control], [jacobian(self.dyn, self.control)])

        # compute the jacobian of the features
        dphix_fn = Function('dphix_fn', [self.state, self.control], [jacobian(self.features, self.state)])
        dphiu_fn = Function('dphiu_fn', [self.state, self.control], [jacobian(self.features, self.control)])

        # compute the gradient of the final cost function
        dhx_fn = Function('dh_fn',[self.state],[jacobian(self.final_cost,self.state)])

        # parse the input
        state_traj_opt = opt_sol['state_traj_opt']
        control_traj_opt = opt_sol['control_traj_opt']
        horizon = opt_sol['horizon']

        # compute the gradient matrix
        curr_x = state_traj_opt[0, :]
        curr_u = control_traj_opt[0, :]
        next_x = state_traj_opt[1, :]
        next_u = control_traj_opt[1, :]
        H2 = mtimes(dfu_fn(curr_x, curr_u).T, dfx_fn(next_x, next_u).T)
        H1 = mtimes(dfu_fn(curr_x, curr_u).T, dphix_fn(next_x, next_u).T) + dphiu_fn(curr_x, curr_u).T
        for t in range(1, horizon - 1):
            curr_x = state_traj_opt[t, :]
            curr_u = control_traj_opt[t, :]
            next_x = state_traj_opt[t + 1, :]
            next_u = control_traj_opt[t + 1, :]
            H1 = vertcat(H1 + mtimes(H2, dphix_fn(next_x, next_u).T),
                           mtimes(dfu_fn(curr_x, curr_u).T, dphix_fn(next_x, next_u).T) + dphiu_fn(curr_x, curr_u).T)
            H2 = vertcat(mtimes(H2, dfx_fn(next_x, next_u).T),
                           mtimes(dfu_fn(curr_x, curr_u).T, dfx_fn(next_x, next_u).T))
        curr_u = control_traj_opt[-1, :]
        curr_x = state_traj_opt[-2, :]
        H1 = vertcat(H1, dphiu_fn(curr_x,curr_u).T)
        H2 = vertcat(H2, dfu_fn(curr_x,curr_u).T)
        final_x = state_traj_opt[-1,:]
        lam = dhx_fn(final_x).T
        H2_lam = mtimes(H2, lam)

        return H1.full(), H2_lam.full().flatten()

    def getHyperplane(self, opt_sol, correction, correction_time):

        H1, H2_lam = self.getRecoveryMat(opt_sol)
        if type(correction_time) is not list:
            if type(correction) is not np.array:
                correction = np.array(correction).reshape((-1, 1))
            else:
                correction = correction.reshape((-1, 1))

            H1_t = H1[correction_time * self.n_control:(correction_time + 1) * self.n_control, :]
            H2_lam_t = H2_lam[correction_time * self.n_control:(correction_time + 1) * self.n_control]

            return np.matmul(H1_t.T, correction).flatten(), np.dot(H2_lam_t.flatten(), correction.flatten())

        else:
            vec_a = np.zeros(H1.shape[1])
            b = 0
            for k, t in enumerate(correction_time):
                H1_t = H1[t * self.n_control:(t + 1) * self.n_control, :]
                H2_lam_t = H2_lam[t * self.n_control:(t + 1) * self.n_control]
                correction_t = correction[k]
                vec_a += np.matmul(H1_t.T, correction_t.reshape((-1,1))).flatten()
                b += np.dot(H2_lam_t.flatten(), correction_t.flatten())
            return vec_a, b





