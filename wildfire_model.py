from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from scipy.optimize import fsolve

def residuals(TS1, TS0, K1, K2, dx, dt, A, B, C1, C2, nu, cT_a, dT_a, hT_a, cT_b, dT_b, hT_b, cS_a, dS_a, hS_a, cS_b, dS_b, hS_b):
    """The nonlinear implicit residuals for the Crank-Nicholson finite difference
    approximation of the wildfire model.

    T_t = T_xx - nu * T_x + A * (S * e ^ (-B / T) - C1 * T)
    S_t = -C2 * S * e ^ (-B / T)

    We use the Crank-Nicholson finite difference approximations

    u_xx = (1 / (4 * dx ** 2)) * (u_(t+1)_(j+1) - 2*u_(t+1)_(j) + u_(t+1)_(j-1)
                                + u_(t)_(j+1) - 2 * u_(t)_(j) + u_(t)_(j-1))
    u_x = (1 / (4 * dx)) * (u_(t+1)_(j+1) - u_(t+1)_(j-1)
                          + u_(t)_(j+1) - u_(t)_(j-1))
    u_t = (1 / (2 * dt)) * (u_(t+1)_(j) - u_(t)_(j))


    Parameters
        TS1 (ndarary): the concatenation of T and S at the next timestep
        TS0 (ndarray): the concatenation of T and S at the current timestep
        K1 (float): dt / (2*(dx)**2)
        K2 (float): -nu * dt / (4*dx), note the negative

    Returns
        residuals (ndarray): zero if the diffeq is satisfied.
    """
    # split the input into two parts, temperature and fuel
    T1, S1 = np.split(TS1, 2)
    T0, S0 = np.split(TS0, 2)

    # helper expression
    SeBT = S1[1:-1] * np.exp(-B/T1[1:-1])

    # set up the ODE for temperature
    T_lhs = T1[1:-1] - T0[1:-1]

    # approximating T_xx
    T_rhs_1 = K1 * ((T1[2:] - 2*T1[1:-1] + T1[:-2]) + (T0[2:] - 2*T0[1:-1] + T0[:-2]))
    # approximating -v*T_x
    T_rhs_2 = K2 * ((T1[2:] - T1[:-2]) + (T0[2:] - T0[:-2]))
    # exponential term
    T_rhs_3 = dt * A * (SeBT - C1 * T1[1:-1])
    # sum them to get the right hand side
    T_rhs = T_rhs_1 + T_rhs_2 + T_rhs_3

    # set up the ODE for fuel
    S_lhs = S1[1:-1] - S0[1:-1]
    S_rhs = dt * (-C2 * SeBT)

    # boundary conditions for left side
    Ta_condition = (dx * cT_a - dT_a) * T1[0] + dT_a * T1[1]
    Sa_condition = (dx * cS_a - dS_a) * S1[0] + dS_a * S1[1]

    # boundary conditions for right side
    Tb_condition = (dx * cT_b + dT_b) * T1[-1] - dT_b * T1[-2]
    Sb_condition = (dx * cS_b + dS_b) * S1[-1] - dS_b * S1[-2]

    # concatenate the boundary condition residuals and interior residuals
    res = np.concatenate((
        [dx * hT_a - Ta_condition],
        [dx * hS_a - Sa_condition],
        T_lhs - T_rhs,
        S_lhs - S_rhs,
        [dx * hT_b - Tb_condition],
        [dx * hS_b - Sb_condition]
    ))
    return res

    
def wildfire_model(a, b, end_time, N_x, N_t, T_0, S_0, cT_a, dT_a, hT_a, cT_b, dT_b, hT_b, cS_a, dS_a, hS_a, cS_b, dS_b, hS_b, A, B, C1, C2, nu):
    """Solves the wildfire model differential equation

    T_t = T_xx - nu * T_x + A * (S * e ^ (-B / T) - C1 * T)
    S_t = -C2 * S * e ^ (-B / T)

    with boundary conditions

    h(t) = c(t) * u(x, t) + d(t) * u_x(x, t)

    for x at endpoints `a` and `b`. 

    Parameters
        a (float): the spatial left endpoint
        b (float): the spatial right endpoint, a < b
        end_time (float): greater than 0
        N_x (int): number of mesh nodes in x, N_x > 2
        N_t (int): number of mesh nodes in t, N_t > 1
        T_0 (function): initial condition for T(x, 0).
        S_0 (function): initial condition for S(x, 0).
        cT_a (function): boundary condition
        dT_a (function): boundary condition
        hT_a (function): boundary condition
        cT_b (function): boundary condition
        dT_b (function): boundary condition
        hT_b (function): boundary condition
        cS_a (function): boundary condition
        dS_a (function): boundary condition
        hS_a (function): boundary condition
        cS_b (function): boundary condition
        dS_b (function): boundary condition
        hS_b (function): boundary condition
        A (float): equation parameter, A > 0
        B (float): equation parameter, B > 0, smaller consumes more fuel
        C1 (float): equation parameter, C1 > 0, greater means temperature decreases faster
        C2 (float): equation parameter, C2 > 0, greater consumes more fuel
        nu (float): wind speed, positive means moving to the right
    
    Returns
        T (ndarray): temperature. T[0,:] is the initial condition; 
                                  T[-1,:] is the final condition 
        S (ndarray): fuel, same shape as T
    """

    # create mesh nodes
    x, dx = np.linspace(a, b, N_x, retstep=True)
    t, dt = np.linspace(0, end_time, N_t, retstep=True)
    
    # helper variables
    K1 = dt/(2*(dx**2))
    K2 = -nu*dt/(4*dx)

    # initial condition
    T0 = T_0(x)
    S0 = S_0(x)
    TS0 = np.concatenate((T0, S0))
    TSs = [TS0]
    Ts = [T0]
    Ss = [S0]
    
    
    # iterate over timesteps
    for t_i in t[1:]:
        args = (TSs[-1], K1, K2, dx, dt, A, B, C1, C2, nu, cT_a(t_i), dT_a(t_i), hT_a(t_i), cT_b(t_i), dT_b(t_i), hT_b(t_i), cS_a(t_i), dS_a(t_i), hS_a(t_i), cS_b(t_i), dS_b(t_i), hS_b(t_i))
        
        TS_i, info_dict, _, _ = fsolve(func=residuals, x0=TSs[-1], args=args, full_output=True) # isolate solver and finite difference logic
        print("residuals: {}".format(np.max(np.abs(info_dict['fvec']))))

        T_i, S_i = np.split(TS_i, 2)
        Ts.append(T_i)
        Ss.append(S_i)
        TSs.append(TS_i)
    
    return np.array(Ts), np.array(Ss)

def test_wildfire():
    """Test case for the wildfire equation.
    """

    # Sample intial conditions
    a = -10
    b = 10
    end_time = 1.0
    N_x = 100
    N_t = 10
    T_0 = lambda x: 1/np.cosh(x)
    S_0 = lambda x: np.tanh(x)
    A = 1 #1.8793e2
    B = 0.1 #5.5849e2
    C1 = 1 #4.8372e-5
    C2 = 1 #1.625e-1
    nu = 1

    # Dirichlet boundary conditions
    hT_a = lambda t: T_0(a)
    cT_a = lambda t: 1
    dT_a = lambda t: 0

    hT_b = lambda t: T_0(b)
    cT_b = lambda t: 1 
    dT_b = lambda t: 0

    hS_a = lambda t: S_0(a)
    cS_a = lambda t: 1
    dS_a = lambda t: 0

    hS_b = lambda t: S_0(b)
    cS_b = lambda t: 1
    dS_b = lambda t: 0

    # solve the equation
    x = np.linspace(a, b, N_x)
    Ts, Ss = wildfire_model(a, b, end_time, N_x, N_t, T_0, S_0, cT_a, dT_a, hT_a, cT_b, dT_b, hT_b, cS_a, dS_a, hS_a, cS_b, dS_b, hS_b, A, B, C1, C2, nu)
    print("solved finite difference method")

    # plot the initial and final states
    fig, axs = plt.subplots(1, 2, figsize=(8,5))
    axs[0].plot(x, Ts[0], color='red', label=r"$T(x,0)$")
    axs[0].plot(x, Ss[0], color='green', label=r"$S(x,0)$")
    funcT, = axs[1].plot(x, Ts[-1], color='red', label=r"$T(x,{})$".format(end_time))
    funcS, = axs[1].plot(x, Ss[-1], color='green', label=r"$S(x,{})$".format(end_time))

    # make it pretty
    for ax in axs:
        ax.set_xlim([a, b])
        ax.set_ylim([0, 1.5])
        ax.set_title("Crank-Nicolson Method")
        ax.legend()
    plt.show()

    def update(i):
        funcT.set_data(x, Ts[i])
        funcS.set_data(x, Ss[i])
        return funcT, funcS

    # save the animation
    ani = animation.FuncAnimation(fig, update, frames=list(range(len(Ts))), interval=50)
    ani.save("wildfire.mp4")
    plt.close()
    print("saved animation")

    # print("delta S: \n{}".format(Ss[-1] - Ss[0]))
    # import pdb; pdb.set_trace()

if __name__ == "__main__":
    test_wildfire()