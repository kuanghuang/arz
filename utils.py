import numpy as np

# relaxation time
tau = 1.

# pressure function, its derivatives and inverse
beta = 0.1

def pressure(rho):
    return beta * np.sqrt(rho / (1-rho))

def pressure_1st_derivative(rho):
    return beta * 0.5 * (rho**(-0.5) * (1-rho)**(-0.5) + rho**0.5 * (1-rho)**(-1.5))

def pressure_2nd_derivative(rho):
    return beta * (-0.25 * rho**(-1.5) + rho**(-0.5)) * (1-rho)**(-2.5)

def pressure_inverse(u):
    if any(x < 0 for x in u):
        raise ValueError("There exist vacuum problems.")
    else:
        return u*u / (u*u + beta*beta)

# desired speed function
def desired_speed(rho):
    return 1-rho

# initial data
def get_initial_data(num_cells, ini_type="rp"):
    dx = 1.0 / num_cells
    x = np.linspace(dx/2, 1-dx/2, num_cells)

    if ini_type == "rp":
        # Riemann problem
        rho_ini = 0.25 * (x<0.5) + 0.5 * (x>0.5)
        u_ini = 0.5 * (x<0.5) + 0.25 * (x>0.5)
    elif ini_type == "contact":
        # moving square wave
        rho_ini = 0.25 * (x<0.5) + 0.75 * (x>0.5)
        u_ini = 0.5 * np.ones(num_cells)
    elif ini_type == "bell":
        # bell shape
        rho_ini = 0.25 + 0.5 * np.exp(-100 * (x-0.5)**2)
        u_ini = 0.5 * np.ones(num_cells)
    else:
        raise ValueError("Invalid initial data type.")

    return rho_ini, u_ini

# relaxation term integration with implicit Euler
def step_relaxation(solver, state, dt):
    c = dt / tau
    Q_ds = lambda rho: rho * desired_speed(rho)
    state.q[1] = (state.q[1] + c * (Q_ds(state.q[0]) + state.q[0] * pressure(state.q[0]))) / (1 + c)