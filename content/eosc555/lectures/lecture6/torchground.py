import torch
from torch import nn
import matplotlib.pyplot as plt

def RungeKutta4(dxdt, n_time_steps, period):
    class RK4:
        def __init__(self, dxdt, n_time_steps, period):
            self.dxdt = dxdt
            self.n_time_steps = n_time_steps
            self.period = period
            self.dt = period / n_time_steps

        def integrate(self, x0):
            x = x0.clone()
            trajectory = [x0.clone()]
            for _ in range(self.n_time_steps):
                k1 = self.dxdt(x, 0)
                k2 = self.dxdt(x + 0.5 * self.dt * k1, 0)
                k3 = self.dxdt(x + 0.5 * self.dt * k2, 0)
                k4 = self.dxdt(x + self.dt * k3, 0)
                x = x + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                trajectory.append(x.clone())
            return torch.stack(trajectory, dim=1)  # Shape: (2, n_time_steps + 1)
    return RK4(dxdt, n_time_steps, period)

class LotkaVolterra(nn.Module):
    """
    Lotka-Volterra (Predator-Prey) Model with Trainable Parameters.
    """
    def __init__(self, period, n_time_steps, perturbation=None, time_variant=False):
        """
            Initializes the Lotka-Volterra model.

            Args:
                period (float): Length of the time horizon.
                n_time_steps (int): Number of time steps for integration.
                perturbation (torch.Tensor, optional): Tensor to perturb alpha parameters. Defaults to None.
                time_variant (bool, optional): If True, parameters are time-variant. Defaults to False.
        """
        super(LotkaVolterra, self).__init__()
        self.time_steps = n_time_steps
        self.period = period
        self.time_variant = time_variant

        if perturbation is None:
            perturbation = torch.zeros(n_time_steps + 1, dtype=torch.float32)

        # Initialize trainable parameters, which can vary over time steps
        if time_variant:
            self.alpha = nn.Parameter((2/3) * torch.ones(n_time_steps + 1,) + perturbation)
            self.beta = nn.Parameter((4/3) * torch.ones(n_time_steps + 1))
            self.gamma = nn.Parameter(1.0 * torch.ones(n_time_steps + 1))
            self.delta = nn.Parameter(1.0 * torch.ones(n_time_steps + 1))
        else:
            self.alpha = nn.Parameter((2/3) * torch.ones(1,) + perturbation)
            self.beta = nn.Parameter((4/3) * torch.ones(1))
            self.gamma = nn.Parameter(1.0 * torch.ones(1))
            self.delta = nn.Parameter(1.0 * torch.ones(1))

    def dxdt(self, x, i):
        """
            Computes the derivatives for the Lotka-Volterra equations.
        """

        if not self.time_variant:
            # In case where parameters are not time-variant,
            # we set i to 0 to use the first and only parameter value
            i = 0
        
        dx1dt = self.alpha[i]*x[0] - self.beta[i]*x[0]*x[1]
        dx2dt = -self.gamma[i]*x[1] + self.delta[i]*x[0]*x[1]
        dxdt = torch.zeros(2)
        dxdt[0] = dx1dt
        dxdt[1] = dx2dt
        return dxdt

    def forward(self, x0, pk=None):
        """
            Solves the Lotka-Volterra equations using RK4.

            Args:
                x0 (torch.Tensor): Initial state tensor [prey, predator].
                pk (torch.Tensor, optional): Override existing parameters with pk. Defaults to None.

            Returns:
                torch.Tensor: Solution tensor over time of shape (2, time_steps + 1).
        """
        rk4 = RungeKutta4(self.dxdt, self.time_steps, self.period)
        return rk4.integrate(x0)

period = 40.0  # Time horizon as a single float
n_time_steps = 2000

# Initialize the Lotka-Volterra model
model = LotkaVolterra(period=period, n_time_steps=n_time_steps, time_variant=True)

# Initial populations: [prey, predator]
initial_state = torch.rand(2)

# Perform integration
solution = model(initial_state)


from torch.autograd.functional import jvp, vjp
from torch.nn.functional import pad

def generate_data_set(forward_model, period=40.0, n_time_steps=2000, n_realizations=10):
    """
    Generates a training dataset for the Lotka-Volterra model by simulating multiple realizations
    with perturbed parameters.

    Args:
        forward_model (callable): The Lotka-Volterra model to simulate.
        period (float, optional): Length of the time horizon. Defaults to 40.0.
        n_time_steps (int, optional): Number of time steps for integration. Defaults to 2000.
        n_realizations (int, optional): Number of realizations to simulate. Defaults to 10.
    
    Returns:
        tuple: A tuple containing two lists:
            - XX (list of torch.Tensor): Simulated population trajectories for each realization.
            - M (list of torch.Tensor): Perturbations for each realization.
    """
    
    pop_data_runs = []
    perturbations = []
    
    # Iterate over the number of desired realizations
    for run_idx in range(n_realizations):
        print(f'Computing realization {run_idx + 1}/{n_realizations}')
        
        # Noise pertubation to make more interesting dynamics
        noise = torch.randn(1, 1, n_time_steps + 1)        
        # Apply some smoothing to the pertubation using a low-pass filter to make natural variations
        for i in range(250):
            noise = pad(noise, pad=(1, 1), mode='reflect')
            noise = (noise[:, :, :-2] + 
                            2 * noise[:, :, 1:-1] + 
                            noise[:, :, 2:]) / 4
        noise = noise.squeeze()
        
        # Create a time variant model with the perturbation
        model = forward_model(period, n_time_steps, noise, time_variant=True)
        
        # Generate random initial conditions
        initial_pop = torch.rand(2)
        
        # Run the forward dynamics to generate the data
        pop_data = model(initial_pop)
        
        # Append the results to the respective lists
        pop_data_runs.append(pop_data)
        perturbations.append(noise)
    
    return pop_data_runs, perturbations

XX, M = generate_data_set(LotkaVolterra, period=40, n_time_steps=2000, n_realizations=1)

# Define a functional Lotka-Volterra model that can have parameters passed in to it with a fixed
# x0, etc.
def generate_lotka(x0, period=40.0, n_time_steps=2000):
    """
    Generates a Lotka-Volterra model with parameters p0 and initial conditions x0.

    Args:
        x0 (torch.Tensor): Initial state tensor [prey, predator].
        p0 (torch.Tensor): Parameter tensor [alpha, beta, gamma, delta].
        period (float, optional): Length of the time horizon. Defaults to 40.0.
        n_time_steps (int, optional): Number of time steps for integration. Defaults to 2000.
    
    Returns:
        tuple: A tuple containing the model and the solution.
    """
    
    def lotka(pk):
        
        def dxdt(x, i):
            dx1dt = pk[0]*x[0] - pk[1]*x[0]*x[1]
            dx2dt = -pk[2]*x[1] + pk[3]*x[0]*x[1]
            dxdt = torch.zeros(2)
            dxdt[0] = dx1dt
            dxdt[1] = dx2dt
            return dxdt
        
        output = RungeKutta4(dxdt, n_time_steps, period).integrate(x0)
        return output
    return lotka
        
# Define the J_G^T J_G operator, this functions as the A in Ax = b
def Hv(model, pk, v):
    """
    Computes the value of Jacobian(pk)^T * Jacobian(pk) * v.
    """
    # Compute the Jacobian-vector product
    Jv = jvp(model, (pk,), (v,))[1]
    # Compute the Jacobian-transpose-vector product
    JvT = vjp(model, (pk,), (Jv,))[1]
    
    return JvT


def conjugate_gradient(A, b, x0, tol=1e-5, max_iter=1000):
    """
    Conjugate Gradient method for solving the linear system Ax = b.
    """
    x = x0
    r = b - A(x)
    p = r
    rsold = torch.dot(r.flatten(), r.flatten())
    
    for i in range(max_iter):
        Ap = A(p)
        alpha = rsold / torch.dot(p.flatten(), Ap.flatten())
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = torch.dot(r.flatten(), r.flatten())
        
        if torch.sqrt(rsnew) < tol:
            break
        
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    
    return x

# Select the first realization
data = XX[0]  # Shape: (2, 2001)

# Initialize the functional model
x0 = data[:, 0]  # Initial conditions
lotka = generate_lotka(x0, period=40, n_time_steps=2000)

# Initialize parameters
p0 = torch.tensor([2/3, 4/3, 1.0, 1.0], requires_grad=True)

# Compute the model's current output
x_curr = lotka(p0)  # Shape: (2, 2001)

# Compute the residual
rk = x_curr - data  # Shape: (2, 2001)

b = vjp(lotka, (p0,), (rk,))[1].flatten()

print(b)




