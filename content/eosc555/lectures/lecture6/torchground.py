import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

class RungeKutta4:
    """
    Runge-Kutta 4th Order Integrator for solving ODEs.
    """
    def __init__(self, func, time_steps, period):
        """
        Initializes the RK4 integrator.

        Args:
            func (callable): The function defining the ODE system, f(x, i).
            time_steps (int): Number of time steps to integrate over.
            period (list or tuple): [start_time, end_time].
        """
        self.func = func
        self.time_steps = time_steps
        self.start_time, self.end_time = period
        self.dt = (self.end_time - self.start_time) / self.time_steps

    def integrate(self, x0):
        """
        Performs the RK4 integration.

        Args:
            x0 (torch.Tensor): Initial state tensor of shape (n_vars,).

        Returns:
            torch.Tensor: Tensor containing the solution at each time step of shape (n_vars, time_steps + 1).
        """
        X = torch.zeros(x0.size(0), self.time_steps + 1)
        X[:, 0] = x0

        for i in range(self.time_steps):
            k1 = self.func(X[:, i], i)
            k2 = self.func(X[:, i] + self.dt * k1 / 2, i)
            k3 = self.func(X[:, i] + self.dt * k2 / 2, i)
            k4 = self.func(X[:, i] + self.dt * k3, i)
            X[:, i + 1] = X[:, i] + (self.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        return X

class LotkaVolterra(nn.Module):
    """
    Lotka-Volterra (Predator-Prey) Model with Trainable Parameters.
    """
    def __init__(self, period, n_time_steps, perturbation=None):
        """
        Initializes the Lotka-Volterra model.

        Args:
            period (list or tuple): [start_time, end_time].
            n_time_steps (int): Number of time steps for integration.
            perturbation (torch.Tensor, optional): Tensor to perturb alpha parameters. Defaults to None.
        """
        super(LotkaVolterra, self).__init__()
        self.time_steps = n_time_steps
        self.period = period

        if perturbation is None:
            perturbation = torch.zeros(n_time_steps + 1)

        # Initialize trainable parameters
        self.alpha = nn.Parameter( (2/3) * torch.ones(n_time_steps + 1) + perturbation )
        self.beta = nn.Parameter( (4/3) * torch.ones(n_time_steps + 1) )
        self.gamma = nn.Parameter( torch.ones(n_time_steps + 1) )
        self.delta = nn.Parameter( torch.ones(n_time_steps + 1) )

    def predator_prey_derivatives(self, state, i):
        """
        Computes the derivatives for the Lotka-Volterra equations.

        Args:
            state (torch.Tensor): Current state tensor [prey, predator].
            i (int): Current time step index.

        Returns:
            torch.Tensor: Derivatives [dprey/dt, dpredator/dt].
        """
        prey, predator = state
        dprey_dt = self.alpha[i] * prey - self.beta[i] * prey * predator
        dpredator_dt = -self.gamma[i] * predator + self.delta[i] * prey * predator
        derivatives = torch.stack([dprey_dt, dpredator_dt])
        return derivatives

    def forward(self, x0):
        """
        Solves the Lotka-Volterra equations using RK4.

        Args:
            x0 (torch.Tensor): Initial state tensor [prey, predator].

        Returns:
            torch.Tensor: Solution tensor over time of shape (2, time_steps + 1).
        """
        rk4 = RungeKutta4(self.predator_prey_derivatives, self.time_steps, self.period)
        return rk4.integrate(x0)


# Define model parameters
period = [0, 40]
n_time_steps = 2000

# Initialize the Lotka-Volterra model
model = LotkaVolterra(period=period, n_time_steps=n_time_steps)

# Initial populations: [prey, predator]
initial_state = torch.rand(2) # Example initial conditions

def 