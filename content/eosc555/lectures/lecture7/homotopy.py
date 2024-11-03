import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch

def objective_function(X):
    """
    A 2D function with multiple local minima.

    Args:
        X: A tensor of shape (N, 2), where each row is a point [x1, x2].

    Returns:
        y: A tensor of shape (N,), representing the function value at each point.
    """
    x1 = X[:, 0]
    x2 = X[:, 1]
    y = -torch.exp(-((x1 - 2)**2 + (x2 - 2)**2) / 0.1) \
        - 2 * torch.exp(-((x1 + 2)**2 + (x2 + 2)**2) / 0.1)
    return y

# plot the function
x1 = torch.linspace(-8, 8, 100)
x2 = torch.linspace(-8, 8, 100)
X1, X2 = torch.meshgrid(x1, x2, indexing='ij')
X = torch.stack([X1.flatten(), X2.flatten()], dim=1)
y = objective_function(X).reshape(100, 100)

def gaussian_homotopy(func, x, batch_size=1000, sigma=1.0, t=0.5):
    """
    Computes the Gaussian homotopy function h(x, t) using Monte Carlo approximation.

    Args:
        func: The original objective function to be optimized.
        x: A tensor of shape (N, D), where N is the number of points and D is the dimension.
        batch_size: Number of samples to use in Monte Carlo approximation.
        sigma: Standard deviation of the Gaussian kernel.
        t: Homotopy parameter, varies from 0 (original function) to 1 (smoothed function).

    Returns:
        y: A tensor of shape (N,), the approximated h(x, t) values at each point x.
    """
    N, D = x.shape
    
    # Sample from the t=1 gaussian kernel
    kernel = torch.randn(batch_size, D) * sigma
    
    # Repeat x and z to compute all combinations
    x_repeated = x.unsqueeze(1).repeat(1, batch_size, 1).view(-1, D)
    kernel_repeated = kernel.repeat(N, 1)
    
    # Compute the monte carlo set of points surrounding each x
    x_input = x_repeated - t * kernel_repeated
    
    # Evaluate the function at the sampled points
    y_input = func(x_input)
    
    # Reshape and average over the batch size to approximate the expectation
    y = y_input.view(N, batch_size).mean(dim=1)
    return y

# Create grid points
t_values = torch.linspace(-8, 8, 129)
x_grid, y_grid = torch.meshgrid(t_values, t_values, indexing="ij")
X = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=1)

# Define the range of homotopy parameters
T = torch.linspace(1.0, 0.0, steps=30)

# Initialize figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

def update(i):
    ax1.clear()
    ax2.clear()

    t = T[i]
    y_original = objective_function(X).view(129, 129).detach().numpy()
    y_homotopy = gaussian_homotopy(objective_function, X, batch_size=12000, sigma=4.0, t=t).view(129, 129).detach().numpy()

    ax1.contourf(t_values.numpy(), t_values.numpy(), y_original, levels=50, cmap="viridis")
    ax1.set_title("Original Function")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")

    ax2.contourf(t_values.numpy(), t_values.numpy(), y_homotopy, levels=50, cmap="viridis")
    ax2.set_title(f"Homotopy Function (t = {t:.2f})")
    ax2.set_xlabel("x1")
    ax2.set_ylabel("x2")

    plt.tight_layout()

# Create animation
ani = FuncAnimation(fig, update, frames=len(T), interval=200)

# Save as GIF
ani.save("imgs/homotopy_2d.gif", writer="imagemagick", fps=4)