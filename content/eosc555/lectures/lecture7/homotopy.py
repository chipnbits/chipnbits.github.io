import matplotlib.pyplot as plt
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
x1 = torch.linspace(-5, 5, 100)
x2 = torch.linspace(-5, 5, 100)
X1, X2 = torch.meshgrid(x1, x2, indexing='ij')
X = torch.stack([X1.flatten(), X2.flatten()], dim=1)
y = objective_function(X).reshape(100, 100)
plt.contourf(X1, X2, y, levels=100)
plt.colorbar()
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Objective function')
plt.show()
