import torch
from torch.nn.functional import pad
import matplotlib.pyplot as plt

#set torch rand
torch.manual_seed(0)

def runge_kutta_4(func, x0, params, time_horizon, time_steps):
    dt = time_horizon / time_steps
    X = [x0]
    for i in range(time_steps):
        x = X[-1]
        k1 = func(x, params[i])
        k2 = func(x + dt * k1 / 2, params[i])
        k3 = func(x + dt * k2 / 2, params[i])
        k4 = func(x + dt * k3, params[i])
        X_next = x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        X.append(X_next)
    return torch.stack(X, dim=1)

def lv_func(x, params):
    alpha, beta, gamma, delta = params
    dxdt = torch.zeros(2)
    dxdt[0] = alpha * x[0] - beta * x[0] * x[1]  # Prey population change
    dxdt[1] = -gamma * x[1] + delta * x[0] * x[1] # Predator population change
    return dxdt

def lotka_volterra(params, x0, T=10, nt=1000):
    # Check if params has shape [4,] and expand to [nt, 4] if needed
    if params.ndim == 1 and params.shape[0] == 4:
        # Repeat params along the time dimension to make it [nt, 4]
        params = params.unsqueeze(0).expand(nt, -1)
    elif params.shape != (nt, 4):
        raise ValueError("params must be either [4,] or [nt, 4]")
    
    # Proceed with the Runge-Kutta 4 integration
    return runge_kutta_4(lv_func, x0, params, T, nt)

period = 40.0  # Time horizon as a single float
n_time_steps = 100
params = torch.tensor([2/3, 4/3, 1.0, 1.0], requires_grad=True)
initial_pop = torch.tensor([0.1, 1.0])

solution = lotka_volterra(params, initial_pop, T=period, nt=n_time_steps)

def generate_data_set(initial_pop = initial_pop, period=40.0, n_time_steps=2000, n_realizations=10):
    pop_data_runs = []
    perturbations = []
    
    for run_idx in range(n_realizations):
        print(f'Computing realization {run_idx + 1}/{n_realizations}')
        
        # Generate noise for perturbing alpha across time steps
        noise = torch.randn(1, n_time_steps)  # Shape [1, n_time_steps] for a single parameter over time
        for _ in range(250):  # Smooth out the noise to resemble realistic fluctuations
            noise = pad(noise, pad=(1, 1), mode='reflect')
            noise = (noise[:, :-2] + 2 * noise[:, 1:-1] + noise[:, 2:]) / 4
        noise = noise.squeeze()  # Shape [n_time_steps]
        
        # Base parameters without perturbation, as shape [n_time_steps, 4]
        base_params = torch.tensor([4/3, 2/3, 1, 1]).expand(n_time_steps, 4)
        
        # Apply perturbation to alpha (the first parameter)
        params = base_params.clone()
        params[:, 0] += noise  # Modify alpha over time
                
        # Solve ODE with perturbed parameters
        pop_data = lotka_volterra(params, initial_pop, T=period, nt=n_time_steps)
        
        pop_data_runs.append(pop_data)
        perturbations.append(noise)
    
    return pop_data_runs, perturbations

initial_pop = torch.rand(2)
X_truth_runs, M = generate_data_set(initial_pop = initial_pop, period=period, n_time_steps=n_time_steps, n_realizations=1)

X_true = X_truth_runs[0]  # Shape: (2, 2001)  
d_true = X_true[0, :]

# import python partial function
from functools import partial
from torch.autograd.functional import jacobian, jvp, vjp

# fix all parts of the problem except the parameters
def forward_model(params):
    X = lotka_volterra(params, initial_pop, T=period, nt=n_time_steps)
    prey = X[0, :]
    return prey

# set an initial guess for the parameters
params = torch.tensor([2/3, 4/3, 1.0, 1.0], requires_grad=True)
v = torch.randn_like(params)

d,q = jvp(forward_model, params, v)

w = torch.randn_like(d_true)
d, a = vjp(forward_model, params, w)

# Check adjoint consistency
print(torch.sum(q*w), torch.sum(a*v))


def Hmv(forProb, p, sk):
  q = torch.autograd.functional.jvp(forProb, p, sk)[1]
  a = torch.autograd.functional.vjp(forProb, p, q)[1]
  return a

def conj_gradient(A, b, x0=None, niter=20, tol=1e-2, alpha=1e-2, verbose=True):
    """
    Solve Ax = b using the conjugate gradient method.
    
    Paramters:
        A (callable): A function that computes the matrix-vector product Ax.
        b (torch.Tensor): The right-hand side vector.
        x0 (torch.Tensor, optional): The initial guess. Defaults to None.
        niter (int, optional): Maximum number of iterations. Defaults to 20.
        tol (float, optional): Tolerance for the residual. Defaults to 1e-2.
        alpha (float, optional): Step size for the conjugate gradient method. Defaults to 1e-2. 
    """
    if x0 is None:
        r = b
    else:
        r = b - A(x0)

    q = r
    x = torch.zeros_like(b)
    for i in range(niter):
        Hq    = A(q)
        alpha = (r*r).sum()/(q*Hq).sum()
        x  = x + alpha*q
        rnew  = r - alpha*Hq
        beta = (rnew**2).sum()/(r**2).sum()
        q    = rnew + beta*q
        r    = rnew.clone()
        if verbose:
            print('iter = %3d    res = %3.2e'%(i, r.norm()/b.norm()))
        if r.norm()/b.norm() < tol:
            break
    return x

period = 40.0  # Time horizon as a single float
n_time_steps = 200
initial_pop = torch.rand(2)

# Making a true data set to fit to
XX, M = generate_data_set(initial_pop=initial_pop, period=period, n_time_steps=n_time_steps, n_realizations=1)
X = XX[0]
d_true = X[0, :]  # Use the prey population as the data to fit
# Start with an initial guess for the parameters
p0 = torch.tensor([.7, 1.7, .7, .7], requires_grad=True)
# Extend p0 to repeat over the time steps with individual gradients
p0 = p0.unsqueeze(0).expand(n_time_steps, -1)

#

# fix all parts of the problem except the parameters
def forward_model(params):
    X = lotka_volterra(params, initial_pop, T=period, nt=n_time_steps)
    prey = X[0, :]
    return prey


def gauss_newton_solver(forward_model, p0, data, max_iter=100, tol=1e-6, mu=1):
    predictions = []  # To store predictions at each iteration for animation
    
    params = p0
    for i in range(max_iter):
        # Compute residual
        data_pred = forward_model(params)
        rk = data - data_pred
        
        # Store the current predicted data for animation
        predictions.append(data_pred.detach())
        
        # Compute parts for conjugate gradient
        b = torch.autograd.functional.vjp(forward_model, params, rk)[1]
        def A(sk):
            q = torch.autograd.functional.jvp(forward_model, params, sk)[1]
            a = torch.autograd.functional.vjp(forward_model, params, q)[1]
            return a
        s_k = conj_gradient(A, b, niter=20, tol=1e-2, alpha=1e-2, verbose=False)
        
        # Update parameters
        params = params + mu * s_k
        
        # Check for convergence
        if s_k.norm() < tol:
            print(f'Converged in {i+1} iterations')
            break
        print(f'Iteration {i+1}/{max_iter}: Residual = {rk.norm().item()}')
    
    return params, predictions


from matplotlib.animation import FuncAnimation

def create_animation(true_data, predictions, filename="fitting_animation.gif"):
    fig, ax = plt.subplots()
    line1, = ax.plot([], [], 'r-', label="Predicted Fit")
    line2, = ax.plot([], [], 'b--', label="True Data")
    ax.legend()

    # Set titles and labels
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Population")
    
    def init():
        # Set x and y limits based on true_data and predictions
        ax.set_xlim(0, len(true_data))
        ax.set_ylim(min(true_data.min(), predictions[0].min()) - 0.1, max(true_data.max(), predictions[0].max()) + 0.1)
        line2.set_data(range(len(true_data)), true_data)  # Set true data once, as it remains constant
        ax.set_title("Iteration: 0")  # Initial title for iteration count
        return line1, line2

    def update(i):
        # Update predicted data and title with the current iteration count
        line1.set_data(range(len(predictions[i])), predictions[i])
        ax.set_title(f"Iteration: {i + 1}")
        return line1, line2

    # Create animation with updated frames
    ani = FuncAnimation(fig, update, frames=len(predictions), init_func=init, blit=True)
    ani.save(filename, writer="imagemagick")

# Solve the problem
p_opt, predictions = gauss_newton_solver(forward_model, p0, d_true, max_iter=20, tol=1e-4, mu=1e-1)

# Create the animation
create_animation(d_true.cpu().detach().numpy(), [pred.cpu().numpy() for pred in predictions], "fitting_animation.gif")

# Make a final plot of the both pred prey true data and the predicted data
X_hat = lotka_volterra(p_opt, initial_pop, T=period, nt=n_time_steps)

plt.figure()
plt.plot(X[0, :].detach().numpy(), label='True Prey Population')
plt.plot(X[1, :].detach().numpy(), label='True Predator Population')
plt.plot(X_hat[0, :].detach().numpy(), label='Predicted Prey Population')
plt.plot(X_hat[1, :].detach().numpy(), label='Predicted Predator Population')
plt.legend()
plt.xlabel('Time Steps')
plt.ylabel('Population')
plt.title('True vs Predicted Population')
plt.show()

print("")






