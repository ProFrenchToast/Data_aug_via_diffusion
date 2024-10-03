import numpy as np
from scipy import integrate

import torch


# combines the state and action into a single array 
def combine_state_action(state, action):
    padded_action = torch.nn.functional.pad(action, pad=(0, 103), value=0)
    expanded_action = padded_action.expand([1,1,128,128])
    combined = torch.cat((state, expanded_action), dim=1)
    return combined

# splits the combined state action in to seperate parts
def split_state_action(combined):
    state = combined[:, :3, :, :]
    action_flattened = combined[:, 3, :, :]
    action = action_flattened[0, 0, :25]
    return state, action


class Sde:
    def __init__(self, sigma=25, max_noise=1, img_size=256, action_size=25, img_channels=3, device="cuda"):
        self.sigma = sigma
        self.img_size = img_size
        self.action_size = action_size
        self.img_channels = img_channels
        self.max_noise = max_noise
        self.device = device

        self.example_state = torch.randn(1, self.img_channels, self.img_size, self.img_size, device=self.device)
        self.example_action = torch.randn(1, self.action_size, device=self.device)

    def reset(self):
        """ resets the starting state and action to another from the prior"""
        self.example_state = torch.randn(1, self.img_channels, self.img_size, self.img_size, device=self.device)
        self.example_action = torch.randn(1, self.action_size, device=self.device)

    def marginal_prob_std(self, t):
        """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

        Args:    
            t: A vector of time steps.
            sigma: The $\sigma$ in our SDE.  
        
        Returns:
            The standard deviation.
        """    
        #t = torch.tensor(t, device=self.device)
        return torch.sqrt((self.sigma**(2 * t) - 1.) / 2. / np.log(self.sigma))

    def diffusion_coeff(self, t):
        """Compute the diffusion coefficient of our SDE.

        Args:
            t: A vector of time steps.
            sigma: The $\sigma$ in our SDE.
        
        Returns:
            The vector of diffusion coefficients.
        """
        #return torch.tensor(self.sigma**t, device=self.device)
        return self.sigma**t

    def noise_images(self, x, t):
        z = torch.randn_like(x)
        std = self.marginal_prob_std(t)
        perturbed_x = x + z * std[:, None, None, None]
        return perturbed_x

    def sample_timesteps(self, n, eps=1e-5):
        return torch.rand(n, device=self.device) * (1. - eps) + eps

    def sample(self, score_model, 
                           batch_size=1, 
                           num_steps=500, 
                           eps=1e-3):
        """Generate samples from score-based models with the Euler-Maruyama solver.

        Args:
            score_model: A PyTorch model that represents the time-dependent score-based model.
            marginal_prob_std: A function that gives the standard deviation of
            the perturbation kernel.
            diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
            batch_size: The number of samplers to generate by calling this function once.
            num_steps: The number of sampling steps. 
            Equivalent to the number of discretized time steps.
            device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
            eps: The smallest time step for numerical stability.
        
        Returns:
            Samples.    
        """
        t = torch.ones(batch_size, device=self.device)
        init_x_state = self.example_state * self.marginal_prob_std(t)[:, None, None, None]
        init_x_action = self.example_action * self.marginal_prob_std(t)[:, None]
        time_steps = torch.linspace(1., eps, num_steps, device=self.device)
        step_size = time_steps[0] - time_steps[1]
        x_state = init_x_state
        x_action = init_x_action
        with torch.no_grad():
            for time_step in time_steps:      
                batch_time_step = torch.ones(batch_size, device=self.device) * time_step
                g = self.diffusion_coeff(batch_time_step)
                state_score, action_score = score_model(x_state, x_action, batch_time_step)
                mean_x_state = x_state + (g**2)[:, None, None, None] * state_score * step_size
                mean_x_action = x_action = (g**2)[:, None] * action_score * step_size
                x_state = mean_x_state + torch.sqrt(step_size) * g[:, None, None, None] * torch.randn_like(x_state)  
                x_action = mean_x_action + torch.sqrt(step_size) * g[:, None] * torch.randn_like(x_action)      
        # Do not include any noise in the last sampling step.
        mean_x_state = (mean_x_state.clamp(-1, 1) + 1) / 2
        mean_x_state = (mean_x_state * 255).type(torch.uint8)
        return mean_x_state, mean_x_action

    def loss_fn(self, model, state, action, eps=1e-5):
        """The loss function for training score-based generative models.

        Args:
            model: A PyTorch model instance that represents a 
            time-dependent score-based model.
            x: A mini-batch of training data.    
            marginal_prob_std: A function that gives the standard deviation of 
            the perturbation kernel.
            eps: A tolerance value for numerical stability.
        """
        random_t = torch.rand(state.shape[0], device=state.device) * (1. - eps) + eps  
        z_state = torch.randn_like(state)
        z_action = torch.rand_like(action)
        std = self.marginal_prob_std(random_t)
        perturbed_state = state + z_state * std[:, None, None, None]
        perturbed_action = action + z_action * std[:, None]
        score_state, score_action = model(perturbed_state, perturbed_action, random_t)
        loss_state = torch.mean(torch.sum((score_state * std[:, None, None, None] + z_state)**2, dim=(1,2,3)))
        loss_action = torch.mean(torch.sum((score_action * std[:, None] + z_action)**2, dim=(1)))
        return loss_state + loss_action
    


class ode_sampler:

    def __init__(self, marginal_prob_std, diffusion_coeff, device='cpu'):
        self.device = device
        self.marginal_prob_fn = marginal_prob_std
        self.diffusion_fn = diffusion_coeff


    def ode_sampler(self, score_model,
                batch_size=1, 
                atol=1e-5, 
                rtol=1e-5, 
                z=None,
                eps=1e-3):
        """Generate samples from score-based models with black-box ODE solvers.

        Args:
        score_model: A PyTorch model that represents the time-dependent score-based model.
        marginal_prob_std: A function that returns the standard deviation 
            of the perturbation kernel.
        diffusion_coeff: A function that returns the diffusion coefficient of the SDE.
        batch_size: The number of samplers to generate by calling this function once.
        atol: Tolerance of absolute errors.
        rtol: Tolerance of relative errors.
        device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
        z: The latent code that governs the final sample. If None, we start from p_1;
            otherwise, we start from the given z.
        eps: The smallest time step for numerical stability.
        """
        t = torch.ones(batch_size, device=self.device)
        # Create the latent code
        if z is None:
            init_x = torch.randn(batch_size, 4, 128, 128, device=self.device) \
                * self.marginal_prob_fn(t)[:, None, None, None]
        else:
            init_x = z

        shape = init_x.shape

        def score_eval_wrapper(sample, time_steps):
            """A wrapper of the score-based model for use by the ODE solver."""
            sample = torch.tensor(sample, device=self.device, dtype=torch.float32).reshape(shape)
            time_steps = torch.tensor(time_steps, device=self.device, dtype=torch.float32).reshape((sample.shape[0], ))    
            with torch.no_grad():    
                state, action = split_state_action(sample)
                score_state, score_action = score_model(state, action, time_steps)
                score = combine_state_action(score_state, score_action)
            return score.cpu().numpy().reshape((-1,)).astype(np.float64)

        def ode_func(t, x):        
            """The ODE function for use by the ODE solver."""
            time_steps = np.ones((shape[0],)) * t    
            g = self.diffusion_fn(torch.tensor(t)).cpu().numpy()
            return  -0.5 * (g**2) * score_eval_wrapper(x, time_steps)

        # Run the black-box ODE solver.
        res = integrate.solve_ivp(ode_func, (1., eps), init_x.reshape(-1).cpu().numpy(), rtol=rtol, atol=atol, method='RK45')  
        print(f"Number of function evaluations: {res.nfev}")
        x = torch.tensor(res.y[:, -1], device=self.device).reshape(shape)

        state, action = split_state_action(x)
        return state, action
    

    def prior_likelihood(self, z, sigma):
        """The likelihood of a Gaussian distribution with mean zero and 
            standard deviation sigma."""
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2. * torch.log(2*np.pi*sigma**2) - torch.sum(z**2, dim=(1,2,3)) / (2 * sigma**2)

    def ode_likelihood(self, x, 
                    score_model, 
                    batch_size=1, 
                    device='cuda',
                    eps=1e-5):
        """Compute the likelihood with probability flow ODE.

        Args:
        x: Input data.
        score_model: A PyTorch model representing the score-based model.
        marginal_prob_std: A function that gives the standard deviation of the 
            perturbation kernel.
        diffusion_coeff: A function that gives the diffusion coefficient of the 
            forward SDE.
        batch_size: The batch size. Equals to the leading dimension of `x`.
        device: 'cuda' for evaluation on GPUs, and 'cpu' for evaluation on CPUs.
        eps: A `float` number. The smallest time step for numerical stability.

        Returns:
        z: The latent code for `x`.
        bpd: The log-likelihoods in bits/dim.
        """

        # Draw the random Gaussian sample for Skilling-Hutchinson's estimator.
        epsilon = torch.randn_like(x, device=device)
            
        def divergence_eval(sample, time_steps, epsilon):      
            """Compute the divergence of the score-based model with Skilling-Hutchinson."""
            with torch.enable_grad():
                sample.requires_grad_(True)
                state, action = split_state_action(sample)
                score_state, score_action = score_model(state, action, time_steps)
                score_combined = combine_state_action(score_state, score_action)
                score_e = torch.sum(score_combined * epsilon)
                grad_score_e = torch.autograd.grad(score_e, sample)[0]
            return torch.sum(grad_score_e * epsilon, dim=(1, 2, 3))    

        shape = x.shape

        def score_eval_wrapper(sample, time_steps):
            """A wrapper for evaluating the score-based model for the black-box ODE solver."""
            sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
            time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))    
            with torch.no_grad():  
                state, action = split_state_action(sample)  
                score_state, score_action = score_model(state, action, time_steps)
                score = combine_state_action(score_state, score_action)
            return score.cpu().numpy().reshape((-1,)).astype(np.float64)

        def divergence_eval_wrapper(sample, time_steps):
            """A wrapper for evaluating the divergence of score for the black-box ODE solver."""
            with torch.no_grad():
                # Obtain x(t) by solving the probability flow ODE.
                sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
                time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))    
                # Compute likelihood.
                div = divergence_eval(sample, time_steps, epsilon)
                return div.cpu().numpy().reshape((-1,)).astype(np.float64)

        def ode_func(t, x):
            """The ODE function for the black-box solver."""
            time_steps = np.ones((shape[0],)) * t    
            sample = x[:-shape[0]]
            logp = x[-shape[0]:]
            g = self.diffusion_fn(torch.tensor(t)).cpu().numpy()
            sample_grad = -0.5 * g**2 * score_eval_wrapper(sample, time_steps)
            logp_grad = -0.5 * g**2 * divergence_eval_wrapper(sample, time_steps)
            return np.concatenate([sample_grad, logp_grad], axis=0)

        init = np.concatenate([x.cpu().numpy().reshape((-1,)), np.zeros((shape[0],))], axis=0)
        # Black-box ODE solver
        res = integrate.solve_ivp(ode_func, (eps, 1.), init, rtol=1e-5, atol=1e-5, method='RK45')  
        zp = torch.tensor(res.y[:, -1], device=device)
        z = zp[:-shape[0]].reshape(shape)
        delta_logp = zp[-shape[0]:].reshape(shape[0])
        sigma_max = self.marginal_prob_fn(torch.tensor([1.], device=device))
        prior_logp = self.prior_likelihood(z, sigma_max)
        bpd = -(prior_logp + delta_logp) / np.log(2)
        N = np.prod(shape[1:])
        bpd = bpd / N + 8.
        return z, bpd
        
    

    


        

