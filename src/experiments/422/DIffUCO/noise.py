from __future__ import annotations

import torch

class BernoulliNoiseSchedule:
    def __init__(self, steps: int, beta_start: float, beta_end: float, device: torch.device):
        self.steps = steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, steps, device=device)

    def prior_log_prob(self, state: torch.Tensor) -> torch.Tensor:
        log_half = torch.log(torch.tensor(0.5, device=state.device, dtype=state.dtype))
        return torch.full((state.shape[0],), state.shape[1] * log_half, device=state.device, dtype=state.dtype)

    def sample_prior(self, batch_size: int, num_nodes: int, stochastic: bool) -> torch.Tensor:
        if stochastic:
            return torch.bernoulli(0.5 * torch.ones(batch_size, num_nodes, device=self.device))
        return torch.full((batch_size, num_nodes), 0.5, device=self.device)

    def transition_log_prob(self, x_prev: torch.Tensor, x_next: torch.Tensor, time_step: int) -> torch.Tensor:
        beta_t = self.betas[time_step]
        stay_prob = torch.where(x_prev == x_next, 1.0 - beta_t, beta_t)
        stay_prob = stay_prob.clamp_min(1e-8)
        return torch.log(stay_prob).sum(dim=-1)

    def relaxed_transition_log_prob(
        self, probabilities_prev: torch.Tensor, x_next: torch.Tensor, time_step: int
    ) -> torch.Tensor:
        beta_t = self.betas[time_step]
        log_stay = torch.log((1.0 - beta_t).clamp_min(1e-8))
        log_flip = torch.log(beta_t.clamp_min(1e-8))
        expected_log_prob = (
            x_next * (probabilities_prev * log_stay + (1.0 - probabilities_prev) * log_flip)
            + (1.0 - x_next) * (probabilities_prev * log_flip + (1.0 - probabilities_prev) * log_stay)
        )
        return expected_log_prob.sum(dim=-1)
