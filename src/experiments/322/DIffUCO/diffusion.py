from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch import nn

if __package__ in (None, ""):
    from energy import GeometricHyperplaneEnergy
    from noise import BernoulliNoiseSchedule
else:
    from .energy import GeometricHyperplaneEnergy
    from .noise import BernoulliNoiseSchedule


@dataclass
class DiffusionConfig:
    steps: int = 12
    beta_start: float = 0.08
    beta_end: float = 0.28
    logit_clip: float = 12.0
    terminal_temperature: float = 1.0
    reinforce_weight: float = 0.0


class BernoulliDiffusionProcess:
    def __init__(self, config: DiffusionConfig, device: torch.device):
        self.config = config
        self.device = device
        self.noise = BernoulliNoiseSchedule(
            steps=config.steps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            device=device,
        )

    def initial_state(self, batch_size: int, num_nodes: int, stochastic: bool) -> torch.Tensor:
        return self.noise.sample_prior(batch_size=batch_size, num_nodes=num_nodes, stochastic=stochastic)

    def _bernoulli_log_prob(self, probabilities: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
        probabilities = probabilities.clamp(1e-5, 1.0 - 1e-5)
        return (samples * torch.log(probabilities) + (1.0 - samples) * torch.log(1.0 - probabilities)).sum(dim=-1)

    def _reinforce_with_baseline(self, values: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
        baseline = values.detach().mean()
        advantages = values.detach() - baseline
        return (advantages * log_q).mean()

    def reverse_step(
        self,
        model: nn.Module,
        energy: GeometricHyperplaneEnergy,
        point_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        current_state: torch.Tensor,
        time_step: int,
        stochastic: bool,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        t_index = torch.full((current_state.shape[0],), time_step, device=current_state.device, dtype=torch.long)
        logits = model(point_features, current_state, t_index, edge_index, edge_type)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=self.config.logit_clip, neginf=-self.config.logit_clip)
        logits = logits.clamp(-self.config.logit_clip, self.config.logit_clip)
        probabilities = torch.sigmoid(logits).clamp(1e-4, 1.0 - 1e-4)

        if stochastic:
            previous_state = torch.bernoulli(probabilities.detach())
        else:
            previous_state = probabilities

        log_q_t = self._bernoulli_log_prob(probabilities, previous_state.detach() if stochastic else probabilities.detach())
        log_p_t = self.noise.transition_log_prob(previous_state.detach(), current_state.detach(), time_step)
        relaxed_log_p_t = self.noise.relaxed_transition_log_prob(probabilities, current_state.detach(), time_step)

        energy_terms = energy(previous_state.float())
        relaxed_energy_terms = energy(probabilities)
        stable_energy = {
            key: torch.nan_to_num(value, nan=0.0, posinf=1e6, neginf=-1e6) if torch.is_tensor(value) else value
            for key, value in energy_terms.items()
        }
        stable_relaxed_energy = {
            key: torch.nan_to_num(value, nan=0.0, posinf=1e6, neginf=-1e6) if torch.is_tensor(value) else value
            for key, value in relaxed_energy_terms.items()
        }
        return previous_state, {
            "state": previous_state,
            "probabilities": probabilities,
            "guided_probabilities": probabilities,
            "log_q_t": log_q_t,
            "log_p_t": log_p_t,
            "relaxed_log_p_t": relaxed_log_p_t,
            "energy": stable_energy["total"],
            "relaxed_energy": stable_relaxed_energy["total"],
            "plane": stable_energy["plane"],
            "facet_dim": stable_energy.get("facet_dim", torch.zeros_like(stable_energy["plane"])),
            "second_eigenvalue": stable_energy.get("second_eigenvalue", torch.zeros_like(stable_energy["plane"])),
            "boundary": stable_energy["boundary"],
            "cardinality": stable_energy["cardinality"],
        }

    def rollout(
        self,
        model: nn.Module,
        energy: GeometricHyperplaneEnergy,
        point_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        batch_size: int,
        stochastic: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        current_state = self.initial_state(batch_size=batch_size, num_nodes=point_features.shape[0], stochastic=stochastic)
        history: List[Dict[str, torch.Tensor]] = []
        log_q_T = self.noise.prior_log_prob(current_state.float())

        for time_step in reversed(range(self.config.steps)):
            current_state, step_stats = self.reverse_step(
                model=model,
                energy=energy,
                point_features=point_features,
                edge_index=edge_index,
                edge_type=edge_type,
                current_state=current_state,
                time_step=time_step,
                stochastic=stochastic,
            )
            history.append(step_stats)

        final_state = current_state.float()
        final_probabilities = history[-1]["probabilities"]
        final_energy = history[-1]["energy"]
        final_relaxed_energy = history[-1]["relaxed_energy"]
        log_p_0 = energy.log_p_0(final_state, temperature=self.config.terminal_temperature).detach()
        relaxed_log_p_0 = energy.log_p_0(final_probabilities, temperature=self.config.terminal_temperature)
        return final_state, {
            "steps": history,
            "log_q_T": log_q_T.detach(),
            "log_p_0": log_p_0,
            "relaxed_log_p_0": relaxed_log_p_0,
            "final_state": final_state,
            "final_probabilities": final_probabilities,
            "final_energy": final_energy,
            "final_relaxed_energy": final_relaxed_energy,
        }

    def trajectory_log_q(self, rollout: Dict[str, object]) -> torch.Tensor:
        step_terms = torch.stack([step["log_q_t"] for step in rollout["steps"]], dim=0).sum(dim=0)
        return rollout["log_q_T"] + step_terms

    def trajectory_log_p(self, rollout: Dict[str, object]) -> torch.Tensor:
        step_terms = torch.stack([step["log_p_t"] for step in rollout["steps"]], dim=0).sum(dim=0)
        return rollout["log_p_0"] + step_terms

    def loss_terms(self, rollout: Dict[str, object]) -> Dict[str, torch.Tensor]:
        final_probs = rollout["final_probabilities"].clamp(1e-4, 1.0 - 1e-4)
        entropy_per_step = []
        for step in rollout["steps"]:
            probs = step["probabilities"].clamp(1e-4, 1.0 - 1.0e-4)
            entropy = -(probs * torch.log(probs) + (1.0 - probs) * torch.log(1.0 - probs)).sum(dim=-1)
            entropy_per_step.append(entropy)
        entropy_per_sample = torch.stack(entropy_per_step, dim=0).sum(dim=0)
        l_entropy = entropy_per_sample.mean()

        log_p_transitions = torch.stack([step["log_p_t"] for step in rollout["steps"]], dim=0).sum(dim=0)
        relaxed_log_p_transitions = torch.stack([step["relaxed_log_p_t"] for step in rollout["steps"]], dim=0).sum(dim=0)
        noise_cost_per_sample = -self.config.terminal_temperature * log_p_transitions
        relaxed_noise_cost_per_sample = -self.config.terminal_temperature * relaxed_log_p_transitions
        l_noise = relaxed_noise_cost_per_sample.mean()

        energy_cost_per_sample = rollout.get("terminal_cost", rollout["final_energy"])
        relaxed_energy_cost_per_sample = rollout.get("final_relaxed_total_energy", rollout["final_relaxed_energy"])
        l_energy = relaxed_energy_cost_per_sample.mean()
        log_q_per_sample = self.trajectory_log_q(rollout)
        log_p_per_sample = self.trajectory_log_p(rollout)

        entropy_reinforce = self._reinforce_with_baseline(-entropy_per_sample, log_q_per_sample)
        noise_reinforce = self._reinforce_with_baseline(noise_cost_per_sample, log_q_per_sample)
        energy_reinforce = self._reinforce_with_baseline(energy_cost_per_sample, log_q_per_sample)
        return {
            "entropy": l_entropy,
            "noise": l_noise,
            "energy": l_energy,
            "memory_energy": rollout.get(
                "final_relaxed_memory_energy",
                torch.zeros_like(relaxed_energy_cost_per_sample),
            ).mean(),
            "population_objective": rollout.get(
                "population_objective_term",
                torch.zeros((), device=relaxed_energy_cost_per_sample.device, dtype=relaxed_energy_cost_per_sample.dtype),
            ),
            "entropy_reinforce": entropy_reinforce,
            "noise_reinforce": noise_reinforce,
            "energy_reinforce": energy_reinforce,
            "trajectory_log_q": log_q_per_sample.mean(),
            "trajectory_log_p": log_p_per_sample.mean(),
        }

    def training_loss(self, rollout: Dict[str, object]) -> torch.Tensor:
        terms = self.loss_terms(rollout)
        relaxed_loss = (
            -self.config.terminal_temperature * terms["entropy"]
            + terms["noise"]
            + terms["energy"]
            + terms["population_objective"]
        )
        reinforce_loss = (
            -self.config.terminal_temperature * terms["entropy_reinforce"]
            + terms["noise_reinforce"]
            + terms["energy_reinforce"]
        )
        loss = relaxed_loss + self.config.reinforce_weight * reinforce_loss
        return torch.nan_to_num(loss, nan=1e6, posinf=1e6, neginf=-1e6)
