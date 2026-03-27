from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import optim

if __package__ in (None, ""):
    import sys

    CURRENT_DIR = Path(__file__).resolve().parent
    sys.path.append(str(CURRENT_DIR))
    from diffusion import BernoulliDiffusionProcess, DiffusionConfig
    from energy import EnergyConfig, GeometricHyperplaneEnergy
    from facet_reference import build_reference_database
    from geometry import generate_points_322
    from inference import sample_unique_facets
    from model import GeometricDiffUCOModel
    from orbit_memory import OrbitMemoryBank, OrbitMemoryConfig
    from plane_density import PlaneDensityBank, PlaneDensityConfig
    from plane_population import PlanePopulationConfig, PlanePopulationObjective
    from plane_set_dpp import PlaneSetDPPConfig, PlaneSetDPPObjective
    from plane_set_coverage import PlaneSetCoverageConfig, PlaneSetCoverageObjective
    from plane_memory import PlaneOrbitMemoryBank, PlaneOrbitMemoryConfig
    from orbit_terminal import OrbitTerminalConfig, build_orbit_terminal_terms
    from orbit_tracker import OrbitCoverageTracker, OrbitRewardConfig
    from symmetry_graph import build_symmetry_graph
else:
    from .diffusion import BernoulliDiffusionProcess, DiffusionConfig
    from .energy import EnergyConfig, GeometricHyperplaneEnergy
    from .facet_reference import build_reference_database
    from .geometry import generate_points_322
    from .inference import sample_unique_facets
    from .model import GeometricDiffUCOModel
    from .orbit_memory import OrbitMemoryBank, OrbitMemoryConfig
    from .plane_density import PlaneDensityBank, PlaneDensityConfig
    from .plane_population import PlanePopulationConfig, PlanePopulationObjective
    from .plane_set_dpp import PlaneSetDPPConfig, PlaneSetDPPObjective
    from .plane_set_coverage import PlaneSetCoverageConfig, PlaneSetCoverageObjective
    from .plane_memory import PlaneOrbitMemoryBank, PlaneOrbitMemoryConfig
    from .orbit_terminal import OrbitTerminalConfig, build_orbit_terminal_terms
    from .orbit_tracker import OrbitCoverageTracker, OrbitRewardConfig
    from .symmetry_graph import build_symmetry_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Geometric-DiffUCO experiment for the 3-2-2 Bell polytope.")
    parser.add_argument("--epochs", type=int, default=400, help="Number of optimization epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Number of parallel noise seeds per batch.")
    parser.add_argument("--steps", type=int, default=12, help="Diffusion reverse steps.")
    parser.add_argument("--hidden-dim", type=int, default=128, help="GAT hidden width.")
    parser.add_argument("--layers", type=int, default=4, help="Number of message passing layers.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout used in the GAT backbone.")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate.")
    parser.add_argument("--active-probability-power", type=float, default=1.0, help="Exponent applied to active-point probabilities before computing the active boundary loss.")
    parser.add_argument("--active-loss-type", type=str, choices=["quadratic", "huber"], default="quadratic", help="Residual used by the active boundary loss.")
    parser.add_argument("--active-huber-delta", type=float, default=0.1, help="Huber delta used when active-loss-type=huber.")
    parser.add_argument("--plane-topk", type=int, default=0, help="If > 0, use only the top-k soft points to fit the relaxed hyperplane.")
    parser.add_argument("--active-topk", type=int, default=0, help="If > 0, use only the top-k active weights in the relaxed active loss.")
    parser.add_argument("--logit-clip", type=float, default=12.0, help="Clamp applied to logits during diffusion.")
    parser.add_argument("--terminal-temperature", type=float, default=1.0, help="Temperature used in the terminal Boltzmann factor p(X_0) proportional to exp(-E(X_0)/T).")
    parser.add_argument(
        "--reinforce-weight",
        type=float,
        default=0.0,
        help="Weight applied to the high-variance REINFORCE trajectory correction. Use 0.0 for the low-variance relaxed objective.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Torch device string.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--sample-every", type=int, default=50, help="How often to run validation sampling. Use 0 to disable sampling during training.")
    parser.add_argument("--orbit-reward-scale", type=float, default=1.0, help="Scale applied to the orbit-aware novelty reward correction.")
    parser.add_argument("--exact-novelty-weight", type=float, default=0.5, help="Novelty reward weight for exact matched classes.")
    parser.add_argument("--unknown-novelty-weight", type=float, default=0.2, help="Novelty reward weight for unknown supporting faces grouped by canonical row.")
    parser.add_argument("--orbit-size-penalty", type=float, default=0.15, help="Penalty coefficient for large orbits in the orbit-aware reward.")
    parser.add_argument("--orbit-invalid-energy", type=float, default=250.0, help="Fallback terminal energy assigned to invalid hard terminal states in the orbit-aware objective.")
    parser.add_argument("--memory-repulsion-weight", type=float, default=1.0, help="Maximum weight for the relaxed orbit-memory repulsion added to the geometric energy.")
    parser.add_argument("--memory-warmup-start", type=float, default=0.2, help="Training progress fraction where relaxed orbit-memory repulsion starts ramping up.")
    parser.add_argument("--memory-warmup-end", type=float, default=0.5, help="Training progress fraction where relaxed orbit-memory repulsion reaches full strength.")
    parser.add_argument("--memory-per-class-capacity", type=int, default=8, help="Maximum stored exact-match prototypes per canonical class.")
    parser.add_argument("--memory-activation-threshold", type=int, default=5, help="Minimum exact-match count before a canonical class contributes relaxed repulsion.")
    parser.add_argument("--memory-max-active-classes", type=int, default=4, help="Maximum number of high-frequency canonical classes contributing relaxed repulsion.")
    parser.add_argument("--memory-dedup-threshold", type=float, default=0.92, help="Soft Jaccard similarity threshold used to skip near-duplicate memory prototypes.")
    parser.add_argument("--memory-sigma", type=float, default=0.18, help="Width of the local kernel used by the relaxed orbit-memory repulsion.")
    parser.add_argument("--memory-gamma", type=float, default=0.5, help="Scale applied to per-class relaxed orbit-memory repulsion weights.")
    parser.add_argument("--memory-tau-orbit", type=float, default=0.05, help="Softmin temperature across symmetry-expanded prototype permutations.")
    parser.add_argument("--memory-tau-prototype", type=float, default=0.05, help="Softmin temperature across prototypes within the same canonical class.")
    parser.add_argument("--plane-memory-weight", type=float, default=0.0, help="Maximum weight for plane-space orbit-aware relaxed repulsion.")
    parser.add_argument("--plane-memory-warmup-start", type=float, default=0.2, help="Training progress fraction where plane-space relaxed repulsion starts ramping up.")
    parser.add_argument("--plane-memory-warmup-end", type=float, default=0.5, help="Training progress fraction where plane-space relaxed repulsion reaches full strength.")
    parser.add_argument("--plane-memory-activation-threshold", type=int, default=5, help="Minimum exact-match count before a canonical class contributes plane-space relaxed repulsion.")
    parser.add_argument("--plane-memory-max-active-classes", type=int, default=4, help="Maximum number of canonical classes contributing plane-space relaxed repulsion.")
    parser.add_argument("--plane-memory-gamma", type=float, default=0.5, help="Scale applied to per-class plane-space relaxed repulsion weights.")
    parser.add_argument("--plane-memory-sigma", type=float, default=0.12, help="Width of the local kernel used by plane-space relaxed repulsion.")
    parser.add_argument("--plane-memory-tau-orbit", type=float, default=0.03, help="Softmax temperature across orbit-expanded plane prototypes.")
    parser.add_argument("--plane-density-weight", type=float, default=0.0, help="Maximum weight for density-based plane-space novelty bonus.")
    parser.add_argument("--plane-density-warmup-start", type=float, default=0.2, help="Training progress fraction where plane-density novelty starts ramping up.")
    parser.add_argument("--plane-density-warmup-end", type=float, default=0.5, help="Training progress fraction where plane-density novelty reaches full strength.")
    parser.add_argument("--plane-density-capacity", type=int, default=128, help="Maximum number of plane prototypes kept in the density memory.")
    parser.add_argument("--plane-density-dedup-threshold", type=float, default=0.995, help="Cosine similarity threshold used to skip near-duplicate plane prototypes.")
    parser.add_argument("--plane-density-sigma", type=float, default=0.12, help="Kernel width used by the plane-density novelty estimator.")
    parser.add_argument("--plane-density-novelty-scale", type=float, default=1.0, help="Scale of the raw plane-density novelty bonus before warmup weighting.")
    parser.add_argument("--plane-density-plane-threshold", type=float, default=0.08, help="Soft plane eigenvalue threshold used by the novelty quality gate.")
    parser.add_argument("--plane-density-boundary-threshold", type=float, default=6.0, help="Soft boundary threshold used by the novelty quality gate.")
    parser.add_argument("--plane-density-cardinality-margin", type=float, default=1.0, help="Extra cardinality margin above min_cardinality required by the novelty quality gate.")
    parser.add_argument("--plane-density-gate-sharpness", type=float, default=25.0, help="Sharpness of the sigmoid quality gates used by plane-density novelty.")
    parser.add_argument("--plane-set-coverage-weight", type=float, default=0.0, help="Maximum weight for batch-level plane-space coverage bonus.")
    parser.add_argument("--plane-set-coverage-warmup-start", type=float, default=0.2, help="Training progress fraction where batch-level plane coverage starts ramping up.")
    parser.add_argument("--plane-set-coverage-warmup-end", type=float, default=0.5, help="Training progress fraction where batch-level plane coverage reaches full strength.")
    parser.add_argument("--plane-set-coverage-scale", type=float, default=1.0, help="Scale of the raw batch-level plane coverage bonus.")
    parser.add_argument("--plane-set-coverage-plane-threshold", type=float, default=0.08, help="Soft plane eigenvalue threshold used by the batch-level coverage quality gate.")
    parser.add_argument("--plane-set-coverage-boundary-threshold", type=float, default=6.0, help="Soft boundary threshold used by the batch-level coverage quality gate.")
    parser.add_argument("--plane-set-coverage-cardinality-margin", type=float, default=1.0, help="Extra cardinality margin above min_cardinality required by the batch-level coverage quality gate.")
    parser.add_argument("--plane-set-coverage-gate-sharpness", type=float, default=25.0, help="Sharpness of the sigmoid quality gates used by the batch-level coverage bonus.")
    parser.add_argument("--plane-set-coverage-neighbors", type=int, default=1, help="How many nearest plane neighbors to penalize when computing batch-level coverage.")
    parser.add_argument("--plane-set-dpp-weight", type=float, default=0.0, help="Maximum weight for batch-level DPP/log-det plane diversity bonus.")
    parser.add_argument("--plane-set-dpp-warmup-start", type=float, default=0.2, help="Training progress fraction where batch-level DPP bonus starts ramping up.")
    parser.add_argument("--plane-set-dpp-warmup-end", type=float, default=0.5, help="Training progress fraction where batch-level DPP bonus reaches full strength.")
    parser.add_argument("--plane-set-dpp-scale", type=float, default=1.0, help="Scale of the raw batch-level DPP bonus.")
    parser.add_argument("--plane-set-dpp-plane-threshold", type=float, default=0.08, help="Soft plane eigenvalue threshold used by the batch-level DPP quality gate.")
    parser.add_argument("--plane-set-dpp-boundary-threshold", type=float, default=6.0, help="Soft boundary threshold used by the batch-level DPP quality gate.")
    parser.add_argument("--plane-set-dpp-cardinality-margin", type=float, default=1.0, help="Extra cardinality margin above min_cardinality required by the batch-level DPP quality gate.")
    parser.add_argument("--plane-set-dpp-gate-sharpness", type=float, default=25.0, help="Sharpness of the sigmoid quality gates used by the batch-level DPP bonus.")
    parser.add_argument("--plane-set-dpp-similarity-temperature", type=float, default=8.0, help="Temperature used to convert plane similarities into a DPP kernel.")
    parser.add_argument("--plane-set-dpp-diagonal-jitter", type=float, default=1e-3, help="Diagonal jitter added to the DPP kernel for numerical stability.")
    parser.add_argument("--plane-population-weight", type=float, default=0.0, help="Weight applied to the batch-level elite population objective.")
    parser.add_argument("--plane-population-warmup-start", type=float, default=0.2, help="Training progress fraction where the elite population objective starts ramping up.")
    parser.add_argument("--plane-population-warmup-end", type=float, default=0.5, help="Training progress fraction where the elite population objective reaches full strength.")
    parser.add_argument("--plane-population-elite-count", type=int, default=4, help="Number of elite soft terminal samples used by the population objective.")
    parser.add_argument("--plane-population-diversity-scale", type=float, default=1.0, help="Relative scale of elite diversity inside the population objective.")
    parser.add_argument("--plane-population-plane-threshold", type=float, default=0.08, help="Soft plane eigenvalue threshold used by the population objective quality gate.")
    parser.add_argument("--plane-population-boundary-threshold", type=float, default=6.0, help="Soft boundary threshold used by the population objective quality gate.")
    parser.add_argument("--plane-population-cardinality-margin", type=float, default=1.0, help="Extra cardinality margin above min_cardinality required by the population objective quality gate.")
    parser.add_argument("--plane-population-gate-sharpness", type=float, default=25.0, help="Sharpness of the sigmoid quality gates used by the population objective.")
    parser.add_argument("--plane-population-gate-penalty-scale", type=float, default=50.0, help="Penalty added to low-gate samples before elite selection in the population objective.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("src/experiments/322/DIffUCO/outputs"),
        help="Directory for checkpoints and summaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    points = generate_points_322().to(device)
    graph = build_symmetry_graph()
    edge_index = graph.edge_index.to(device)
    edge_type = graph.edge_type.to(device)

    model = GeometricDiffUCOModel(
        point_dim=points.shape[1],
        edge_types=len(graph.generator_names),
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        dropout=args.dropout,
    ).to(device)
    energy_config = EnergyConfig(
        active_probability_power=args.active_probability_power,
        active_loss_type=args.active_loss_type,
        active_huber_delta=args.active_huber_delta,
        plane_topk=args.plane_topk,
        active_topk=args.active_topk,
    )
    energy = GeometricHyperplaneEnergy(points=points, config=energy_config)
    diffusion_config = DiffusionConfig(
        steps=args.steps,
        logit_clip=args.logit_clip,
        terminal_temperature=args.terminal_temperature,
        reinforce_weight=args.reinforce_weight,
    )
    diffusion = BernoulliDiffusionProcess(diffusion_config, device=device)
    reference = build_reference_database()
    orbit_tracker = OrbitCoverageTracker()
    orbit_reward_config = OrbitRewardConfig(
        exact_novelty_weight=args.exact_novelty_weight * args.orbit_reward_scale,
        unknown_novelty_weight=args.unknown_novelty_weight * args.orbit_reward_scale,
        orbit_size_penalty=args.orbit_size_penalty * args.orbit_reward_scale,
        apply_orbit_reward=args.orbit_reward_scale != 0.0,
    )
    orbit_terminal_config = OrbitTerminalConfig(invalid_energy=args.orbit_invalid_energy)
    memory_config = OrbitMemoryConfig(
        per_class_capacity=args.memory_per_class_capacity,
        activation_count_threshold=args.memory_activation_threshold,
        max_active_classes=args.memory_max_active_classes,
        similarity_dedup_threshold=args.memory_dedup_threshold,
        sigma=args.memory_sigma,
        gamma=args.memory_gamma,
        tau_orbit=args.memory_tau_orbit,
        tau_prototype=args.memory_tau_prototype,
    )
    orbit_memory = OrbitMemoryBank(memory_config)
    plane_memory_config = PlaneOrbitMemoryConfig(
        activation_count_threshold=args.plane_memory_activation_threshold,
        max_active_classes=args.plane_memory_max_active_classes,
        gamma=args.plane_memory_gamma,
        sigma=args.plane_memory_sigma,
        tau_orbit=args.plane_memory_tau_orbit,
    )
    plane_orbit_memory = PlaneOrbitMemoryBank(plane_memory_config)
    plane_density_config = PlaneDensityConfig(
        capacity=args.plane_density_capacity,
        dedup_similarity_threshold=args.plane_density_dedup_threshold,
        sigma=args.plane_density_sigma,
        novelty_scale=args.plane_density_novelty_scale,
        plane_threshold=args.plane_density_plane_threshold,
        boundary_threshold=args.plane_density_boundary_threshold,
        cardinality_margin=args.plane_density_cardinality_margin,
        gate_sharpness=args.plane_density_gate_sharpness,
    )
    plane_density_bank = PlaneDensityBank(plane_density_config)
    plane_set_coverage_config = PlaneSetCoverageConfig(
        novelty_scale=args.plane_set_coverage_scale,
        plane_threshold=args.plane_set_coverage_plane_threshold,
        boundary_threshold=args.plane_set_coverage_boundary_threshold,
        cardinality_margin=args.plane_set_coverage_cardinality_margin,
        gate_sharpness=args.plane_set_coverage_gate_sharpness,
        neighbor_count=args.plane_set_coverage_neighbors,
    )
    plane_set_coverage = PlaneSetCoverageObjective(plane_set_coverage_config)
    plane_set_dpp_config = PlaneSetDPPConfig(
        reward_scale=args.plane_set_dpp_scale,
        plane_threshold=args.plane_set_dpp_plane_threshold,
        boundary_threshold=args.plane_set_dpp_boundary_threshold,
        cardinality_margin=args.plane_set_dpp_cardinality_margin,
        gate_sharpness=args.plane_set_dpp_gate_sharpness,
        similarity_temperature=args.plane_set_dpp_similarity_temperature,
        diagonal_jitter=args.plane_set_dpp_diagonal_jitter,
    )
    plane_set_dpp = PlaneSetDPPObjective(plane_set_dpp_config)
    plane_population_config = PlanePopulationConfig(
        elite_count=args.plane_population_elite_count,
        diversity_scale=args.plane_population_diversity_scale,
        plane_threshold=args.plane_population_plane_threshold,
        boundary_threshold=args.plane_population_boundary_threshold,
        cardinality_margin=args.plane_population_cardinality_margin,
        gate_sharpness=args.plane_population_gate_sharpness,
        gate_penalty_scale=args.plane_population_gate_penalty_scale,
    )
    plane_population = PlanePopulationObjective(plane_population_config)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "geometric_diffuco_322.pt"
    summary_path = args.output_dir / "training_summary.json"

    history = []
    best_loss = float("inf")
    best_payload = None
    serializable_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, rollout_history = diffusion.rollout(
            model=model,
            energy=energy,
            point_features=points,
            edge_index=edge_index,
            edge_type=edge_type,
            batch_size=args.batch_size,
            stochastic=True,
        )
        rollout_history["geometric_log_p_0"] = rollout_history["log_p_0"]
        terminal_terms = build_orbit_terminal_terms(
            masks=rollout_history["final_state"].detach(),
            geometric_energy=rollout_history["final_energy"].detach(),
            energy=energy,
            tracker=orbit_tracker,
            reward_config=orbit_reward_config,
            terminal_temperature=args.terminal_temperature,
            terminal_config=orbit_terminal_config,
            reference=reference,
        )
        rollout_history["log_p_0"] = terminal_terms["log_p_0"]
        rollout_history["terminal_cost"] = terminal_terms["terminal_cost"]
        rollout_history["terminal_novelty_bonus"] = terminal_terms["novelty_bonus"]
        progress = epoch / max(args.epochs, 1)
        if progress <= args.memory_warmup_start:
            memory_weight = 0.0
        elif progress >= args.memory_warmup_end:
            memory_weight = args.memory_repulsion_weight
        else:
            ramp = (progress - args.memory_warmup_start) / max(args.memory_warmup_end - args.memory_warmup_start, 1e-6)
            memory_weight = args.memory_repulsion_weight * ramp
        if progress <= args.plane_memory_warmup_start:
            plane_memory_weight = 0.0
        elif progress >= args.plane_memory_warmup_end:
            plane_memory_weight = args.plane_memory_weight
        else:
            ramp = (progress - args.plane_memory_warmup_start) / max(args.plane_memory_warmup_end - args.plane_memory_warmup_start, 1e-6)
            plane_memory_weight = args.plane_memory_weight * ramp
        if progress <= args.plane_density_warmup_start:
            plane_density_weight = 0.0
        elif progress >= args.plane_density_warmup_end:
            plane_density_weight = args.plane_density_weight
        else:
            ramp = (progress - args.plane_density_warmup_start) / max(args.plane_density_warmup_end - args.plane_density_warmup_start, 1e-6)
            plane_density_weight = args.plane_density_weight * ramp
        if progress <= args.plane_set_coverage_warmup_start:
            plane_set_coverage_weight = 0.0
        elif progress >= args.plane_set_coverage_warmup_end:
            plane_set_coverage_weight = args.plane_set_coverage_weight
        else:
            ramp = (progress - args.plane_set_coverage_warmup_start) / max(args.plane_set_coverage_warmup_end - args.plane_set_coverage_warmup_start, 1e-6)
            plane_set_coverage_weight = args.plane_set_coverage_weight * ramp
        if progress <= args.plane_set_dpp_warmup_start:
            plane_set_dpp_weight = 0.0
        elif progress >= args.plane_set_dpp_warmup_end:
            plane_set_dpp_weight = args.plane_set_dpp_weight
        else:
            ramp = (progress - args.plane_set_dpp_warmup_start) / max(args.plane_set_dpp_warmup_end - args.plane_set_dpp_warmup_start, 1e-6)
            plane_set_dpp_weight = args.plane_set_dpp_weight * ramp
        if progress <= args.plane_population_warmup_start:
            plane_population_weight = 0.0
        elif progress >= args.plane_population_warmup_end:
            plane_population_weight = args.plane_population_weight
        else:
            ramp = (progress - args.plane_population_warmup_start) / max(args.plane_population_warmup_end - args.plane_population_warmup_start, 1e-6)
            plane_population_weight = args.plane_population_weight * ramp
        relaxed_memory_energy = orbit_memory.repulsion(rollout_history["final_probabilities"])
        relaxed_plane_terms = energy(rollout_history["final_probabilities"])
        relaxed_plane_memory_energy = plane_orbit_memory.repulsion(
            offset=relaxed_plane_terms["offset"],
            normal=relaxed_plane_terms["normal"],
        )
        plane_density_terms = plane_density_bank.novelty_bonus(
            offset=relaxed_plane_terms["offset"],
            normal=relaxed_plane_terms["normal"],
            plane=relaxed_plane_terms["plane"],
            boundary=relaxed_plane_terms["boundary"],
            cardinality=relaxed_plane_terms["cardinality"],
            min_cardinality=energy.config.min_cardinality,
        )
        relaxed_plane_density_bonus = plane_density_terms["bonus"]
        plane_set_coverage_terms = plane_set_coverage.coverage_bonus(
            offset=relaxed_plane_terms["offset"],
            normal=relaxed_plane_terms["normal"],
            plane=relaxed_plane_terms["plane"],
            boundary=relaxed_plane_terms["boundary"],
            cardinality=relaxed_plane_terms["cardinality"],
            min_cardinality=energy.config.min_cardinality,
        )
        relaxed_plane_set_coverage_bonus = plane_set_coverage_terms["bonus"]
        plane_set_dpp_terms = plane_set_dpp.dpp_bonus(
            offset=relaxed_plane_terms["offset"],
            normal=relaxed_plane_terms["normal"],
            plane=relaxed_plane_terms["plane"],
            boundary=relaxed_plane_terms["boundary"],
            cardinality=relaxed_plane_terms["cardinality"],
            min_cardinality=energy.config.min_cardinality,
        )
        relaxed_plane_set_dpp_bonus = plane_set_dpp_terms["bonus"]
        plane_population_terms = plane_population.objective_terms(
            offset=relaxed_plane_terms["offset"],
            normal=relaxed_plane_terms["normal"],
            plane=relaxed_plane_terms["plane"],
            boundary=relaxed_plane_terms["boundary"],
            cardinality=relaxed_plane_terms["cardinality"],
            total_energy=rollout_history["final_relaxed_energy"],
            min_cardinality=energy.config.min_cardinality,
        )
        total_relaxed_aux_energy = (
            memory_weight * relaxed_memory_energy
            + plane_memory_weight * relaxed_plane_memory_energy
            - plane_density_weight * relaxed_plane_density_bonus
            - plane_set_coverage_weight * relaxed_plane_set_coverage_bonus
            - plane_set_dpp_weight * relaxed_plane_set_dpp_bonus
        )
        rollout_history["final_relaxed_mask_memory_energy"] = relaxed_memory_energy
        rollout_history["final_relaxed_plane_memory_energy"] = relaxed_plane_memory_energy
        rollout_history["final_relaxed_plane_density_bonus"] = relaxed_plane_density_bonus
        rollout_history["final_relaxed_plane_density_gate"] = plane_density_terms["gate"]
        rollout_history["final_relaxed_plane_density_score"] = plane_density_terms["density"]
        rollout_history["final_relaxed_plane_set_coverage_bonus"] = relaxed_plane_set_coverage_bonus
        rollout_history["final_relaxed_plane_set_coverage_gate"] = plane_set_coverage_terms["gate"]
        rollout_history["final_relaxed_plane_set_coverage_similarity"] = plane_set_coverage_terms["max_similarity"]
        rollout_history["final_relaxed_plane_set_dpp_bonus"] = relaxed_plane_set_dpp_bonus
        rollout_history["final_relaxed_plane_set_dpp_gate"] = plane_set_dpp_terms["gate"]
        rollout_history["final_relaxed_plane_set_dpp_similarity"] = plane_set_dpp_terms["mean_similarity"]
        rollout_history["final_relaxed_plane_set_dpp_logdet"] = plane_set_dpp_terms["logdet"]
        rollout_history["final_relaxed_plane_set_dpp_gain"] = plane_set_dpp_terms["diversity_gain"]
        rollout_history["population_objective_term"] = plane_population_weight * plane_population_terms["objective"]
        rollout_history["population_elite_energy"] = plane_population_terms["elite_energy"]
        rollout_history["population_elite_diversity"] = plane_population_terms["elite_diversity"]
        rollout_history["population_gate"] = plane_population_terms["gate"]
        rollout_history["population_elite_mask"] = plane_population_terms["elite_mask"]
        rollout_history["final_relaxed_memory_energy"] = total_relaxed_aux_energy
        rollout_history["final_relaxed_total_energy"] = rollout_history["final_relaxed_energy"] + total_relaxed_aux_energy
        loss = diffusion.training_loss(rollout_history)
        analyzed_items = terminal_terms["items"]
        per_sample_items = terminal_terms["per_sample_items"]
        novelty_bonus_tensor = terminal_terms["novelty_bonus"]
        if not torch.isfinite(loss):
            epoch_summary = {"epoch": epoch, "status": "stopped_non_finite_loss"}
            history.append(epoch_summary)
            print(json.dumps(epoch_summary, ensure_ascii=False))
            break

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            optimizer.zero_grad(set_to_none=True)
            epoch_summary = {"epoch": epoch, "status": "stopped_non_finite_grad"}
            history.append(epoch_summary)
            print(json.dumps(epoch_summary, ensure_ascii=False))
            break
        optimizer.step()
        orbit_tracker.update(analyzed_items)
        orbit_memory.update(per_sample_items, rollout_history["final_state"])
        plane_orbit_memory.update(per_sample_items)
        plane_density_bank.update(per_sample_items)

        final_stats = rollout_history["steps"][-1]
        loss_terms = diffusion.loss_terms(rollout_history)
        trajectory_log_q = loss_terms["trajectory_log_q"]
        trajectory_log_p = loss_terms["trajectory_log_p"]
        matched_classes = sorted({class_id for item in analyzed_items for class_id in item.get("matched_classes", [])})
        canonical_hits = sorted({str(item["canonical_key"]) for item in analyzed_items if item.get("canonical_key") is not None})
        epoch_summary = {
            "epoch": epoch,
            "loss": float(loss.item()),
            "objective_mode": "relaxed" if args.reinforce_weight == 0.0 else "hybrid",
            "reinforce_weight": float(args.reinforce_weight),
            "orbit_reward_scale": float(args.orbit_reward_scale),
            "mean_orbit_terminal_cost": float(rollout_history["terminal_cost"].mean().item()),
            "mean_terminal_novelty_bonus": float(novelty_bonus_tensor.mean().item()) if novelty_bonus_tensor.numel() > 0 else 0.0,
            "num_orbit_rewarded": int((novelty_bonus_tensor != 0.0).sum().item()) if novelty_bonus_tensor.numel() > 0 else 0,
            "num_canonical_hits": len(canonical_hits),
            "num_matched_classes_seen_this_epoch": len(matched_classes),
            "matched_classes_seen_this_epoch": matched_classes,
            "mean_energy": float(rollout_history["final_energy"].mean().item()),
            "reversekl_entropy_term": float(loss_terms["entropy"].item()),
            "reversekl_noise_term": float(loss_terms["noise"].item()),
            "reversekl_energy_term": float(loss_terms["energy"].item()),
            "reversekl_memory_term": float(loss_terms["memory_energy"].item()),
            "reversekl_entropy_reinforce": float(loss_terms["entropy_reinforce"].item()),
            "reversekl_noise_reinforce": float(loss_terms["noise_reinforce"].item()),
            "reversekl_energy_reinforce": float(loss_terms["energy_reinforce"].item()),
            "mean_log_q_trajectory": float(trajectory_log_q.item()),
            "mean_log_p_trajectory": float(trajectory_log_p.item()),
            "mean_log_p_0": float(rollout_history["log_p_0"].mean().item()),
            "mean_geometric_log_p_0": float(rollout_history["geometric_log_p_0"].mean().item()),
            "mean_plane": float(final_stats["plane"].mean().item()),
            "mean_facet_dim": float(final_stats["facet_dim"].mean().item()),
            "mean_second_eigenvalue": float(final_stats["second_eigenvalue"].mean().item()),
            "mean_boundary": float(final_stats["boundary"].mean().item()),
            "mean_cardinality": float(final_stats["cardinality"].mean().item()),
            "memory_weight": float(memory_weight),
            "plane_memory_weight": float(plane_memory_weight),
            "plane_density_weight": float(plane_density_weight),
            "plane_set_coverage_weight": float(plane_set_coverage_weight),
            "plane_set_dpp_weight": float(plane_set_dpp_weight),
            "plane_population_weight": float(plane_population_weight),
            "mean_relaxed_memory_energy": float(relaxed_memory_energy.mean().item()),
            "mean_relaxed_plane_memory_energy": float(relaxed_plane_memory_energy.mean().item()),
            "mean_relaxed_plane_density_bonus": float(relaxed_plane_density_bonus.mean().item()),
            "mean_relaxed_plane_density_gate": float(plane_density_terms["gate"].mean().item()),
            "mean_relaxed_plane_density_score": float(plane_density_terms["density"].mean().item()),
            "mean_relaxed_plane_set_coverage_bonus": float(relaxed_plane_set_coverage_bonus.mean().item()),
            "mean_relaxed_plane_set_coverage_gate": float(plane_set_coverage_terms["gate"].mean().item()),
            "mean_relaxed_plane_set_coverage_similarity": float(plane_set_coverage_terms["max_similarity"].mean().item()),
            "mean_relaxed_plane_set_dpp_bonus": float(relaxed_plane_set_dpp_bonus.mean().item()),
            "mean_relaxed_plane_set_dpp_gate": float(plane_set_dpp_terms["gate"].mean().item()),
            "mean_relaxed_plane_set_dpp_similarity": float(plane_set_dpp_terms["mean_similarity"].mean().item()),
            "plane_set_dpp_logdet": float(plane_set_dpp_terms["logdet"].item()),
            "plane_set_dpp_gain": float(plane_set_dpp_terms["diversity_gain"].item()),
            "population_objective": float((plane_population_weight * plane_population_terms["objective"]).item()),
            "population_elite_energy": float(plane_population_terms["elite_energy"].item()),
            "population_elite_diversity": float(plane_population_terms["elite_diversity"].item()),
            "population_gate_mean": float(plane_population_terms["gate"].mean().item()),
            "population_elite_count": int(plane_population_terms["elite_mask"].sum().item()),
            "mean_relaxed_total_aux_energy": float(total_relaxed_aux_energy.mean().item()),
            "active_memory_classes": len(orbit_memory.active_classes()),
            "active_plane_memory_classes": len(plane_orbit_memory.active_classes()),
            "plane_density_bank_size": plane_density_bank.size(),
            "grad_norm": float(torch.as_tensor(grad_norm).item()),
            "seen_class_counter": dict(sorted(orbit_tracker.class_counts.items())),
            "seen_canonical_counter_size": len(orbit_tracker.canonical_counts),
        }

        if epoch_summary["loss"] < best_loss:
            best_loss = epoch_summary["loss"]
            best_payload = {
                "model_state": model.state_dict(),
                "model_config": {
                    "hidden_dim": args.hidden_dim,
                    "layers": args.layers,
                    "dropout": args.dropout,
                },
                "diffusion_config": diffusion_config.__dict__,
                "energy_config": energy_config.__dict__,
                "training_args": serializable_args,
                "best_epoch": epoch,
                "best_loss": best_loss,
            }
            torch.save(best_payload, checkpoint_path)

        should_sample = args.sample_every > 0 and (epoch % args.sample_every == 0 or epoch == args.epochs)
        if should_sample:
            model.eval()
            facets = sample_unique_facets(
                model=model,
                diffusion=diffusion,
                energy=energy,
                point_features=points,
                edge_index=edge_index,
                edge_type=edge_type,
                num_samples=max(args.batch_size * 2, 64),
                reference=reference,
            )
            epoch_summary["num_valid_facets"] = len(facets)
            epoch_summary["best_cardinality"] = max((facet["cardinality"] for facet in facets), default=0)
            epoch_summary["sample_matched_classes"] = sorted({class_id for facet in facets for class_id in facet.get("matched_classes", [])})
            epoch_summary["sample_canonical_count"] = len({facet.get("canonical_key") for facet in facets if facet.get("canonical_key") is not None})
            if facets:
                epoch_summary["sample_facet"] = facets[0]["indices"]

        history.append(epoch_summary)
        print(json.dumps(epoch_summary, ensure_ascii=False))

    if best_payload is not None:
        summary_path.write_text(
            json.dumps(
                {
                    "checkpoint": str(checkpoint_path),
                    "best_loss": best_loss,
                    "history": history,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
