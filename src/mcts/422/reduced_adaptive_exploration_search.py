from __future__ import annotations

"""Source-local probe cooldown and diverse observed entrances, isolated as v9."""

from collections import deque
from dataclasses import asdict, dataclass
from itertools import islice

from .reduced_fresh_geometry_search import ReducedFreshGeometryMCTS, ReducedFreshProbeOnlyConfig


ADAPTIVE_STATS = (
    "adaptive_probe_opportunities", "adaptive_probe_requests", "adaptive_probe_cooldown_skips",
    "adaptive_probe_empty_outcomes", "adaptive_probe_zero_exit_outcomes", "adaptive_probe_novel_outcomes",
    "adaptive_probe_late_novel_outcomes", "adaptive_probe_new_reduced_exits", "adaptive_probe_cooldown_resets",
    "diversity_fair_entries", "diversity_selected_entries", "diversity_first_starts",
    "diversity_candidates_scored", "diversity_episode_new_reduced_facets",
)


@dataclass(frozen=True)
class ReducedAdaptiveExplorationConfig(ReducedFreshProbeOnlyConfig):
    enable_adaptive_probe_cooldown: bool = True
    enable_diverse_continuation: bool = True
    probe_warmup_requests: int = 2
    probe_cooldown_max_interval: int = 8
    diverse_entry_candidate_window: int = 32
    diverse_entry_recent_count: int = 8

    def __post_init__(self):
        super().__post_init__()
        if self.probe_warmup_requests < 0 or self.probe_cooldown_max_interval < 1:
            raise ValueError("probe warmup must be nonnegative and cooldown interval positive")
        if self.diverse_entry_candidate_window < 2 or self.diverse_entry_recent_count < 1:
            raise ValueError("diverse entry window must be at least two and recent count positive")
        if self.enable_adaptive_probe_cooldown and not self.enable_fresh_geometry_probes:
            raise ValueError("adaptive cooldown requires fresh geometry probes")
        if self.enable_diverse_continuation and not self.enable_local_continuation:
            raise ValueError("diverse entrances require local continuation")
        if self.enable_frontier_anchor_schedule:
            raise ValueError("v9 entrance diversification cannot be combined with the v7 anchor scheduler")


@dataclass(frozen=True)
class ReducedAdaptiveProbeOnlyConfig(ReducedAdaptiveExplorationConfig):
    enable_diverse_continuation: bool = False


@dataclass(frozen=True)
class ReducedDiverseEntryOnlyConfig(ReducedAdaptiveExplorationConfig):
    enable_fresh_geometry_probes: bool = False
    enable_adaptive_probe_cooldown: bool = False


@dataclass
class ProbeCooldownState:
    opportunities: int = 0
    requests: int = 0
    zero_streak: int = 0
    next_probe_opportunity: int = 1
    empty_outcomes: int = 0
    zero_exit_outcomes: int = 0
    novel_outcomes: int = 0
    new_reduced_exits: int = 0


class ReducedAdaptiveExplorationMCTS(ReducedFreshGeometryMCTS):
    def __init__(self, problem, config: ReducedAdaptiveExplorationConfig, **kwargs):
        super().__init__(problem, config, **kwargs)
        self.probe_cooldowns = {}
        self.unstarted_anchor_queue = deque()
        self.recent_entry_words = deque(maxlen=config.diverse_entry_recent_count)
        self.diverse_entry_events = []
        self.entry_schedule_kind = "empty_root"

    def _probe_state(self, source):
        return self.probe_cooldowns.setdefault(source, ProbeCooldownState())

    def _settle_probe_outcome(self, event, *, novel=False, empty=False):
        source = int(event["source_word_hex"], 16)
        state = self._probe_state(source)
        outcome = event["adaptive_outcome"]
        if novel:
            if outcome != "novel_exit":
                state.novel_outcomes += 1
                self.stats["adaptive_probe_novel_outcomes"] += 1
                self.stats["adaptive_probe_late_novel_outcomes"] += int(outcome != "pending")
            state.new_reduced_exits += 1
            self.stats["adaptive_probe_new_reduced_exits"] += 1
            state.zero_streak = 0
            state.next_probe_opportunity = state.opportunities + 1
            event["adaptive_outcome"] = "novel_exit"
            self.stats["adaptive_probe_cooldown_resets"] += 1
        elif outcome == "pending":
            category = "empty" if empty else "zero_exit"
            setattr(state, f"{category}_outcomes", getattr(state, f"{category}_outcomes") + 1)
            self.stats[f"adaptive_probe_{category}_outcomes"] += 1
            state.zero_streak += int(state.requests > self.config.probe_warmup_requests)
            interval = min(self.config.probe_cooldown_max_interval, 2 ** min(state.zero_streak, 30))
            state.next_probe_opportunity = max(state.opportunities, event["adaptive_opportunity"]) + interval
            event["adaptive_outcome"] = "no_geometry" if empty else "zero_exit"
        event["adaptive_next_probe_opportunity"] = state.next_probe_opportunity
        event["adaptive_feedback_iteration"] = self.current_iteration

    def _probe_action(self, node, *, rollout):
        if not self.config.enable_adaptive_probe_cooldown:
            return super()._probe_action(node, rollout=rollout)
        source = node.state.support_word
        if source in self.probe_tickets:
            return super()._probe_action(node, rollout=rollout)
        state = self._probe_state(source)
        opportunity = state.opportunities + 1
        if opportunity < state.next_probe_opportunity:
            self.stats["adaptive_probe_cooldown_skips"] += 1
            self.probe_tickets[source] = None
            return None
        action = super()._probe_action(node, rollout=rollout)
        state.requests += 1
        self.stats["adaptive_probe_requests"] += 1
        event = self.probe_events[-1]
        event.update(adaptive_opportunity=opportunity, adaptive_outcome="pending",
                     adaptive_feedback_iteration=None,
                     adaptive_next_probe_opportunity=state.next_probe_opportunity)
        if action is None:
            self._settle_probe_outcome(event, empty=True)
        return action

    def _register_anchor(self, terminal, witness, new_class):
        new_anchor = terminal.valid and terminal.closure_word not in self.anchors
        super()._register_anchor(terminal, witness, new_class)
        if new_anchor:
            self.unstarted_anchor_queue.append(terminal.closure_word)

    def _diverse_candidates(self):
        queue = self.unstarted_anchor_queue
        while queue and self.anchors[queue[0]].starts:
            queue.popleft()
        while queue and self.anchors[queue[-1]].starts:
            queue.pop()
        window = self.config.diverse_entry_candidate_window
        older = list(islice(queue, window // 2))
        recent = list(islice(reversed(queue), window - len(older)))
        candidates = list(dict.fromkeys(word for word in older + recent if not self.anchors[word].starts))
        return candidates or list(islice(self.anchor_queue, window))

    def _entry_distance(self, word):
        return min(((word ^ other).bit_count() / (word | other).bit_count()
                    for other in self.recent_entry_words), default=1.0)

    def _choose_diverse_entry(self):
        candidates = self._diverse_candidates()
        self.stats["diversity_candidates_scored"] += len(candidates)
        word = max(candidates, key=lambda word: (
            int(not self.anchors[word].starts), self._entry_distance(word),
            -self.anchors[word].starts, -self.source_removals[word],
            -self.anchors[word].observed_iteration, -word,
        ))
        return word, len(candidates)

    def _iteration_start(self, iteration):
        self.entry_schedule_kind = "empty_root"
        event = None
        if (self.config.enable_diverse_continuation and self.config.enable_local_continuation
                and self.anchor_queue and iteration % self.config.continuation_interval == 0):
            diverse = self.stats["continuation_episodes"] % 2 == 1
            word, count = self._choose_diverse_entry() if diverse else (self.anchor_queue[0], 0)
            self.entry_schedule_kind = "diverse" if diverse else "fair"
            anchor = self.anchors[word]
            event = {"iteration": iteration, "kind": self.entry_schedule_kind,
                     "source_word_hex": f"0x{word:x}", "starts_before": anchor.starts,
                     "actual_source_removals": self.source_removals[word], "candidate_count": count,
                     "minimum_support_distance": self._entry_distance(word)}
            self.stats["diversity_selected_entries" if diverse else "diversity_fair_entries"] += 1
            self.stats["diversity_first_starts"] += int(not anchor.starts)
            if diverse:
                self.anchor_queue.remove(word)
                self.anchor_queue.appendleft(word)
        node = super()._iteration_start(iteration)
        if event is not None:
            self.recent_entry_words.append(self.episode_origin.word)
            self.diverse_entry_events.append(event)
        return node

    def _step_reward(self, node, edge, path_actions, path_terminals, tree_edge):
        if (self.config.enable_adaptive_probe_cooldown and edge.action.kind == "remove"
                and self._slot() == "fresh_geometry"):
            self._probe_state(node.state.support_word).opportunities += 1
            self.stats["adaptive_probe_opportunities"] += 1
        new_reduced = bool(edge.terminal and (self.problem.node_id, edge.terminal.closure_word)
                           not in self.shared_discovery.reduced_keys)
        event = None
        if self.config.enable_adaptive_probe_cooldown and edge.terminal and node.state.source_facet_word:
            removal = next((action for action in reversed(path_actions) if action.kind == "remove"), None)
            if removal is not None and removal.sampled_facet_word == node.state.source_facet_word:
                event = self.probe_face_events.get((removal.sampled_facet_word, removal.retained_word))
        outcome = super()._step_reward(node, edge, path_actions, path_terminals, tree_edge)
        if event is not None:
            self._settle_probe_outcome(event, novel=new_reduced)
        if new_reduced and self.entry_schedule_kind == "diverse":
            self.stats["diversity_episode_new_reduced_facets"] += 1
        return outcome

    def _diagnostic_snapshot(self):
        snapshot = super()._diagnostic_snapshot()
        snapshot.update({key: self.stats[key] for key in ADAPTIVE_STATS})
        return snapshot

    def result(self, *, target_ids=()):
        result = super().result(target_ids=target_ids)
        result["algorithm"] = "bell422_reduced_adaptive_probe_diverse_entrances_v9"
        if self.config.enable_diverse_continuation:
            result["continuation"]["selection"] = "half FIFO fair, half bounded diverse unstarted observed entrances"
        result["adaptive_exploration"] = {
            "cooldown_enabled": self.config.enable_adaptive_probe_cooldown,
            "diverse_continuation_enabled": self.config.enable_diverse_continuation,
            "warmup_requests_per_source": self.config.probe_warmup_requests,
            "cooldown_max_interval": self.config.probe_cooldown_max_interval,
            "cooldown_clock": "actual fresh-slot source removals, not queries or global iterations",
            "probe_feedback": "no new geometry, or actual new reduced exit; never lift/class labels",
            "probe_queries_are_discoveries": False,
            "pending_feedback_stays_pending": True,
            "late_novel_exit_resets_cooldown": True,
            "periodic_escape_probes": True,
            "entrance_cycle": ["fair", "diverse"],
            "candidate_window": self.config.diverse_entry_candidate_window,
            "recent_entry_count": self.config.diverse_entry_recent_count,
            "entrance_candidates": "oldest/recent windows of observed unstarted anchors; FIFO fallback",
            "entrance_distance": "minimum support Jaccard distance to recent actual continuation entrances",
            "selection_uses_lifting_or_catalog_labels": False,
            "reward_rollout_entrance_fraction_correction_budgets_unchanged": True,
            "unstarted_anchor_count": sum(not anchor.starts for anchor in self.anchors.values()),
            "statistics": {key: self.stats[key] for key in ADAPTIVE_STATS},
            "sources": [{"word_hex": f"0x{word:x}", **asdict(state)}
                        for word, state in sorted(self.probe_cooldowns.items())],
            "entrance_events": self.diverse_entry_events,
        }
        return result
