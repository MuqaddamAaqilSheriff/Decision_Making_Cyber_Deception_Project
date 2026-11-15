from __future__ import annotations

from contextlib import contextmanager
import random
from typing import Any, Iterable, Sequence

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:
    raise ImportError(
        "Gymnasium is required. Install with: pip install gymnasium numpy networkx"
    ) from exc

from Cyber_Deception_Attackers_and_Network import (
    ATTACKER_VARIANTS,
    DEFAULT_LAMBDA,
    DEFAULT_MAX_GAME_STEPS,
    MAX_OUTCOMES_PER_NODE,
    N_LAYERS,
    NODES_PER_LAYER,
    Attacker,
    Graph,
    build_fixed_environment,
)


class CyberDeceptionEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(
        self,
        attacker_variants: Sequence[str] = (
            "random",
            "rational",
            "weighting_loss_aversion_investment",
        ),
        lih_budget: int = 2,
        max_steps: int = DEFAULT_MAX_GAME_STEPS,
        network_seed: int | None = 1000,
        loss_aversion_lambda: float = DEFAULT_LAMBDA,
        controlled_investment: float | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.attacker_variants = tuple(attacker_variants)
        if not self.attacker_variants:
            raise ValueError("attacker_variants must contain at least one attacker type")
        unknown = set(self.attacker_variants) - set(ATTACKER_VARIANTS)
        if unknown:
            raise ValueError(f"Unknown attacker variants: {sorted(unknown)}")
        if not (0 <= lih_budget <= NODES_PER_LAYER):
            raise ValueError(f"lih_budget must be between 0 and {NODES_PER_LAYER}")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        if loss_aversion_lambda < 0:
            raise ValueError("loss_aversion_lambda must be nonnegative")
        if controlled_investment is not None and not 0 <= controlled_investment <= 1:
            raise ValueError("controlled_investment must be within [0, 1]")
        if render_mode not in (None, "ansi"):
            raise ValueError("Only render_mode=None or 'ansi' is supported")

        self.lih_budget = int(lih_budget)
        self.max_steps = int(max_steps)
        self.network_seed = network_seed
        self.loss_aversion_lambda = float(loss_aversion_lambda)
        self.controlled_investment = controlled_investment
        self.render_mode = render_mode
        self.locations = tuple(
            (layer, path_id)
            for layer in range(1, N_LAYERS)
            for path_id in range(1, NODES_PER_LAYER + 1)
        )
        self.all_nodes = tuple(
            (layer, path_id)
            for layer in range(1, N_LAYERS + 1)
            for path_id in range(1, NODES_PER_LAYER + 1)
        )
        self.action_to_location = {i: loc for i, loc in enumerate(self.locations)}
        self.location_to_action = {loc: i for i, loc in enumerate(self.locations)}
        self.wait_action = len(self.locations)
        self.action_space = spaces.Discrete(len(self.locations) + 1)

        self.graph: Graph | None = None
        self.attacker: Attacker | None = None
        self.environment_data: dict[str, Any] | None = None
        self.current_attacker_variant: str | None = None
        self.remaining_budget = self.lih_budget
        self.placed_lihs = 0
        self.episode_finished = False
        self._episode_rng = random.Random()
        self._last_wasted_lihs: tuple[tuple[int, int], ...] = ()

        sample_seed = 0 if network_seed is None else int(network_seed)
        with self._attacker_random_context():
            environment_data = build_fixed_environment(sample_seed)
            example_graph = Graph(None, environment=environment_data)
            example_attacker = Attacker(None, example_graph, variant=self.attacker_variants[0])
            example_observation = self._observation_from(
                example_graph, example_attacker, environment_data
            )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(example_observation.size,),
            dtype=np.float32,
        )

    @contextmanager
    def _attacker_random_context(self):
        saved_state = random.getstate()
        random.setstate(self._episode_rng.getstate())
        try:
            yield
        finally:
            self._episode_rng.setstate(random.getstate())
            random.setstate(saved_state)

    @staticmethod
    def _padded_outcomes(outcomes: Iterable[float]) -> list[float]:
        values = [float(value) for value in outcomes]
        return values[:MAX_OUTCOMES_PER_NODE] + [0.0] * max(
            0, MAX_OUTCOMES_PER_NODE - len(values)
        )

    def _observation_from(
        self,
        graph: Graph,
        attacker: Attacker,
        environment_data: dict[str, Any],
    ) -> np.ndarray:
        current = graph.current_node
        in_hih_node = (
            isinstance(current, tuple)
            and len(current) == 2
            and current[0] == "H"
        )
        current_layer = 0 if in_hih_node else int(current[0])
        current_path = 0 if in_hih_node else int(current[1])
        hih_depth = int(current[1]) if in_hih_node else 0
        entry_layer = graph.honeynet_entry_layer or 0
        entry_path = graph.honeynet_path_id or 0
        moves_used = attacker.explorationCounter
        features: list[float] = [
            current_layer / N_LAYERS,
            current_path / NODES_PER_LAYER,
            hih_depth / 5.0,
            float(graph.in_honeynet),
            entry_layer / N_LAYERS,
            entry_path / NODES_PER_LAYER,
            moves_used / self.max_steps,
            (self.max_steps - moves_used) / self.max_steps,
            self.remaining_budget / max(1, self.lih_budget),
            self.placed_lihs / max(1, self.lih_budget),
            float(attacker.path_investment),
            float(attacker.honeynet_investment),
            float(attacker.total_investment_paid),
            float(attacker.total_realized_outcome),
            float(attacker.entered_honeynet),
            float(attacker.loss_aversion_lambda),
            (
                -1.0
                if attacker.controlled_investment is None
                else float(attacker.controlled_investment)
            ),
        ]
        features.extend(
            float(attacker.variant == variant) for variant in ATTACKER_VARIANTS
        )

        for loc in self.all_nodes:
            node = graph.g.nodes[loc]
            outcomes = list(node["outcomes"])
            features.extend(self._padded_outcomes(outcomes))
            features.extend(
                (
                    len(outcomes) / MAX_OUTCOMES_PER_NODE,
                    float(node["investment"]),
                    float(node.get("known", False)),
                    float(loc in graph.active_lihs),
                    float(loc in graph.pending_lihs),
                    float(loc in graph.triggered_lih_locations),
                    float(attacker.node_beliefs[loc]),
                )
            )

        for loc in self.locations:
            spec = environment_data["low_honeypots"][loc]
            outcomes = list(spec["outcomes"])
            features.extend(self._padded_outcomes(outcomes))
            features.extend(
                (len(outcomes) / MAX_OUTCOMES_PER_NODE, float(spec["investment"]))
            )

        for depth in range(1, 6):
            loc = ("H", depth)
            spec = environment_data["hih_nodes"][depth]
            outcomes = list(spec["outcomes"])
            features.extend(self._padded_outcomes(outcomes))
            features.extend(
                (
                    len(outcomes) / MAX_OUTCOMES_PER_NODE,
                    float(spec["investment"]),
                    float(loc in graph.g and graph.g.nodes[loc].get("known", False)),
                )
            )

        return np.asarray(features, dtype=np.float32)

    def _observation(self) -> np.ndarray:
        assert self.graph is not None and self.attacker is not None
        assert self.environment_data is not None
        return self._observation_from(
            self.graph, self.attacker, self.environment_data
        )

    def get_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space.n, dtype=np.int8)
        mask[self.wait_action] = 1
        if self.graph is None or self.episode_finished or self.remaining_budget <= 0:
            return mask
        valid = set(self.graph.eligible_lih_locations())
        for i, location in self.action_to_location.items():
            if location in valid:
                mask[i] = 1
        return mask

    def _info(self) -> dict[str, Any]:
        assert self.graph is not None and self.attacker is not None
        return {
            "action_mask": self.get_action_mask(),
            "attacker_variant": self.current_attacker_variant,
            "attacker_location": self.graph.current_node,
            "in_honeynet": bool(self.graph.in_honeynet),
            "entered_honeynet": bool(self.attacker.entered_honeynet),
            "ended_in_honeynet": bool(
                self.graph.in_honeynet and self.attacker.entered_honeynet
            ),
            "active_lihs": tuple(sorted(self.graph.active_lihs)),
            "pending_lihs": tuple(sorted(self.graph.pending_lihs)),
            "wasted_lihs_this_step": self._last_wasted_lihs,
            "placed_lihs": self.placed_lihs,
            "remaining_budget": self.remaining_budget,
            "moves_used": self.attacker.explorationCounter,
            "hih_depth": self.attacker.max_hih_depth_reached,
            "termination_reason": self.attacker.termination_reason,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = {} if options is None else dict(options)
        selected_variant = options.get("attacker_variant")
        if selected_variant is None:
            index = int(self.np_random.integers(len(self.attacker_variants)))
            selected_variant = self.attacker_variants[index]
        if selected_variant not in ATTACKER_VARIANTS:
            raise ValueError(f"Unknown attacker variant: {selected_variant}")

        selected_network_seed = options.get("network_seed", self.network_seed)
        if selected_network_seed is None:
            selected_network_seed = int(self.np_random.integers(0, 2**32))
        episode_seed = int(self.np_random.integers(0, 2**32))
        self._episode_rng.seed(episode_seed)

        self.current_attacker_variant = str(selected_variant)
        self.remaining_budget = self.lih_budget
        self.placed_lihs = 0
        self.episode_finished = False
        self._last_wasted_lihs = ()
        with self._attacker_random_context():
            self.environment_data = build_fixed_environment(int(selected_network_seed))
            self.graph = Graph(None, environment=self.environment_data)
            self.attacker = Attacker(None, self.graph, variant=selected_variant,
                loss_aversion_lambda=self.loss_aversion_lambda,
                controlled_investment=self.controlled_investment)
        return self._observation(), self._info()

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.graph is None or self.attacker is None:
            raise RuntimeError("Call reset() before step()")
        if self.episode_finished:
            raise RuntimeError("Episode has finished. Call reset() before another step()")
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action!r} is outside the action space")
        action = int(action)
        mask = self.get_action_mask()
        if not mask[action]:
            raise ValueError(f"Illegal LIH placement action {action}; use info['action_mask']")

        if action != self.wait_action:
            location = self.action_to_location[action]
            self.graph.schedule_lih(location)
            self.remaining_budget -= 1
            self.placed_lihs += 1

        staged_lihs = dict(self.graph.pending_lihs)
        self.graph.pending_lihs.clear()
        prior_moves = self.attacker.explorationCounter
        with self._attacker_random_context():
            self.attacker.makeDecision()
        made_move = self.attacker.explorationCounter > prior_moves

        wasted: list[tuple[int, int]] = []
        if made_move:
            for location, spec in staged_lihs.items():
                if self.graph.g.nodes[location].get("known", False):
                    wasted.append(location)
                else:
                    self.graph.pending_lihs[location] = spec
            self.graph.activate_scheduled_lihs()
        self._last_wasted_lihs = tuple(sorted(wasted))

        terminated = bool(self.attacker.completed_game)
        truncated = False
        if not terminated and self.attacker.explorationCounter >= self.max_steps:
            self.attacker.completed_game = True
            self.attacker.termination_reason = "horizon"
            terminated = True
        if terminated or truncated:
            self.episode_finished = True

        ended_in_honeynet = bool(
            self.graph.in_honeynet and self.attacker.entered_honeynet
        )
        reward = float(ended_in_honeynet) if self.episode_finished else 0.0
        return self._observation(), reward, terminated, truncated, self._info()

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None
        if self.graph is None or self.attacker is None:
            return "Environment not initialized"
        return (
            f"step={self.attacker.explorationCounter}/{self.max_steps}, "
            f"attacker={self.current_attacker_variant}, "
            f"node={self.graph.current_node}, "
            f"in_honeynet={self.graph.in_honeynet}, "
            f"active_lihs={sorted(self.graph.active_lihs)}, "
            f"budget={self.remaining_budget}"
        )

    def close(self) -> None:
        self.graph = None
        self.attacker = None
        self.environment_data = None


if __name__ == "__main__":
    env = CyberDeceptionEnv(render_mode="ansi")
    obs, info = env.reset(seed=42)
    print("Observation shape:", obs.shape)
    print("Actions:", env.action_space.n, "(wait =", env.wait_action, ")")
    for _ in range(env.max_steps):
        mask = info["action_mask"]
        valid = np.flatnonzero(mask)
        action = int(env.np_random.choice(valid))
        obs, reward, terminated, truncated, info = env.step(action)
        print(env.render(), "reward=", reward)
        if terminated or truncated:
            break
    print("Final reward:", reward)
    env.close()