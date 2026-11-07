from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from datetime import datetime
from typing import Any

import numpy as np

try:
    import networkx as nx
    from networkx import DiGraph as NxGraph, single_source_shortest_path_length
except ImportError as exc:
    raise SystemExit("Missing dependency: networkx. Install with: pip install numpy networkx") from exc

try:
    from pyvis.network import Network
except ImportError:
    Network = None

N_LAYERS = 5
NODES_PER_LAYER = 4


POSITIVE_OUTCOMES = [0.25, 0.50, 0.75]
NEGATIVE_OUTCOMES = [-0.25, -0.50, -0.75]
ALL_OUTCOMES = NEGATIVE_OUTCOMES + POSITIVE_OUTCOMES


MIN_OUTCOMES_PER_NODE = 2
MAX_OUTCOMES_PER_NODE = len(ALL_OUTCOMES)


INITIAL_BELIEF_MIN = 0.0
INITIAL_BELIEF_MAX = 1.0

DEFAULT_LAMBDA = 2.0
DEFAULT_GAMMA = 0.61
DEFAULT_DELTA = 0.69


LIH_DEPLOY_COST = 0.25
HIH_DEPLOY_COST = 0.75


DEFAULT_MAX_GAME_STEPS = 20

Location = tuple[int, int]
FixedEnvironment = dict[str, Any]


SIGNAL_LIKELIHOODS = {
    "real": [(0.45, 0.55), (0.45, 0.55)],
    "low":  [(0.80, 0.20), (0.79, 0.21)],
    "high": [(0.92, 0.08), (0.90, 0.10)],
}


ATTACKER_VARIANTS = {
    "random": {
        "label": "Random",
        "weighting": False,
        "loss_aversion": False,
        "investment": False,
    },
    "rational": {
        "label": "Rational",
        "weighting": False,
        "loss_aversion": False,
        "investment": False,
    },
    "weighting": {
        "label": "Weighting",
        "weighting": True,
        "loss_aversion": False,
        "investment": False,
    },
    "loss_aversion": {
        "label": "Loss Aversion",
        "weighting": False,
        "loss_aversion": True,
        "investment": False,
    },
    "weighting_loss_aversion": {
        "label": "Weighting + Loss Aversion",
        "weighting": True,
        "loss_aversion": True,
        "investment": False,
    },
    "investment": {
        "label": "Investment",
        "weighting": False,
        "loss_aversion": False,
        "investment": True,
    },
    "weighting_investment": {
        "label": "Weighting + Investment",
        "weighting": True,
        "loss_aversion": False,
        "investment": True,
    },
    "loss_aversion_investment": {
        "label": "Loss Aversion + Investment",
        "weighting": False,
        "loss_aversion": True,
        "investment": True,
    },
    "weighting_loss_aversion_investment": {
        "label": "Weighting + Loss Aversion + Investment",
        "weighting": True,
        "loss_aversion": True,
        "investment": True,
    },
}


VIS_OPTIONS_JSON = r"""
{
  "physics": { "enabled": false },
  "layout": {
    "hierarchical": {
      "enabled": true,
      "direction": "UD",
      "sortMethod": "directed",
      "levelSeparation": 190,
      "nodeSpacing": 260,
      "treeSpacing": 320
    }
  },
  "nodes": {
    "shape": "box",
    "margin": 12,
    "borderWidth": 2,
    "widthConstraint": { "minimum": 170, "maximum": 240 },
    "font": { "size": 13, "face": "Arial", "color": "#1f2937" }
  },
  "edges": {
    "width": 0.8,
    "color": { "color": "#cbd5e1", "opacity": 0.24 },
    "arrows": { "to": { "enabled": true, "scaleFactor": 0.55 } },
    "smooth": {
      "enabled": true,
      "type": "cubicBezier",
      "forceDirection": "vertical",
      "roundness": 0.28
    }
  }
}
"""


class LFMLog:
    def __init__(self, name: str):
        self.base_log = logging.getLogger(name)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_log, name)

    def close_logger(self) -> None:
        for handler in list(self.handlers):
            self.removeHandler(handler)
            handler.close()


class LogFileMaker:
    global_lf_lev = logging.DEBUG
    global_c_lev = logging.INFO

    def __init__(self, logfile_level: str, console_level: str):
        self.set_globals(logfile_level, console_level)

    @classmethod
    def set_globals(cls, logfile_level: str, console_level: str) -> None:
        levels = {"debug": logging.DEBUG, "info": logging.INFO, "none": None}
        cls.global_lf_lev = levels.get(logfile_level)
        cls.global_c_lev = levels.get(console_level)

    @classmethod
    def create_logger(
        cls,
        logfile_name: str,
        logfile_level: str = "global",
        console_level: str = "global",
    ) -> LFMLog:
        levels = {"debug": logging.DEBUG, "info": logging.INFO, "none": None}
        lf = cls.global_lf_lev if logfile_level == "global" else levels.get(logfile_level)
        con = cls.global_c_lev if console_level == "global" else levels.get(console_level)

        logger = LFMLog(logfile_name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        for h in list(logger.handlers):
            logger.removeHandler(h)
            h.close()

        if lf is not None:
            os.makedirs(os.path.dirname(logfile_name) or ".", exist_ok=True)
            h = logging.FileHandler(logfile_name, "a+", "utf-8")
            h.setLevel(lf)
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
            logger.addHandler(h)

        if con is not None:
            h = logging.StreamHandler()
            h.setLevel(con)
            h.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(h)

        return logger


LFMGen = LogFileMaker


def validate_unit_interval(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1.")
    return value


def continuous_investment(minimum: float = 0.0, maximum: float = 1.0) -> float:
    return round(random.uniform(minimum, maximum), 4)


def conditional_outcome_probabilities(
    outcomes: list[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:


    values = np.asarray(outcomes, dtype=float)

    if values.size < 2:
        raise ValueError("Each node must have at least two outcomes.")

    order = np.argsort(values)
    n = values.size


    rank_weights = np.array(
        [rank * (rank + 1) / 2 for rank in range(1, n + 1)],
        dtype=float,
    )
    rank_weights /= rank_weights.sum()

    p_r = np.empty(n, dtype=float)
    p_h = np.empty(n, dtype=float)


    p_r[order] = rank_weights
    p_h[order] = rank_weights[::-1]

    return p_h, p_r


def overall_prob(
    belief_h: float,
    outcomes: list[float] | np.ndarray,
) -> np.ndarray:
    belief_h = validate_unit_interval(belief_h, "belief_h")
    p_h, p_r = conditional_outcome_probabilities(outcomes)
    return belief_h * p_h + (1.0 - belief_h) * p_r


def w_plus(p: float, gamma: float = DEFAULT_GAMMA) -> float:
    return p ** gamma / ((p ** gamma + (1.0 - p) ** gamma) ** (1.0 / gamma))


def w_minus(p: float, delta: float = DEFAULT_DELTA) -> float:
    return p ** delta / ((p ** delta + (1.0 - p) ** delta) ** (1.0 / delta))


def pt_value(
    outcomes: list[float] | np.ndarray,
    probabilities: list[float] | np.ndarray,
    weighting: bool = False,
    loss_aversion: bool = False,
    lambda_loss: float = DEFAULT_LAMBDA,
) -> float:
    lam = float(lambda_loss) if loss_aversion else 1.0
    value = 0.0

    for x, p in zip(outcomes, probabilities):
        x = float(x)

        if x >= 0:
            weight = np.floor(w_plus(float(p)) * 100) / 100 if weighting else float(p)
            value += weight * x
        else:
            weight = np.floor(w_minus(float(p)) * 100) / 100 if weighting else float(p)
            value += weight * lam * x

    return float(value)


def bayesian_update(
    prior_h: float,
    signal_likelihoods: list[tuple[float, float]],
) -> float:
    prior_h = validate_unit_interval(prior_h, "prior_h")
    lh = 1.0
    lr = 1.0

    for p_signal_h, p_signal_r in signal_likelihoods:
        lh *= float(p_signal_h)
        lr *= float(p_signal_r)

    numerator = prior_h * lh
    denominator = numerator + (1.0 - prior_h) * lr

    return prior_h if denominator == 0 else float(numerator / denominator)


def make_random_outcomes() -> list[float]:


    n_outcomes = random.randint(
        MIN_OUTCOMES_PER_NODE,
        MAX_OUTCOMES_PER_NODE,
    )

    return random.sample(
        ALL_OUTCOMES,
        k=n_outcomes,
    )


def _make_random_outcomes_with_rng(rng: random.Random) -> list[float]:
    n_outcomes = rng.randint(MIN_OUTCOMES_PER_NODE, MAX_OUTCOMES_PER_NODE)
    return rng.sample(ALL_OUTCOMES, k=n_outcomes)


def build_fixed_environment(seed: int) -> FixedEnvironment:

    rng = random.Random(seed)

    real_assets: dict[Location, dict[str, Any]] = {}
    low_honeypots: dict[Location, dict[str, Any]] = {}
    hih_nodes: dict[int, dict[str, Any]] = {}

    for layer in range(1, N_LAYERS + 1):
        for node_index in range(1, NODES_PER_LAYER + 1):
            loc = (layer, node_index)
            real_assets[loc] = {
                "Asset": f"Real Node L{layer}N{node_index}",
                "Type": "Real",
                "interaction": "real",
                "layer": layer,
                "outcomes": _make_random_outcomes_with_rng(rng),
                "investment": round(rng.uniform(0.0, 1.0), 4),
                "deploy_cost": 0.0,
            }

            if layer < N_LAYERS:
                low_honeypots[loc] = {
                    "Asset": "Low Interaction Honeypot",
                    "Type": "Fake",
                    "interaction": "low",
                    "layer": layer,
                    "outcomes": _make_random_outcomes_with_rng(rng),
                    "investment": round(rng.uniform(0.0, 0.50), 4),
                    "deploy_cost": LIH_DEPLOY_COST,
                }

    for step in range(1, 6):
        hih_nodes[step] = {
            "Asset": "High Interaction Honeypot",
            "Type": "Fake",
            "interaction": "high",
            "layer": step,
            "outcomes": _make_random_outcomes_with_rng(rng),
            "investment": round(rng.uniform(0.50, 1.0), 4),
            "deploy_cost": HIH_DEPLOY_COST,
        }

    return {
        "real_assets": real_assets,
        "low_honeypots": low_honeypots,
        "hih_nodes": hih_nodes,
    }


def _copy_asset_spec(spec: dict[str, Any]) -> dict[str, Any]:
    copied = dict(spec)
    copied["outcomes"] = list(spec["outcomes"])
    return copied


def realized_outcome_for_node(outcomes: list[float], node_type: str) -> float:
    p_h, p_r = conditional_outcome_probabilities(outcomes)
    probs = p_h if node_type == "Fake" else p_r

    return float(
        random.choices(
            outcomes,
            weights=probs,
            k=1,
        )[0]
    )


def all_network_locations(
    n_layers: int = N_LAYERS,
    nodes_per_layer: int = NODES_PER_LAYER,
) -> list[Location]:
    return [(layer, idx)
            for layer in range(1, n_layers + 1)
            for idx in range(1, nodes_per_layer + 1)]


class Graph:
    def __init__(
        self,
        global_log: LFMLog | None,
        n_layers: int = N_LAYERS,
        nodes_per_layer: int = NODES_PER_LAYER,
        environment: FixedEnvironment | None = None,
    ):
        self.globalLog = global_log
        self.environment = environment

        self.active_lihs: set[Location] = set()
        self.pending_lihs: dict[Location, dict[str, Any]] = {}
        self.hih_honeynet_deployed = False
        self.g = nx.DiGraph()
        self.n_layers = n_layers
        self.nodes_per_layer = nodes_per_layer
        self.current_layer = 0
        self.current_node: Any = (0, 0)

        self.in_honeynet = False
        self.honeynet_nodes: dict[tuple[str, int], dict[str, Any]] = {}
        self.honeynet_path_id: int | None = None
        self.honeynet_entry_layer: int | None = None
        self.honeynet_entry_node: Location | None = None


        self.triggered_lih_locations: set[Location] = set()

        self._build_graph()

    def eligible_lih_locations(self) -> list[Location]:


        used_paths = {idx for _layer, idx in self.active_lihs | self.pending_lihs.keys()}
        return [
            (layer, idx)
            for layer in range(1, self.n_layers)
            for idx in range(1, self.nodes_per_layer + 1)
            if idx not in used_paths
            and (layer, idx) != self.current_node
            and not self.g.nodes[(layer, idx)].get("known", False)
        ]

    def _lih_spec(self, location: Location) -> dict[str, Any]:
        if self.environment is not None:
            return _copy_asset_spec(self.environment["low_honeypots"][location])
        return {
            "Asset": "Low Interaction Honeypot",
            "Type": "Fake",
            "interaction": "low",
            "layer": location[0],
            "outcomes": make_random_outcomes(),
            "investment": continuous_investment(0.0, 0.50),
            "deploy_cost": LIH_DEPLOY_COST,
        }

    def schedule_lih(self, location: Location) -> None:


        location = tuple(location)
        if location not in self.eligible_lih_locations():
            raise ValueError(f"Ineligible LIH location: {location}")
        self.pending_lihs[location] = self._lih_spec(location)

    def install_initial_lih(self, location: Location) -> None:


        location = tuple(location)
        if location not in self.eligible_lih_locations():
            raise ValueError(f"Ineligible initial LIH location: {location}")
        self._activate_lih(location, self._lih_spec(location))

    def _activate_lih(self, location: Location, spec: dict[str, Any]) -> None:

        node = self.g.nodes[location]
        node.update(
            asset=spec["Asset"],
            type=spec["Type"],
            interaction=spec["interaction"],
            outcomes=list(spec["outcomes"]),
            investment=float(spec["investment"]),
            deploy_cost=float(spec["deploy_cost"]),
        )
        for u in list(self.g.predecessors(location)):
            edge = self.g[u][location]
            edge["outcomes"] = list(spec["outcomes"])
            edge["investment"] = float(spec["investment"])
        self.active_lihs.add(location)

    def activate_scheduled_lihs(self) -> None:


        for location, spec in list(self.pending_lihs.items()):
            self._activate_lih(location, spec)
        self.pending_lihs.clear()

    def _build_graph(self) -> None:
        self.g.add_node(
            (0, 0),
            known=True,
            depth=0,
            layer=0,
            asset="Entry Point",
            type="Entry",
            interaction="entry",
            outcomes=[0.0, 0.0, 0.0],
            investment=0.0,
            deploy_cost=0.0,
            belief_before=None,
            belief_after=None,
            signal_type=None,
        )

        for layer in range(1, self.n_layers + 1):
            for idx in range(1, self.nodes_per_layer + 1):
                if self.environment is not None:
                    asset = _copy_asset_spec(self.environment["real_assets"][(layer, idx)])
                else:
                    asset = {
                        "Asset": f"Real Node L{layer}N{idx}",
                        "Type": "Real",
                        "interaction": "real",
                        "layer": layer,
                        "outcomes": make_random_outcomes(),
                        "investment": continuous_investment(),
                        "deploy_cost": 0.0,
                    }

                self.g.add_node(
                    (layer, idx),
                    known=False,
                    depth=layer,
                    layer=layer,
                    asset=asset["Asset"],
                    type=asset["Type"],
                    interaction=asset["interaction"],
                    outcomes=list(asset["outcomes"]),
                    investment=float(asset["investment"]),
                    deploy_cost=float(asset["deploy_cost"]),
                    belief_before=None,
                    belief_after=None,
                    signal_type=None,
                )

        for idx in range(1, self.nodes_per_layer + 1):
            self._add_edge((0, 0), (1, idx))

        for layer in range(1, self.n_layers):
            for src_idx in range(1, self.nodes_per_layer + 1):
                for dst_idx in range(1, self.nodes_per_layer + 1):
                    self._add_edge(
                        (layer, src_idx),
                        (layer + 1, dst_idx),
                    )

    def _add_edge(self, u: tuple[int, int], v: tuple[int, int]) -> None:
        data = self.g.nodes[v]
        self.g.add_edge(
            u,
            v,
            outcomes=list(data["outcomes"]),
            investment=float(data["investment"]),
        )

    def enter_honeynet(self, path_id: int, entry_layer: int, entry_node: Location) -> None:

        self.in_honeynet = True
        self.honeynet_path_id = int(path_id)
        self.honeynet_entry_layer = int(entry_layer)
        self.honeynet_entry_node = entry_node
        self.honeynet_nodes = {}
        self.hih_honeynet_deployed = True

        previous: Any = entry_node
        for step in range(1, 6):
            node_id = ("H", step)
            if self.environment is not None:
                data = _copy_asset_spec(self.environment["hih_nodes"][step])
            else:
                data = {
                    "outcomes": make_random_outcomes(),
                    "investment": continuous_investment(0.50, 1.0),
                }
            self.honeynet_nodes[node_id] = {
                "known": False,
                "depth": step,
                "layer": step,
                "asset": f"HIH Honeynet H{step}",
                "type": "Fake",
                "interaction": "high",
                "outcomes": list(data["outcomes"]),
                "investment": float(data["investment"]),
                "deploy_cost": 0.0,
                "belief_before": None,
                "belief_after": None,
                "signal_type": "high",
            }
            self.g.add_node(node_id, **self.honeynet_nodes[node_id])
            self.g.add_edge(
                previous,
                node_id,
                outcomes=list(data["outcomes"]),
                investment=float(data["investment"]),
                honeynet=True,
            )
            previous = node_id

    def leave_honeynet(self) -> None:
        self.in_honeynet = False
        self.honeynet_path_id = None
        self.honeynet_entry_layer = None
        self.honeynet_entry_node = None

    def attackerPeekLayer(
        self,
    ) -> dict[
        tuple[Any, Any],
        dict[str, Any],
    ]:
        if self.in_honeynet:
            visible: dict[Any, dict[str, Any]] = {}
            current = self.current_node

            if current == self.honeynet_entry_node:
                next_id = ("H", 1)
                data = self.g.nodes[next_id]
                visible[(current, next_id)] = {
                    "outcomes": list(data["outcomes"]),
                    "investment": float(data["investment"]),
                    "interaction": "high",
                    "type": "Fake",
                    "action": "enter_honeynet",
                }
                return visible

            if isinstance(current, tuple) and len(current) == 2 and current[0] == "H":
                step = int(current[1])
                if step >= 5:
                    return {}

                next_h = ("H", step + 1)
                data = self.g.nodes[next_h]
                visible[(current, next_h)] = {
                    "outcomes": list(data["outcomes"]),
                    "investment": float(data["investment"]),
                    "interaction": "high",
                    "type": "Fake",
                    "action": "stay",
                }


                if self.honeynet_path_id is not None:
                    start_layer = 1
                    for idx in range(1, self.nodes_per_layer + 1):
                        if idx == self.honeynet_path_id:
                            continue
                        node_id = (start_layer, idx)
                        node_data = self.g.nodes[node_id]
                        if node_data.get("known"):
                            continue
                        visible[(current, node_id)] = {
                            "outcomes": list(node_data["outcomes"]),
                            "investment": float(node_data["investment"]),
                            "interaction": node_data["interaction"],
                            "type": node_data["type"],
                            "action": "switch",
                        }
                return visible

        if self.current_layer >= self.n_layers:
            return {}

        visible = {}

        if self.current_layer == 0:
            next_layer = 1
            for idx in range(1, self.nodes_per_layer + 1):
                node_id = (next_layer, idx)
                data = self.g.nodes[node_id]

                visible[(self.current_node, node_id)] = {
                    "outcomes": list(data["outcomes"]),
                    "investment": float(data["investment"]),
                    "interaction": data["interaction"],
                    "type": data["type"],
                    "action": "main",
                }

            return visible

        current_path_id = self.current_node[1]

        stay_node = (self.current_layer + 1, current_path_id)
        stay_data = self.g.nodes[stay_node]
        visible[(self.current_node, stay_node)] = {
            "outcomes": list(stay_data["outcomes"]),
            "investment": float(stay_data["investment"]),
            "interaction": stay_data["interaction"],
            "type": stay_data["type"],
            "action": "stay",
        }

        for idx in range(1, self.nodes_per_layer + 1):
            if idx == current_path_id:
                continue

            switch_node = (1, idx)
            switch_data = self.g.nodes[switch_node]

            if switch_data.get("known"):
                continue

            visible[(self.current_node, switch_node)] = {
                "outcomes": list(switch_data["outcomes"]),
                "investment": float(switch_data["investment"]),
                "interaction": switch_data["interaction"],
                "type": switch_data["type"],
                "action": "switch",
            }

        return visible

    def signal_type_for_node(self, node_id: Any) -> str:
        data = self.g.nodes[node_id]

        if data.get("type") == "Real":
            return "real"
        if data.get("interaction") == "low":
            return "low"
        return "high"

    def record_belief_update(
        self,
        node_id: Any,
        prior: float,
        posterior: float,
        signal_type: str,
    ) -> None:
        self.g.nodes[node_id].update(
            belief_before=float(prior),
            belief_after=float(posterior),
            signal_type=signal_type,
        )

    def exploreNode(
        self,
        u: Any,
        v: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if v not in self.g:
            raise KeyError(f"Unknown node: {v}")

        self.g.nodes[v]["known"] = True
        self.current_node = v

        if isinstance(v, tuple) and len(v) == 2 and v[0] == "H":
            self.current_layer = self.honeynet_entry_layer or 0
        elif isinstance(v, tuple):
            self.current_layer = int(v[0])

        if self.globalLog is not None:
            self.globalLog.info(
                f"Explored node {v}: "
                f"type={self.g.nodes[v].get('type')}, "
                f"interaction={self.g.nodes[v].get('interaction')}, "
                f"outcomes={self.g.nodes[v].get('outcomes')}, "
                f"investment={self.g.nodes[v].get('investment')}"
            )


        edge_data = (
            dict(self.g[u][v])
            if self.g.has_edge(u, v)
            else {
                "outcomes": list(self.g.nodes[v].get("outcomes", [])),
                "investment": float(self.g.nodes[v].get("investment", 0.0)),
                "honeynet": False,
            }
        )

        return dict(self.g.nodes[v]), edge_data

    def viewCopy(self) -> NxGraph:
        return self.g.copy(as_view=False)

    def __getattr__(self, name: str) -> Any:
        if name in {"globalLog", "defender", "g"}:
            return super().__getattribute__(name)
        return getattr(self.g, name)


class Attacker:
    def __init__(
        self,
        global_log: LFMLog | None,
        net_graph: Graph,
        variant: str = "rational",
        loss_aversion_lambda: float = DEFAULT_LAMBDA,
        controlled_investment: float | None = None,
        initial_beliefs: dict[tuple[int, int], float] | None = None,
    ):
        if variant not in ATTACKER_VARIANTS:
            raise ValueError(f"Unknown attacker variant: {variant}")
        if loss_aversion_lambda < 0:
            raise ValueError("loss_aversion_lambda must be non-negative")
        if controlled_investment is not None:
            validate_unit_interval(controlled_investment, "controlled_investment")
        self.global_log = global_log
        self.g = net_graph
        self.variant = variant
        self.variant_config = dict(ATTACKER_VARIANTS[variant])
        self.attacker_type = self.variant_config["label"]

        self.loss_aversion_lambda = float(loss_aversion_lambda)
        self.controlled_investment = controlled_investment


        if initial_beliefs is None:
            configured_beliefs = {
                (layer, path_id): round(
                    random.uniform(
                        INITIAL_BELIEF_MIN,
                        INITIAL_BELIEF_MAX,
                    ),
                    4,
                )
                for layer in range(1, self.g.n_layers + 1)
                for path_id in range(1, self.g.nodes_per_layer + 1)
            }
        else:
            configured_beliefs = {
                tuple(node_id): float(value)
                for node_id, value in initial_beliefs.items()
            }

            expected_nodes = {
                (layer, path_id)
                for layer in range(1, self.g.n_layers + 1)
                for path_id in range(1, self.g.nodes_per_layer + 1)
            }

            if set(configured_beliefs) != expected_nodes:
                raise ValueError(
                    "initial_beliefs must provide one belief for every "
                    "network node (layer, path_id)."
                )

        for node_id, belief in configured_beliefs.items():
            validate_unit_interval(
                belief,
                f"initial_belief_node_{node_id}",
            )

        self.initial_node_beliefs = {
            node_id: float(belief)
            for node_id, belief in configured_beliefs.items()
        }


        self.node_beliefs = dict(self.initial_node_beliefs)

        self.path_investment = 0.0
        self.honeynet_investment = 0.0
        self.total_investment_paid = 0.0
        self.total_realized_outcome = 0.0

        self.exploredNodes: dict[Any, int] = {}
        self.explorationCounter = 0
        self.path: list[str] = []
        self.completed_game = False
        self.last_choice = "NONE"

        self.real_assets_chosen = 0
        self.honeypots_chosen = 0
        self.low_interaction_honeypots_chosen = 0
        self.high_interaction_honeypots_chosen = 0
        self.max_hih_depth_reached = 0


        self.real_network_steps = 0
        self.honeynet_steps = 0

        self.entered_honeynet = False
        self.honeynet_entry_node = "NONE"
        self.honeynet_entry_step = 0


        self.termination_reason = ""

        self.path_switches = 0
        self.same_path_choices = 0

        self.belief_trace: list[str] = []
        self.last_posterior_honeypot_belief = self.initial_node_beliefs[(1, 1)]

    def current_sunk_investment(self) -> float:
        if self.controlled_investment is not None:
            return float(self.controlled_investment)
        if self.g.in_honeynet:
            return float(self.honeynet_investment)
        return float(self.path_investment)

    def _belief_for_node(self, node_id: Any) -> float:
        if node_id in self.node_beliefs:
            return float(self.node_beliefs[node_id])
        if isinstance(node_id, tuple) and len(node_id) == 2 and node_id[0] == "H":


            return 1.0
        return float(self.initial_node_beliefs[node_id])

    def _candidate_value(
        self,
        data: dict[str, Any],
        node_id: Any,
        is_switch: bool,
    ) -> float:
        belief = self._belief_for_node(node_id)
        outcomes = np.asarray(data["outcomes"], dtype=float)
        probabilities = overall_prob(belief, outcomes)

        value = pt_value(
            outcomes,
            probabilities,
            weighting=self.variant_config["weighting"],
            loss_aversion=self.variant_config["loss_aversion"],
            lambda_loss=self.loss_aversion_lambda,
        )

        if is_switch and self.variant_config["investment"]:
            sunk_investment = self.current_sunk_investment()

            if self.variant_config["loss_aversion"]:
                value -= self.loss_aversion_lambda * sunk_investment
            else:
                value -= sunk_investment

        return float(value)

    def _update_path_investment(
        self,
        previous_path_id: int | None,
        chosen_path_id: int,
        chosen_investment: float,
    ) -> None:
        chosen_investment = validate_unit_interval(
            chosen_investment,
            "chosen_investment",
        )

        self.total_investment_paid += chosen_investment

        if previous_path_id is None:
            self.path_investment = chosen_investment

        elif chosen_path_id == previous_path_id:
            self.same_path_choices += 1
            self.path_investment = min(
                1.0,
                self.path_investment + chosen_investment,
            )

        else:
            self.path_switches += 1
            self.path_investment = min(
                1.0,
                self.path_investment + chosen_investment,
            )

    def _bayesian_update_after_compromise(
        self,
        node_id: Any,
        node_data: dict[str, Any],
    ) -> None:
        if isinstance(node_id, tuple) and len(node_id) == 2 and node_id[0] == "H":


            self.last_posterior_honeypot_belief = 1.0
            return

        path_id = node_id[1]
        prior = self._belief_for_node(node_id)
        signal_type = self.g.signal_type_for_node(node_id)

        posterior = bayesian_update(
            prior,
            SIGNAL_LIKELIHOODS[signal_type],
        )

        self.node_beliefs[node_id] = posterior
        self.last_posterior_honeypot_belief = posterior

        self.g.record_belief_update(
            node_id,
            prior,
            posterior,
            signal_type,
        )

        self.belief_trace.append(
            f"L{node_id[0]}P{path_id}:{prior:.3f}->{posterior:.3f}({signal_type})"
        )

        if self.global_log is not None:
            self.global_log.info(
                f"Bayesian update at {node_id}: "
                f"signal={signal_type}, b_t={prior:.4f}, b_t+1={posterior:.4f}"
            )

    def _select_initial_choice(
        self,
        potential: dict[Any, dict[str, Any]],
    ) -> Any:
        choice, _ = max(
            (
                (
                    edge,
                    self._candidate_value(
                        data,
                        node_id=edge[1],
                        is_switch=False,
                    ),
                )
                for edge, data in potential.items()
            ),
            key=lambda item: item[1],
        )

        return choice

    def _select_post_investment_choice(
        self,
        potential: dict[Any, dict[str, Any]],
    ) -> Any:


        stay_candidates = [
            (edge, data)
            for edge, data in potential.items()
            if data.get("action") == "stay"
        ]

        switch_candidates = [
            (edge, data)
            for edge, data in potential.items()
            if data.get("action") == "switch"
        ]

        if not stay_candidates:
            return next(iter(potential))

        stay_edge, stay_data = stay_candidates[0]
        stay_value = self._candidate_value(
            stay_data,
            node_id=stay_edge[1],
            is_switch=False,
        )

        if not switch_candidates:
            return stay_edge

        best_switch_edge, best_switch_value = max(
            (
                (
                    edge,
                    self._candidate_value(
                        data,
                        node_id=edge[1],
                        is_switch=True,
                    ),
                )
                for edge, data in switch_candidates
            ),
            key=lambda item: item[1],
        )

        return (
            best_switch_edge
            if best_switch_value > stay_value
            else stay_edge
        )

    def _select_honeynet_choice(
        self,
        potential: dict[Any, dict[str, Any]],
    ) -> Any:


        stay_candidates = [
            (edge, data) for edge, data in potential.items()
            if data.get("action") in {"enter_honeynet", "stay"}
        ]
        switch_candidates = [
            (edge, data) for edge, data in potential.items()
            if data.get("action") == "switch"
        ]

        if not switch_candidates:
            return stay_candidates[0][0]

        stay_edge, stay_data = stay_candidates[0]
        stay_value = self._candidate_value(stay_data, stay_edge[1], is_switch=False)

        best_switch_edge, best_switch_value = max(
            (
                (
                    edge,
                    self._candidate_value(data, edge[1], is_switch=True),
                )
                for edge, data in switch_candidates
            ),
            key=lambda item: item[1],
        )

        return best_switch_edge if best_switch_value > stay_value else stay_edge

    def makeDecision(self) -> bool:
        if self.completed_game:
            return False

        potential = self.g.attackerPeekLayer()

        if not potential:
            if self.g.in_honeynet:
                self.termination_reason = "honeynet_end"
            elif self.g.current_layer >= self.g.n_layers:
                self.termination_reason = "network_end"
            else:
                self.termination_reason = "no_available_move"
            self.completed_game = True
            return False

        in_honeynet = self.g.in_honeynet
        previous_path_id = (
            None
            if in_honeynet
            else (None if self.g.current_layer == 0 else self.g.current_node[1])
        )

        if self.variant == "random":

            choice = random.choice(list(potential))
        elif in_honeynet:
            choice = self._select_honeynet_choice(potential)
        elif self.g.current_layer == 0:
            choice = self._select_initial_choice(potential)
        else:
            choice = self._select_post_investment_choice(potential)

        choice_data = potential[choice]
        node_data, edge_data = self.g.exploreNode(choice[0], choice[1])

        self.explorationCounter += 1
        self.exploredNodes[choice[1]] = self.explorationCounter
        self.last_choice = str(choice[1])
        self.path.append(str(choice[1]))


        if (
            isinstance(choice[1], tuple)
            and len(choice[1]) == 2
            and choice[1][0] == "H"
        ):
            self.honeynet_steps += 1
            self.max_hih_depth_reached = max(
                self.max_hih_depth_reached,
                int(choice[1][1]),
            )
        else:
            self.real_network_steps += 1


        if in_honeynet:
            action_now = choice_data.get("action")
            chosen_investment = validate_unit_interval(
                float(choice_data["investment"]),
                "chosen_investment",
            )

            if (
                isinstance(choice[1], tuple)
                and len(choice[1]) == 2
                and choice[1][0] == "H"
                and action_now in {"enter_honeynet", "stay"}
            ):
                self.total_investment_paid += chosen_investment
                self.honeynet_investment = min(
                    1.0,
                    self.honeynet_investment + chosen_investment,
                )

            elif action_now == "switch":
                self.total_investment_paid += chosen_investment
                self.path_switches += 1
                self.path_investment = min(
                    1.0,
                    self.honeynet_investment + chosen_investment,
                )

        else:
            self._update_path_investment(
                previous_path_id,
                choice[1][1],
                float(choice_data["investment"]),
            )

        self._bayesian_update_after_compromise(choice[1], node_data)
        self.compute_outcomes(node_data, edge_data)


        self.g.activate_scheduled_lihs()

        node_type = node_data.get("type")
        interaction = node_data.get("interaction")
        action = choice_data.get("action")


        if (
            node_type == "Fake"
            and interaction == "high"
            and isinstance(choice[1], tuple)
            and len(choice[1]) == 2
            and choice[1][0] == "H"
            and int(choice[1][1]) >= 5
        ):
            self.termination_reason = "honeynet_end"
            self.completed_game = True
            return False

        if (
            node_type == "Fake"
            and interaction == "low"
            and choice[1] not in self.g.triggered_lih_locations
        ):
            self.g.triggered_lih_locations.add(choice[1])
            if self.global_log is not None:
                self.global_log.info(
                    f"LIH encountered at layer={choice[1][0]}, path={choice[1][1]}"
                )
            self.honeynet_investment = self.path_investment
            self.g.enter_honeynet(
                path_id=choice[1][1],
                entry_layer=choice[1][0],
                entry_node=choice[1],
            )
            return True

        if in_honeynet and action == "switch":
            self.g.leave_honeynet()

        if (
            not self.g.in_honeynet
            and isinstance(choice[1], tuple)
            and len(choice[1]) == 2
            and isinstance(choice[1][0], int)
            and self.g.current_layer >= self.g.n_layers
            and node_type == "Real"
        ):
            self.termination_reason = "network_end"
            self.completed_game = True
            return False

        return True

    def compute_outcomes(
        self,
        node_data: dict[str, Any],
        edge_data: dict[str, Any],
    ) -> None:
        del edge_data

        realized = realized_outcome_for_node(
            node_data["outcomes"],
            node_data.get("type", "Real"),
        )
        self.total_realized_outcome += realized

        node_type = node_data.get("type")
        interaction = node_data.get("interaction")

        if node_type == "Real":
            self.real_assets_chosen += 1

        elif node_type == "Fake":
            self.honeypots_chosen += 1

            if interaction == "low":
                self.low_interaction_honeypots_chosen += 1
            elif interaction == "high":
                self.high_interaction_honeypots_chosen += 1

            if interaction == "high" and not self.entered_honeynet:
                self.entered_honeynet = True
                self.honeynet_entry_node = self.last_choice
                self.honeynet_entry_step = self.explorationCounter


def render_html(
    graph: Graph,
    output_dir: str = "graphvisuals_test",
    open_in_browser: bool = True,
    root_node: Any = None,
    explored_nodes: dict[Any, int] | None = None,
    scenario_name: str = "",
) -> str:
    if Network is None:
        raise SystemExit("Missing visualization dependency: pyvis")

    explored = explored_nodes or {}
    vis_graph = graph.viewCopy()

    for node, data in vis_graph.nodes(data=True):
        outcomes = data.get("outcomes", [0.0, 0.0, 0.0])
        investment = float(data.get("investment", 0.0))
        layer = int(data.get("layer", 0))
        node_type = data.get("type", "")
        interaction = data.get("interaction", "")
        belief_before = data.get("belief_before")
        belief_after = data.get("belief_after")

        if node_type == "Entry":
            data["color"] = {
                "background": "#c6e0b4",
                "border": "#548235",
            }
            data["label"] = "Entry Point"
        else:
            if node_type == "Fake":
                name = (
                    "Low-interaction HP"
                    if interaction == "low"
                    else "High-interaction HP"
                )
                data["color"] = {
                    "background": "#f8d7b5",
                    "border": "#c55a11",
                }
            else:
                name = "Real asset"
                data["color"] = {
                    "background": "#d9eaf7",
                    "border": "#2e75b6",
                }

            belief_text = ""
            if belief_before is not None and belief_after is not None:
                belief_text = (
                    f"\nb(H): {float(belief_before):.2f}"
                    f" -> {float(belief_after):.2f}"
                )

            data["label"] = (
                f"Layer {layer} · Node {node[1]}\n"
                f"{name}\n"
                f"X={outcomes}\n"
                f"I={investment:.2f}"
                f"{belief_text}"
            )

        data["explored"] = node in explored
        data["exploredOrder"] = explored.get(node, -1)
        data["borderWidth"] = 4 if node in explored else 2

        if node in explored:
            data["color"] = {
                "background": "#ffe0e0",
                "border": "#cc0000",
            }

    for _, _, data in vis_graph.edges(data=True):
        data["color"] = {
            "color": "#cbd5e1",
            "opacity": 0.24,
        }
        data["width"] = 0.8

    levels = (
        single_source_shortest_path_length(vis_graph, root_node)
        if root_node in vis_graph
        else None
    )

    mapping = {node: str(node) for node in vis_graph.nodes()}
    vis_graph = nx.relabel_nodes(vis_graph, mapping)

    if levels is not None:
        for original_node, level in levels.items():
            vis_graph.nodes[mapping[original_node]]["level"] = level

    net = Network(height="1000px", width="100%", notebook=False)
    net.from_nx(vis_graph)
    net.set_options(VIS_OPTIONS_JSON)

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    filename = (
        f"{timestamp}_{scenario_name}_graph.html"
        if scenario_name
        else f"{timestamp}_graph.html"
    )

    filepath = os.path.join(output_dir, filename)

    if open_in_browser:
        net.show(filepath, notebook=False)
    else:
        net.save_graph(filepath)

    return filepath


def run_attacker_episode(
    graph: Graph,
    attacker: Attacker,
    max_steps: int = DEFAULT_MAX_GAME_STEPS,
    on_step: Any = None,
) -> dict[str, Any]:


    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    for _ in range(max_steps):
        if attacker.completed_game:
            break
        if not graph.attackerPeekLayer():
            attacker.makeDecision()
            break
        if on_step is not None:
            on_step(graph, attacker)
        if not attacker.makeDecision():
            break
    if not attacker.completed_game:
        attacker.completed_game = True
        attacker.termination_reason = "horizon"
    return {
        "attacker_variant": attacker.variant,
        "attacker_label": attacker.attacker_type,
        "path": list(attacker.path),
        "moves_used": attacker.explorationCounter,
        "termination_reason": attacker.termination_reason,
        "in_honeynet_at_end": bool(graph.in_honeynet and attacker.entered_honeynet),
        "entered_honeynet": attacker.entered_honeynet,
        "low_interaction_honeypots_chosen": attacker.low_interaction_honeypots_chosen,
        "high_interaction_honeypots_chosen": attacker.high_interaction_honeypots_chosen,
        "honeynet_steps": attacker.honeynet_steps,
        "real_network_steps": attacker.real_network_steps,
        "path_switches": attacker.path_switches,
        "path_investment": attacker.path_investment,
        "honeynet_investment": attacker.honeynet_investment,
        "total_investment_paid": attacker.total_investment_paid,
        "total_realized_outcome": attacker.total_realized_outcome,
        "last_posterior_honeypot_belief": attacker.last_posterior_honeypot_belief,
        "belief_trace": list(attacker.belief_trace),
    }


def parse_locations(value: str) -> list[Location]:
    if not value.strip():
        return []
    result: list[Location] = []
    for part in value.split(","):
        terms = part.strip().split(":")
        if len(terms) != 2:
            raise ValueError("--initial-lihs must look like 1:1,2:3")
        result.append((int(terms[0]), int(terms[1])))
    return result


def run_self_tests() -> None:
    import math
    print("Checking Bayesian updates, PT values, graph, and LIH activation...")
    assert bayesian_update(0.3, SIGNAL_LIKELIHOODS["real"]) < 0.3
    assert bayesian_update(0.3, SIGNAL_LIKELIHOODS["low"]) > 0.3
    assert bayesian_update(0.3, SIGNAL_LIKELIHOODS["high"]) > 0.3
    for _ in range(20):
        vals = make_random_outcomes()
        p_h, p_r = conditional_outcome_probabilities(vals)
        assert math.isclose(float(p_h.sum()), 1.0)
        assert math.isclose(float(p_r.sum()), 1.0)
    for variant in ATTACKER_VARIANTS:
        random.seed(123)
        graph = Graph(None, environment=build_fixed_environment(999))
        attacker = Attacker(None, graph, variant=variant)
        result = run_attacker_episode(graph, attacker, max_steps=20)
        assert result["moves_used"] > 0, variant
        assert result["real_network_steps"] + result["honeynet_steps"] == result["moves_used"]
        assert result["termination_reason"]
    graph = Graph(None, environment=build_fixed_environment(999))
    graph.schedule_lih((2, 1))
    assert (2, 1) in graph.pending_lihs
    assert graph.g.nodes[(2, 1)]["type"] == "Real"
    attacker = Attacker(None, graph, variant="random")
    assert attacker.makeDecision() is True
    assert (2, 1) not in graph.pending_lihs
    assert graph.g.nodes[(2, 1)]["type"] == "Fake"
    assert (2, 1) in graph.active_lihs
    try:
        graph.schedule_lih((3, 1))
        raise AssertionError("same-path placement should be rejected")
    except ValueError:
        pass


    graph = Graph(None, environment=build_fixed_environment(222))
    random.seed(333)
    att = Attacker(None, graph, variant="rational")
    candidates = graph.attackerPeekLayer()
    chosen = att._select_initial_choice(candidates)[1]
    graph.install_initial_lih(chosen)
    assert att.makeDecision() is True
    assert graph.in_honeynet, (chosen, graph.current_node)
    assert graph.attackerPeekLayer()
    assert graph.attackerPeekLayer()[(chosen, ("H", 1))]["action"] == "enter_honeynet"
    print(f"Passed all tests ({len(ATTACKER_VARIANTS)} attacker variants + graph/HIH tests).")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Attacker/network-only SCDG simulator (no defender algorithms)."
    )
    parser.add_argument("--attacker", choices=list(ATTACKER_VARIANTS), default="investment")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_GAME_STEPS)
    parser.add_argument("--lambda-value", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--investment", type=float, default=None)
    parser.add_argument("--initial-lihs", default="", help="Fixture only: e.g., 1:1,2:3")
    parser.add_argument("--html", action="store_true", help="Render optional pyvis HTML graph")
    parser.add_argument("--output-dir", default="graphvisuals_test")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        run_self_tests()
        return 0
    random.seed(args.seed)
    np.random.seed(args.seed)
    environment = build_fixed_environment(args.seed)
    graph = Graph(None, environment=environment)
    for loc in parse_locations(args.initial_lihs):
        graph.install_initial_lih(loc)
    attacker = Attacker(
        None, graph, variant=args.attacker,
        loss_aversion_lambda=args.lambda_value,
        controlled_investment=args.investment,
    )
    result = run_attacker_episode(graph, attacker, max_steps=args.max_steps)
    print("Attacker variant:", result["attacker_variant"])
    print("Moves used:", result["moves_used"])
    print("Path:", " -> ".join(result["path"]))
    print("Termination:", result["termination_reason"])
    print("Ever entered HIH:", result["entered_honeynet"])
    print("In HIH at end:", result["in_honeynet_at_end"])
    print("Belief updates:", " | ".join(result["belief_trace"]))
    print("Total investment:", round(result["total_investment_paid"], 4))
    if args.html:
        print("HTML graph:", render_html(
            graph, output_dir=args.output_dir,
            open_in_browser=False, root_node=(0, 0),
            explored_nodes=attacker.exploredNodes,
            scenario_name=args.attacker,
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
