from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from Cyber_Deception_Attackers_and_Network import ATTACKER_VARIANTS
from cyber_deception_env import CyberDeceptionEnv


BACKENDS = {
    "dqn": ("dqn", "DQNConfig", "DQNAgent"),
    "ddqn": ("doubledqn", "DDQNConfig", "DDQNAgent"),
    "pdqn": ("pdqn", "PDQNConfig", "PDQNAgent"),
}


@dataclass
class DefenderPolicy:
    name: str
    algorithm: str
    checkpoint: Path
    agent: Any
    config: Any

    def act(self, observation: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
        return int(self.agent.select_action(observation, mask, epsilon=0.0, rng=rng))


def backend(algorithm: str):
    import importlib

    if algorithm not in BACKENDS:
        raise ValueError(f"Unsupported oracle {algorithm!r}; available: {sorted(BACKENDS)}")
    module_name, config_name, agent_name = BACKENDS[algorithm]
    module = importlib.import_module(module_name)
    return module, getattr(module, config_name), getattr(module, agent_name)


def device_for(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    result = torch.device(name)
    if result.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but unavailable")
    if result.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise ValueError("MPS device requested but unavailable")
    return result


def read_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_policy(
    name: str, algorithm: str, path: Path, device: torch.device
) -> DefenderPolicy:
    checkpoint = read_checkpoint(path, device)
    _, config_cls, agent_cls = backend(algorithm)
    config = config_cls(**checkpoint["config"])
    agent = agent_cls(
        int(checkpoint["observation_dim"]),
        int(checkpoint["action_dim"]),
        config,
        device,
    )
    agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
    agent.policy_net.eval()
    if "target_state_dict" in checkpoint:
        agent.target_net.load_state_dict(checkpoint["target_state_dict"])
    agent.target_net.eval()
    return DefenderPolicy(name, algorithm, path.resolve(), agent, config)


def parse_policy_spec(spec: str, device: torch.device, index: int) -> DefenderPolicy:
    if "=" not in spec:
        raise ValueError("Policy must be specified as dqn=path, ddqn=path, or pdqn=path")
    algorithm, checkpoint = spec.split("=", 1)
    algorithm = algorithm.strip().lower()
    return load_policy(f"{algorithm}_imported_{index:02d}", algorithm, Path(checkpoint), device)


def common_game_settings(policies: Sequence[DefenderPolicy]) -> dict[str, Any]:
    if not policies:
        raise ValueError("At least one defender policy is required")
    names = (
        "max_steps", "lih_budget", "network_seed", "vary_network",
        "loss_aversion_lambda", "controlled_investment",
    )
    reference = {name: getattr(policies[0].config, name) for name in names}
    for policy in policies[1:]:
        current = {name: getattr(policy.config, name) for name in names}
        if current != reference:
            raise ValueError(
                f"Environment mismatch: {policy.name} has {current}, "
                f"but {policies[0].name} has {reference}. "
                "Train or select checkpoints with identical game settings."
            )
    return reference


def environment(
    settings: dict[str, Any], attackers: Sequence[str], random_network: bool = False
) -> CyberDeceptionEnv:
    return CyberDeceptionEnv(
        attacker_variants=tuple(attackers),
        lih_budget=int(settings["lih_budget"]),
        max_steps=int(settings["max_steps"]),
        network_seed=(None if random_network else settings["network_seed"]),
        loss_aversion_lambda=float(settings["loss_aversion_lambda"]),
        controlled_investment=settings["controlled_investment"],
    )


def verify_dimensions(policies: Sequence[DefenderPolicy], env: CyberDeceptionEnv) -> None:
    obs_dim = int(np.prod(env.observation_space.shape))
    actions = int(env.action_space.n)
    for policy in policies:
        if policy.agent.observation_dim != obs_dim or policy.agent.action_dim != actions:
            raise ValueError(
                f"{policy.name} has incompatible network dimensions: "
                f"{policy.agent.observation_dim}/{policy.agent.action_dim}, "
                f"expected {obs_dim}/{actions}"
            )


def uniform_meta_strategy(policies: Sequence[DefenderPolicy]) -> np.ndarray:
    if not policies:
        raise ValueError("Cannot form a mixture from an empty policy population")
    return np.full(len(policies), 1.0 / len(policies), dtype=np.float64)


def play_episode(
    env: CyberDeceptionEnv,
    variant: str,
    episode_seed: int,
    action_rng: np.random.Generator,
    policy: DefenderPolicy | None,
) -> dict[str, Any]:
    observation, info = env.reset(
        seed=episode_seed, options={"attacker_variant": variant}
    )
    total_reward = 0.0
    while True:
        mask = np.asarray(info["action_mask"], dtype=np.bool_)
        if policy is None:
            valid = np.flatnonzero(mask)
            action = int(action_rng.choice(valid))
        else:
            action = policy.act(observation, mask, action_rng)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            break
    return {
        "success": int(bool(info["ended_in_honeynet"])),
        "entry": int(bool(info["entered_honeynet"])),
        "reward": total_reward,
        "moves": int(info["moves_used"]),
        "lihs": int(info["placed_lihs"]),
    }


def summarize(episodes: list[dict[str, Any]]) -> dict[str, float]:
    if not episodes:
        raise ValueError("Cannot summarize zero episodes")
    n = len(episodes)
    return {
        "episodes": float(n),
        "success_rate": sum(e["success"] for e in episodes) / n,
        "entry_rate": sum(e["entry"] for e in episodes) / n,
        "average_reward": sum(e["reward"] for e in episodes) / n,
        "average_moves": sum(e["moves"] for e in episodes) / n,
        "average_lihs": sum(e["lihs"] for e in episodes) / n,
    }


def evaluate_population(
    policies: Sequence[DefenderPolicy],
    attackers: Sequence[str],
    episodes_per_attacker: int,
    seed_base: int,
    random_network: bool,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    settings = common_game_settings(policies)
    env = environment(settings, attackers, random_network)
    verify_dimensions(policies, env)
    matrix = np.zeros((len(policies), len(attackers)), dtype=np.float64)
    metrics: list[dict[str, Any]] = []
    try:
        for i, policy in enumerate(policies):
            for j, variant in enumerate(attackers):
                results = []
                for k in range(episodes_per_attacker):
                    episode_seed = seed_base + j * 100_000 + k
                    rng = np.random.default_rng(episode_seed + 31)
                    results.append(play_episode(env, variant, episode_seed, rng, policy))
                result = summarize(results)
                matrix[i, j] = result["success_rate"]
                metrics.append({"policy": policy.name, "attacker": variant, **result})
    finally:
        env.close()
    return matrix, metrics


def evaluate_episode_mixture(
    policies: Sequence[DefenderPolicy],
    attackers: Sequence[str],
    episodes_per_attacker: int,
    seed_base: int,
    random_network: bool,
    random_baseline: bool = False,
) -> list[dict[str, Any]]:
    settings = common_game_settings(policies)
    env = environment(settings, attackers, random_network)
    verify_dimensions(policies, env)
    weights = uniform_meta_strategy(policies)
    selection_rng = np.random.default_rng(seed_base + 197)
    results: list[dict[str, Any]] = []
    try:
        for j, variant in enumerate(attackers):
            episodes: list[dict[str, Any]] = []
            for k in range(episodes_per_attacker):
                episode_seed = seed_base + j * 100_000 + k
                action_rng = np.random.default_rng(episode_seed + 31)
                selected = (
                    None if random_baseline
                    else policies[int(selection_rng.choice(len(policies), p=weights))]
                )
                episodes.append(play_episode(env, variant, episode_seed, action_rng, selected))
            results.append({"attacker": variant, **summarize(episodes)})
    finally:
        env.close()
    return results


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_and_save(
    policies: Sequence[DefenderPolicy],
    attackers: Sequence[str],
    output_dir: Path,
    payoff_episodes: int,
    mixture_episodes: int,
    seed_base: int,
    random_network: bool,
) -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = uniform_meta_strategy(policies)
    payoff, pure_metrics = evaluate_population(
        policies, attackers, payoff_episodes, seed_base, random_network
    )
    expected = weights @ payoff
    mixture = evaluate_episode_mixture(
        policies, attackers, mixture_episodes, seed_base + 1_000_000, random_network
    )
    baseline = evaluate_episode_mixture(
        policies, attackers, mixture_episodes, seed_base + 1_000_000,
        random_network, random_baseline=True,
    )

    write_csv(
        output_dir / "empirical_payoffs.csv",
        ["defender_policy", *attackers, "mean_containment", "worst_case_containment"],
        [
            {
                "defender_policy": policy.name,
                **{variant: float(payoff[i, j]) for j, variant in enumerate(attackers)},
                "mean_containment": float(payoff[i].mean()),
                "worst_case_containment": float(payoff[i].min()),
            }
            for i, policy in enumerate(policies)
        ] + [
            {
                "defender_policy": "uniform_expected",
                **{variant: float(expected[j]) for j, variant in enumerate(attackers)},
                "mean_containment": float(expected.mean()),
                "worst_case_containment": float(expected.min()),
            }
        ],
    )
    write_csv(
        output_dir / "pure_policy_metrics.csv",
        ["policy", "attacker", "episodes", "success_rate", "entry_rate", "average_reward", "average_moves", "average_lihs"],
        pure_metrics,
    )
    write_csv(
        output_dir / "uniform_meta_strategy.csv",
        ["policy", "algorithm", "checkpoint", "probability"],
        [
            {
                "policy": policy.name,
                "algorithm": policy.algorithm,
                "checkpoint": str(policy.checkpoint),
                "probability": float(weights[i]),
            }
            for i, policy in enumerate(policies)
        ],
    )
    for filename, rows in (
        ("uniform_mixture_rollouts.csv", mixture),
        ("random_defender_baseline.csv", baseline),
    ):
        write_csv(
            output_dir / filename,
            ["attacker", "episodes", "success_rate", "entry_rate", "average_reward", "average_moves", "average_lihs"],
            rows,
        )
    manifest = {
        "method": "one_sided_psro_inspired_fixed_attacker_population",
        "meta_solver": "uniform_over_pure_defender_policies",
        "attacker_training_mixture": "uniform",
        "attacker_variants": list(attackers),
        "payoff_episodes_per_attacker": payoff_episodes,
        "mixture_episodes_per_attacker": mixture_episodes,
        "payoff_evaluation_seed_base": seed_base,
        "mixture_evaluation_seed_base": seed_base + 1_000_000,
        "evaluation_random_network": random_network,
        "game_settings": common_game_settings(policies),
        "policies": [
            {"name": policy.name, "algorithm": policy.algorithm,
             "checkpoint": str(policy.checkpoint), "weight": float(weights[i])}
            for i, policy in enumerate(policies)
        ],
    }
    with (output_dir / "population.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)

    expected_mean = float(expected.mean())
    sampled_mean = float(np.mean([x["success_rate"] for x in mixture]))
    baseline_mean = float(np.mean([x["success_rate"] for x in baseline]))
    print(f"\nDefender policies: {len(policies)}")
    print("Uniform meta weights:", ", ".join(
        f"{policy.name}={weights[i]:.3f}" for i, policy in enumerate(policies)
    ))
    print("\nAttacker                              Expected mixture  Sampled mixture  Random defender")
    for j, variant in enumerate(attackers):
        print(f"{variant:<37} {expected[j]:>10.3f} {mixture[j]['success_rate']:>15.3f} {baseline[j]['success_rate']:>16.3f}")
    print(f"Overall expected uniform containment: {expected_mean:.3f}")
    print(f"Overall sampled uniform containment:  {sampled_mean:.3f}")
    print(f"Overall random defender containment:  {baseline_mean:.3f}")
    print(f"Worst-attacker expected containment:  {float(expected.min()):.3f}")
    print(f"Results saved to: {output_dir}")
    return {
        "policies": float(len(policies)),
        "expected_uniform_containment": expected_mean,
        "sampled_uniform_containment": sampled_mean,
        "random_defender_containment": baseline_mean,
        "expected_uniform_worst_case": float(expected.min()),
    }


def train_one_oracle(
    algorithm: str,
    round_number: int,
    args: argparse.Namespace,
    attackers: Sequence[str],
    output_dir: Path,
    device: torch.device,
) -> DefenderPolicy:
    module, config_cls, _ = backend(algorithm)
    algorithm_offset = list(BACKENDS).index(algorithm) * 100_000
    training_seed = args.seed + 1009 * round_number + algorithm_offset
    config = config_cls(
        episodes=args.episodes_per_oracle,
        max_steps=args.max_steps,
        lih_budget=args.budget,
        network_seed=args.network_seed,
        vary_network=args.vary_network,
        seed=training_seed,
        evaluation_interval=min(args.training_eval_interval, args.episodes_per_oracle),
        evaluation_episodes=args.training_eval_episodes,
        loss_aversion_lambda=args.loss_aversion_lambda,
        controlled_investment=args.controlled_investment,
    )
    oracle_dir = output_dir / f"round_{round_number:02d}" / algorithm
    print(f"\n=== Round {round_number} | {algorithm.upper()} oracle | seed={training_seed} ===")
    _, best_path = module.run_training(config, attackers, oracle_dir, device)
    return load_policy(f"{algorithm}_r{round_number:02d}", algorithm, best_path, device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-sided PSRO-inspired defender population with a Uniform meta-solver"
    )
    parser.add_argument("--mode", choices=("train", "evaluate"), default="train")
    parser.add_argument("--policies", nargs="*", default=[], metavar="ALGORITHM=CHECKPOINT")
    parser.add_argument("--oracles", nargs="+", choices=tuple(BACKENDS), default=["dqn", "ddqn", "pdqn"])
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--episodes-per-oracle", type=int, default=6000)
    parser.add_argument("--attackers", nargs="+", choices=list(ATTACKER_VARIANTS), default=list(ATTACKER_VARIANTS))
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--network-seed", type=int, default=1000)
    parser.add_argument("--loss-aversion-lambda", type=float, default=2.0)
    parser.add_argument("--controlled-investment", type=float, default=None)
    parser.add_argument("--vary-network", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--training-eval-interval", type=int, default=500)
    parser.add_argument("--training-eval-episodes", type=int, default=25)
    parser.add_argument("--payoff-episodes", type=int, default=100)
    parser.add_argument("--mixture-episodes", type=int, default=100)
    parser.add_argument("--evaluation-seed", type=int, default=7_000_000)
    parser.add_argument("--eval-random-network", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("psro_results"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.attackers or len(set(args.attackers)) != len(args.attackers):
        raise ValueError("Attacker variants must be nonempty and unique")
    if args.payoff_episodes < 1 or args.mixture_episodes < 1:
        raise ValueError("Payoff and mixture episodes must be positive")
    if args.episodes_per_oracle < 1 or args.training_eval_episodes < 1:
        raise ValueError("Training and training-evaluation episode counts must be positive")
    if args.training_eval_interval < 1 or args.rounds < 0:
        raise ValueError("Evaluation interval must be positive and rounds nonnegative")
    if len(set(args.oracles)) != len(args.oracles):
        raise ValueError("Duplicate oracles are not allowed")
    device = device_for(args.device)
    policies = [parse_policy_spec(spec, device, i + 1) for i, spec in enumerate(args.policies)]
    if policies:
        reference = common_game_settings(policies)
        if args.mode == "train":
            requested = {
                "max_steps": args.max_steps,
                "lih_budget": args.budget,
                "network_seed": args.network_seed,
                "vary_network": args.vary_network,
                "loss_aversion_lambda": args.loss_aversion_lambda,
                "controlled_investment": args.controlled_investment,
            }
            if reference != requested:
                raise ValueError(
                    "Existing checkpoints were trained with different game settings. "
                    f"Checkpoint: {reference}; requested new training: {requested}"
                )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "evaluate":
        if not policies:
            raise ValueError("Evaluation requires --policies ALGORITHM=CHECKPOINT (e.g., pdqn=...)")
        evaluate_and_save(
            policies, args.attackers, args.output_dir, args.payoff_episodes,
            args.mixture_episodes, args.evaluation_seed, args.eval_random_network,
        )
        return
    if args.rounds < 1 and not policies:
        raise ValueError("Train at least one oracle or supply an initial --policies checkpoint")
    history: list[dict[str, Any]] = []
    for round_number in range(1, args.rounds + 1):
        for algorithm in args.oracles:
            policy = train_one_oracle(
                algorithm, round_number, args, args.attackers, args.output_dir, device
            )
            policies.append(policy)
        common_game_settings(policies)
        report = evaluate_and_save(
            policies, args.attackers,
            args.output_dir / f"round_{round_number:02d}" / "meta",
            args.payoff_episodes, args.mixture_episodes,
            args.evaluation_seed, args.eval_random_network,
        )
        history.append({"round": round_number, **report})
        write_csv(
            args.output_dir / "psro_history.csv",
            ["round", "policies", "expected_uniform_containment", "sampled_uniform_containment",
             "random_defender_containment", "expected_uniform_worst_case"],
            history,
        )
    evaluate_and_save(
        policies, args.attackers, args.output_dir, args.payoff_episodes,
        args.mixture_episodes, args.evaluation_seed, args.eval_random_network,
    )


if __name__ == "__main__":
    main()
