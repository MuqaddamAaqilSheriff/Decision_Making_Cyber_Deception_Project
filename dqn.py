from __future__ import annotations

import argparse
import csv
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from Cyber_Deception_Attackers_and_Network import ATTACKER_VARIANTS
from cyber_deception_env import CyberDeceptionEnv


@dataclass
class DQNConfig:
    episodes: int = 6000
    max_steps: int = 5
    lih_budget: int = 2
    network_seed: int = 1000
    vary_network: bool = False
    seed: int = 42
    hidden_dim: int = 256
    learning_rate: float = 0.0005
    gamma: float = 0.99
    replay_capacity: int = 20000
    batch_size: int = 64
    warmup_steps: int = 500
    train_frequency: int = 1
    target_update_frequency: int = 250
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 15000
    evaluation_interval: int = 500
    evaluation_episodes: int = 25
    gradient_clip: float = 10.0
    loss_aversion_lambda: float = 2.0
    controlled_investment: float | None = None
    torch_threads: int = 1


class QNetwork(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class ReplayBuffer:
    def __init__(self, capacity: int, observation_dim: int, action_dim: int):
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.capacity = capacity
        self.size = 0
        self.position = 0
        self.states = np.empty((capacity, observation_dim), dtype=np.float32)
        self.next_states = np.empty((capacity, observation_dim), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)
        self.next_masks = np.empty((capacity, action_dim), dtype=np.bool_)

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_mask: np.ndarray,
    ) -> None:
        i = self.position
        self.states[i] = state
        self.actions[i] = int(action)
        self.rewards[i] = float(reward)
        self.next_states[i] = next_state
        self.dones[i] = float(done)
        self.next_masks[i] = np.asarray(next_mask, dtype=np.bool_)
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        rng: np.random.Generator,
        device: torch.device,
    ) -> tuple[torch.Tensor, ...]:
        if self.size < batch_size:
            raise ValueError("Not enough transitions in replay buffer")
        indices = rng.integers(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.states[indices], device=device),
            torch.as_tensor(self.actions[indices], device=device),
            torch.as_tensor(self.rewards[indices], device=device),
            torch.as_tensor(self.next_states[indices], device=device),
            torch.as_tensor(self.dones[indices], device=device),
            torch.as_tensor(self.next_masks[indices], device=device),
        )


class DQNAgent:
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        config: DQNConfig,
        device: torch.device,
    ):
        self.config = config
        self.device = device
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.policy_net = QNetwork(observation_dim, action_dim, config.hidden_dim).to(device)
        self.target_net = QNetwork(observation_dim, action_dim, config.hidden_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=config.learning_rate)
        self.learn_steps = 0

    def select_action(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        epsilon: float,
        rng: np.random.Generator,
    ) -> int:
        mask = np.asarray(action_mask, dtype=np.bool_)
        valid = np.flatnonzero(mask)
        if valid.size == 0:
            raise RuntimeError("There are no valid defender actions")
        if rng.random() < epsilon:
            return int(rng.choice(valid))
        with torch.no_grad():
            state = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
            q_values = self.policy_net(state.unsqueeze(0)).squeeze(0)
            q_values = q_values.masked_fill(
                ~torch.as_tensor(mask, dtype=torch.bool, device=self.device),
                -torch.inf,
            )
            return int(q_values.argmax().item())

    def update(
        self,
        replay: ReplayBuffer,
        rng: np.random.Generator,
    ) -> float:
        states, actions, rewards, next_states, dones, next_masks = replay.sample(
            self.config.batch_size, rng, self.device
        )
        predicted_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            next_q_values = next_q_values.masked_fill(~next_masks, -torch.inf)
            next_max_q = next_q_values.max(dim=1).values
            next_max_q = torch.where(dones.bool(), torch.zeros_like(next_max_q), next_max_q)
            targets = rewards + self.config.gamma * (1.0 - dones) * next_max_q
        loss = F.smooth_l1_loss(predicted_q, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.config.gradient_clip)
        self.optimizer.step()
        self.learn_steps += 1
        if self.learn_steps % self.config.target_update_frequency == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return float(loss.item())


def linear_epsilon(config: DQNConfig, global_steps: int) -> float:
    fraction = min(1.0, global_steps / max(1, config.epsilon_decay_steps))
    return config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)


def make_environment(config: DQNConfig, attackers: Sequence[str]) -> CyberDeceptionEnv:
    return CyberDeceptionEnv(
        attacker_variants=tuple(attackers),
        lih_budget=config.lih_budget,
        max_steps=config.max_steps,
        network_seed=None if config.vary_network else config.network_seed,
        loss_aversion_lambda=config.loss_aversion_lambda,
        controlled_investment=config.controlled_investment,
    )


def validate_config(config: DQNConfig, attackers: Sequence[str]) -> None:
    if not attackers or len(set(attackers)) != len(attackers):
        raise ValueError("Supply at least one attacker variant without duplicates")
    unknown = set(attackers) - set(ATTACKER_VARIANTS)
    if unknown:
        raise ValueError(f"Unknown attacker variants: {sorted(unknown)}")
    if config.episodes < 1 or config.max_steps < 1:
        raise ValueError("Episodes and max_steps must be positive")
    if config.batch_size < 1 or config.replay_capacity < config.batch_size:
        raise ValueError("Replay capacity must be at least batch size")
    if config.warmup_steps < 0 or config.train_frequency < 1:
        raise ValueError("Invalid warmup or training frequency")
    if config.target_update_frequency < 1 or config.evaluation_interval < 1:
        raise ValueError("Target update and evaluation intervals must be positive")
    if config.evaluation_episodes < 1:
        raise ValueError("Evaluation episodes must be positive")
    if not (0.0 <= config.epsilon_start <= 1.0 and 0.0 <= config.epsilon_end <= 1.0):
        raise ValueError("Epsilon values must be in [0, 1]")
    if config.epsilon_start < config.epsilon_end:
        raise ValueError("epsilon_start must be >= epsilon_end")
    if not (0.0 <= config.gamma <= 1.0):
        raise ValueError("Gamma must be in [0, 1]")
    if config.learning_rate <= 0 or config.hidden_dim < 1 or config.gradient_clip <= 0:
        raise ValueError("Learning rate, hidden dimension and gradient clip must be positive")
    if config.torch_threads < 1:
        raise ValueError("torch_threads must be positive")


def set_seed(seed: int, torch_threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(torch_threads)


def append_csv(path: Path, fieldnames: Sequence[str], data: dict[str, Any]) -> None:
    first_write = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if first_write:
            writer.writeheader()
        writer.writerow(data)


TRAIN_COLUMNS = (
    "episode", "attacker", "reward", "success", "entered_honeynet",
    "moves", "lihs_placed", "epsilon", "average_loss_100", "global_steps",
)

EVAL_COLUMNS = (
    "episode", "policy", "attacker", "episodes", "success_rate",
    "entry_rate", "average_reward", "average_moves", "average_lihs_placed",
)


def evaluate_policy(
    agent: DQNAgent | None,
    config: DQNConfig,
    attackers: Sequence[str],
    episodes_per_attacker: int,
    seed_base: int = 1_000_000,
    random_baseline: bool = False,
) -> list[dict[str, Any]]:
    env = make_environment(config, attackers)
    rng = np.random.default_rng(seed_base + 51)
    results: list[dict[str, Any]] = []
    try:
        for variant_index, variant in enumerate(attackers):
            successes = 0
            entries = 0
            total_reward = 0.0
            moves = 0
            placements = 0
            for episode_index in range(episodes_per_attacker):
                episode_seed = seed_base + variant_index * 100_000 + episode_index
                observation, info = env.reset(
                    seed=episode_seed,
                    options={"attacker_variant": variant},
                )
                done = False
                while not done:
                    if random_baseline:
                        action = int(rng.choice(np.flatnonzero(info["action_mask"])))
                    else:
                        if agent is None:
                            raise ValueError("An agent is required for DQN evaluation")
                        action = agent.select_action(
                            observation, info["action_mask"], epsilon=0.0, rng=rng
                        )
                    observation, reward, terminated, truncated, info = env.step(action)
                    done = bool(terminated or truncated)
                successes += int(bool(info["ended_in_honeynet"]))
                entries += int(bool(info["entered_honeynet"]))
                total_reward += float(reward)
                moves += int(info["moves_used"])
                placements += int(info["placed_lihs"])
            n = episodes_per_attacker
            results.append(
                {
                    "attacker": variant,
                    "episodes": n,
                    "success_rate": successes / n,
                    "entry_rate": entries / n,
                    "average_reward": total_reward / n,
                    "average_moves": moves / n,
                    "average_lihs_placed": placements / n,
                }
            )
    finally:
        env.close()
    return results


def weighted_success_rate(rows: Sequence[dict[str, Any]]) -> float:
    n = sum(int(row["episodes"]) for row in rows)
    return sum(float(row["success_rate"]) * int(row["episodes"]) for row in rows) / max(1, n)


def report_evaluation(
    results: Sequence[dict[str, Any]],
    episode: int,
    policy_name: str,
    csv_path: Path | None = None,
) -> None:
    print(f"\n{policy_name} evaluation at episode {episode}")
    for result in results:
        print(
            f"  {result['attacker']:<39} "
            f"containment={result['success_rate']:.3f} "
            f"HIH_entry={result['entry_rate']:.3f} "
            f"LIHs={result['average_lihs_placed']:.2f}"
        )
        if csv_path is not None:
            append_csv(
                csv_path,
                EVAL_COLUMNS,
                {"episode": episode, "policy": policy_name, **result},
            )
    print(f"  Overall containment: {weighted_success_rate(results):.3f}")


def save_checkpoint(
    path: Path,
    agent: DQNAgent,
    config: DQNConfig,
    attackers: Sequence[str],
    episode: int,
    global_steps: int,
    validation_success: float,
) -> None:
    torch.save(
        {
            "policy_state_dict": agent.policy_net.state_dict(),
            "target_state_dict": agent.target_net.state_dict(),
            "optimizer_state_dict": agent.optimizer.state_dict(),
            "observation_dim": agent.observation_dim,
            "action_dim": agent.action_dim,
            "hidden_dim": config.hidden_dim,
            "config": asdict(config),
            "attacker_variants": list(attackers),
            "episode": episode,
            "global_steps": global_steps,
            "validation_success": validation_success,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    return checkpoint


def run_training(
    config: DQNConfig,
    attackers: Sequence[str],
    output_dir: Path,
    device: torch.device,
) -> tuple[DQNAgent, Path]:
    validate_config(config, attackers)
    set_seed(config.seed, config.torch_threads)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = make_environment(config, attackers)
    observation_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(env.action_space.n)
    agent = DQNAgent(observation_dim, action_dim, config, device)
    replay = ReplayBuffer(config.replay_capacity, observation_dim, action_dim)
    rng = np.random.default_rng(config.seed + 7)
    losses: list[float] = []
    recent_successes: list[int] = []
    global_steps = 0
    best_score = -1.0
    best_path = output_dir / "best_dqn.pt"
    final_path = output_dir / "final_dqn.pt"
    train_csv = output_dir / "training_metrics.csv"
    eval_csv = output_dir / "evaluation_metrics.csv"
    for old_log in (train_csv, eval_csv):
        if old_log.exists():
            old_log.unlink()

    print(
        f"Training DQN | device={device} | observation_dim={observation_dim} | "
        f"actions={action_dim} | horizon={config.max_steps} | "
        f"budget={config.lih_budget} | attacker_types={len(attackers)}"
    )
    try:
        attackers_this_round = list(attackers)
        for episode in range(1, config.episodes + 1):
            if (episode - 1) % len(attackers) == 0:
                rng.shuffle(attackers_this_round)
            variant = attackers_this_round[(episode - 1) % len(attackers)]
            observation, info = env.reset(
                seed=config.seed + episode,
                options={"attacker_variant": variant},
            )
            done = False
            episode_reward = 0.0
            epsilon = linear_epsilon(config, global_steps)
            while not done:
                epsilon = linear_epsilon(config, global_steps)
                action = agent.select_action(observation, info["action_mask"], epsilon, rng)
                next_observation, reward, terminated, truncated, next_info = env.step(action)
                done = bool(terminated or truncated)
                replay.add(
                    observation,
                    action,
                    reward,
                    next_observation,
                    done,
                    next_info["action_mask"],
                )
                episode_reward += float(reward)
                global_steps += 1
                if (
                    global_steps >= config.warmup_steps
                    and len(replay) >= config.batch_size
                    and global_steps % config.train_frequency == 0
                ):
                    loss = agent.update(replay, rng)
                    losses.append(loss)
                observation = next_observation
                info = next_info

            success = int(bool(info["ended_in_honeynet"]))
            recent_successes.append(success)
            if len(recent_successes) > 100:
                recent_successes.pop(0)
            append_csv(
                train_csv,
                TRAIN_COLUMNS,
                {
                    "episode": episode,
                    "attacker": variant,
                    "reward": episode_reward,
                    "success": success,
                    "entered_honeynet": int(bool(info["entered_honeynet"])),
                    "moves": info["moves_used"],
                    "lihs_placed": info["placed_lihs"],
                    "epsilon": round(epsilon, 6),
                    "average_loss_100": (
                        round(float(np.mean(losses[-100:])), 6) if losses else ""
                    ),
                    "global_steps": global_steps,
                },
            )

            if episode % 100 == 0 or episode == 1:
                print(
                    f"episode={episode:5d}/{config.episodes} "
                    f"steps={global_steps:6d} "
                    f"epsilon={epsilon:.3f} "
                    f"train_success_last100={np.mean(recent_successes):.3f} "
                    f"replay={len(replay)}"
                )

            if episode % config.evaluation_interval == 0 or episode == config.episodes:
                results = evaluate_policy(
                    agent,
                    config,
                    attackers,
                    config.evaluation_episodes,
                    seed_base=1_000_000 + config.seed,
                )
                report_evaluation(results, episode, "DQN", eval_csv)
                score = weighted_success_rate(results)
                if score > best_score:
                    best_score = score
                    save_checkpoint(
                        best_path, agent, config, attackers, episode, global_steps, score
                    )
                    print(f"  Saved best checkpoint: {best_path}")

        save_checkpoint(
            final_path, agent, config, attackers, config.episodes, global_steps, score
        )
        print(f"\nTraining complete. Best: {best_path}; Final: {final_path}")
    finally:
        env.close()
    return agent, best_path


def run_evaluation(
    checkpoint_path: Path,
    episodes_per_attacker: int,
    device: torch.device,
    output_dir: Path | None = None,
    random_network: bool = False,
) -> None:
    checkpoint = load_checkpoint(checkpoint_path, device)
    config = DQNConfig(**checkpoint["config"])
    if random_network:
        config.vary_network = True
    attackers = list(checkpoint["attacker_variants"])
    set_seed(config.seed, config.torch_threads)
    env = make_environment(config, attackers)
    try:
        if (
            int(np.prod(env.observation_space.shape)) != checkpoint["observation_dim"]
            or int(env.action_space.n) != checkpoint["action_dim"]
        ):
            raise ValueError("Checkpoint dimensions do not match the current environment")
    finally:
        env.close()

    agent = DQNAgent(
        int(checkpoint["observation_dim"]),
        int(checkpoint["action_dim"]),
        config,
        device,
    )
    agent.policy_net.load_state_dict(checkpoint["policy_state_dict"])
    agent.target_net.load_state_dict(checkpoint["target_state_dict"])
    agent.policy_net.eval()
    print(f"Loaded {checkpoint_path} (trained for {checkpoint['episode']} episodes)")
    if config.vary_network:
        print("Evaluation uses a different network realization for each evaluation seed")
    else:
        print(f"Evaluation uses fixed network seed {config.network_seed}")

    results = evaluate_policy(
        agent,
        config,
        attackers,
        episodes_per_attacker,
        seed_base=5_000_000 + config.seed,
    )
    baseline = evaluate_policy(
        None,
        config,
        attackers,
        episodes_per_attacker,
        seed_base=5_000_000 + config.seed,
        random_baseline=True,
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        eval_csv = output_dir / "heldout_evaluation.csv"
        if eval_csv.exists():
            eval_csv.unlink()
    else:
        eval_csv = None
    report_evaluation(results, checkpoint["episode"], "DQN", eval_csv)
    report_evaluation(baseline, checkpoint["episode"], "Random legal defender", eval_csv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Masked DQN defender training for the cyber deception MDP"
    )
    parser.add_argument("--mode", choices=("train", "evaluate"), default="train")
    parser.add_argument("--attackers", nargs="+", choices=list(ATTACKER_VARIANTS), default=list(ATTACKER_VARIANTS))
    parser.add_argument("--episodes", type=int, default=6000)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--network-seed", type=int, default=1000)
    parser.add_argument("--vary-network", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--replay-capacity", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--train-frequency", type=int, default=1)
    parser.add_argument("--target-update-frequency", type=int, default=250)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=15000)
    parser.add_argument("--evaluation-interval", type=int, default=500)
    parser.add_argument("--evaluation-episodes", type=int, default=25)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--loss-aversion-lambda", type=float, default=2.0)
    parser.add_argument("--controlled-investment", type=float, default=None)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("dqn_results"))
    parser.add_argument("--checkpoint", type=Path, default=Path("dqn_results/best_dqn.pt"))
    parser.add_argument("--eval-random-network", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    if args.mode == "evaluate":
        run_evaluation(
            args.checkpoint,
            args.evaluation_episodes,
            device,
            output_dir=args.output_dir,
            random_network=args.eval_random_network,
        )
        return
    config = DQNConfig(
        episodes=args.episodes,
        max_steps=args.max_steps,
        lih_budget=args.budget,
        network_seed=args.network_seed,
        vary_network=args.vary_network,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        replay_capacity=args.replay_capacity,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        train_frequency=args.train_frequency,
        target_update_frequency=args.target_update_frequency,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        evaluation_interval=args.evaluation_interval,
        evaluation_episodes=args.evaluation_episodes,
        gradient_clip=args.gradient_clip,
        loss_aversion_lambda=args.loss_aversion_lambda,
        controlled_investment=args.controlled_investment,
        torch_threads=args.torch_threads,
    )
    run_training(config, args.attackers, args.output_dir, device)


if __name__ == "__main__":
    main()
