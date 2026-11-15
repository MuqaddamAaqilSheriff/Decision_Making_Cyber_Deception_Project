# Cyber Deception Gymnasium Environment

Files:
- `Cyber_Deception_Attackers_and_Network.py` — original attacker/graph source supplied for integration (unchanged).
- `cyber_deception_env.py` — new defender-side Gymnasium MDP wrapper, not a DQN agent.

Use Python 3.11 or 3.12 if your machine has difficulty installing Gymnasium or later PyTorch.

Install:

```bash
python -m pip install gymnasium numpy networkx
```

Run a demo:

```bash
python cyber_deception_env.py
```

Example:

```python
import numpy as np
from cyber_deception_env import CyberDeceptionEnv

env = CyberDeceptionEnv(lih_budget=2, max_steps=20)
state, info = env.reset(seed=42)
done = False
while not done:
    legal_actions = np.flatnonzero(info["action_mask"])
    action = int(env.np_random.choice(legal_actions))
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    state = next_state
print("Reward:", reward)
print("Finished in HIH:", info["ended_in_honeynet"])
```

Action 0..15 corresponds to LIH placements at (layer 1..4, path 1..4), in layer-major order. Action 16 means Wait. Use the action mask for epsilon-greedy action selection AND for the DQN next-state maximum target. Invalid actions raise `ValueError`.

`step` calls the legacy attacker once, advances one attacker movement, and activates newly placed LIHs only after that movement. If the attacker reaches the pending LIH's location during its inactive movement, the deployment is consumed but cannot retroactively intercept; it is reported in `wasted_lihs_this_step`.

The environment returns a 459-feature float32 vector under the default 5x4 network, with network outcomes, investments, node beliefs, LIH status, visited nodes, attacker variant and current state. This deliberately assumes the defender knows the attacker type and internal decision variables to keep the project in a fully-observable MDP setting. If hidden in the intended application, the formal problem becomes partially observable.

Reward is `+1` at termination if the attacker ends inside the HIH after entering H1 or deeper, and `0` otherwise. `terminated` means the attacker reached a game endpoint; `truncated` means the 20-movement horizon was reached. Gymnasium returns `(obs, reward, terminated, truncated, info)`.

The attacker module's functions use Python global `random`. The environment isolates their RNG state per episode during attacker calculations. The full Gymnasium checker has not been run in the generation environment because Gymnasium was unavailable there; source compilation and multi-variant functional simulations were performed with a small API stand-in.
