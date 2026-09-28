"""Train a tabular Q-learning agent on GridWorld and compare it to a random baseline.

Q-learning update:
    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

Run:  python train.py
Outputs: learning_curve.png, policy.png and a printed evaluation.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gridworld import ARROWS, GridWorld


def train(env: GridWorld, episodes=3000, alpha=0.1, gamma=0.95, eps_start=1.0, eps_end=0.05, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.n_states, env.n_actions))
    returns = []
    for ep in range(episodes):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / (0.6 * episodes))  # linear decay
        s, done, total = env.reset(), False, 0.0
        while not done:
            a = int(rng.integers(env.n_actions)) if rng.random() < eps else int(np.argmax(Q[s]))
            s2, r, done = env.step(a)
            target = r if done else r + gamma * np.max(Q[s2])
            Q[s, a] += alpha * (target - Q[s, a])
            s, total = s2, total + r
        returns.append(total)
    return Q, np.array(returns)


def evaluate(env: GridWorld, policy, episodes=500) -> dict:
    rewards, successes, steps = [], 0, []
    for _ in range(episodes):
        s, done, total = env.reset(), False, 0.0
        while not done:
            s, r, done = env.step(policy(s))
            total += r
        rewards.append(total); steps.append(env.steps)
        successes += env.pos == env.goal
    return {"avg_reward": np.mean(rewards), "success_rate": successes / episodes, "avg_steps": np.mean(steps)}


def plot_learning_curve(returns, path="learning_curve.png", window=100):
    smooth = np.convolve(returns, np.ones(window) / window, mode="valid")
    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)
    ax.plot(returns, color="#cbd5e1", linewidth=0.6, label="Episode return")
    ax.plot(range(window - 1, len(returns)), smooth, color="#2563eb", linewidth=2, label=f"{window}-episode average")
    ax.set_xlabel("Episode"); ax.set_ylabel("Total reward")
    ax.set_title("Q-learning: reward per episode", loc="left")
    ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_policy(env: GridWorld, Q, path="policy.png"):
    fig, ax = plt.subplots(figsize=(env.w * 0.7, env.h * 0.7), dpi=120)
    colors = {"#": "#334155", "X": "#ef4444", "G": "#22c55e", "S": "#60a5fa"}
    for r in range(env.h):
        for c in range(env.w):
            cell = env.grid[r][c]
            ax.add_patch(plt.Rectangle((c, env.h - 1 - r), 1, 1, facecolor=colors.get(cell, "#f8fafc"), edgecolor="#cbd5e1"))
            label = cell if cell in "XGS#" and cell != "#" else ""
            if cell in ".S":
                label = ARROWS[int(np.argmax(Q[env.to_state((r, c))]))] if cell == "." else "S"
            ax.text(c + 0.5, env.h - 0.5 - r, label, ha="center", va="center", fontsize=13,
                    color="white" if cell in "XG" else "#0f172a")
    ax.set_xlim(0, env.w); ax.set_ylim(0, env.h); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Learned policy (greedy action per cell)", loc="left")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


if __name__ == "__main__":
    env = GridWorld(seed=0)
    Q, returns = train(env)
    rng = np.random.default_rng(1)
    random_stats = evaluate(GridWorld(seed=1), lambda s: int(rng.integers(4)))
    agent_stats = evaluate(GridWorld(seed=2), lambda s: int(np.argmax(Q[s])))
    print(f"{'Policy':12} {'Success rate':>13} {'Avg reward':>11} {'Avg steps':>10}")
    for name, st in [("Random", random_stats), ("Q-learning", agent_stats)]:
        print(f"{name:12} {st['success_rate']:>12.0%} {st['avg_reward']:>11.1f} {st['avg_steps']:>10.1f}")
    plot_learning_curve(returns)
    plot_policy(env, Q)
    print("Saved learning_curve.png and policy.png")
