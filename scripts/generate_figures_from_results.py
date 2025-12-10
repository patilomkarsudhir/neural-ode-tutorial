import os
import json

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint


plt.style.use("seaborn-v0_8-paper")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 9

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")
os.makedirs(FIG_DIR, exist_ok=True)


class TrueDynamics(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]], device=device)

    def forward(self, t, y):
        return torch.mm(y**3, self.A)


class ResidualODEFunc(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 2)
        self.activation = nn.Tanh()

        for m in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, t, y):
        y_cubed = y**3
        h = self.activation(self.fc1(y_cubed))
        h_res = self.activation(self.fc2(h))
        h = h + h_res
        return self.fc3(h)


def load_results_and_model():
    root = os.path.dirname(os.path.dirname(__file__))
    with open(os.path.join(root, "results", "neural_ode_results.json"), "r") as f:
        results = json.load(f)

    checkpoint = torch.load(
        os.path.join(root, "saved_models", "neural_ode_residual64_optimized.pth"),
        map_location="cpu",
    )

    device = torch.device("cpu")
    model = ResidualODEFunc(hidden_dim=checkpoint.get("hidden_dim", 64))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    true_dynamics = TrueDynamics(device=device)
    true_y0 = torch.tensor([[2.0, 0.0]], device=device)
    t = torch.linspace(0.0, 25.0, 1000, device=device)

    with torch.no_grad():
        true_y = odeint(true_dynamics, true_y0, t, method="dopri5", rtol=1e-5, atol=1e-7)
        pred_y_best = odeint(model, true_y0, t, method="dopri5", rtol=1e-5, atol=1e-7)

    return results, model, true_dynamics, true_y0, t, true_y, pred_y_best, device


def save_spiral_figures(true_y, t, true_y0):
    t_np = t.cpu().numpy()
    y_np = true_y.cpu().numpy()[:, 0, :]

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(t_np, y_np[:, 0], label="x(t)", linewidth=2)
    ax.plot(t_np, y_np[:, 1], label="y(t)", linewidth=2)
    ax.set_xlabel("Time")
    ax.set_ylabel("State")
    ax.set_title("True Dynamics: Time Series")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "spiral_timeseries.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(y_np[:, 0], y_np[:, 1], color="green", linewidth=2)
    ax.scatter(
        true_y0[0, 0].cpu(), true_y0[0, 1].cpu(), color="red", s=40, label="Initial Point"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("True Dynamics: Phase Portrait (Spiral)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "spiral_trajectory.pdf"))
    plt.close(fig)


def save_training_convergence(results):
    train = results["baseline_training"]
    loss_history = np.array(train["loss_history"], dtype=float)
    test_loss_history = np.array(train["test_loss_history"], dtype=float)
    test_freq = int(train["test_freq"])

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(loss_history, alpha=0.4, label="Training Loss (per batch)")

    window = 50
    if len(loss_history) >= window:
        smoothed = np.convolve(loss_history, np.ones(window) / window, mode="valid")
        ax.plot(
            np.arange(window - 1, len(loss_history)),
            smoothed,
            label=f"Smoothed Training Loss (window={window})",
            linewidth=2,
        )

    test_iters = np.arange(1, len(test_loss_history) + 1) * test_freq
    ax.plot(test_iters, test_loss_history, "ro-", label="Test Loss", linewidth=2, markersize=4)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Training Progress (Baseline)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "training_convergence.pdf"))
    plt.close(fig)


def save_timeseries_and_phase(true_y, pred_y_best, t, true_y0):
    t_np = t.cpu().numpy()
    true_np = true_y.cpu().numpy()[:, 0, :]
    pred_np = pred_y_best.cpu().numpy()[:, 0, :]

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(t_np, true_np[:, 0], "g-", label="True x(t)", linewidth=2)
    ax.plot(t_np, true_np[:, 1], "b-", label="True y(t)", linewidth=2)
    ax.plot(t_np, pred_np[:, 0], "g--", label="Pred x(t)", linewidth=2)
    ax.plot(t_np, pred_np[:, 1], "b--", label="Pred y(t)", linewidth=2)
    ax.set_xlabel("Time")
    ax.set_ylabel("State")
    ax.set_title("True vs Predicted Trajectories (Best Model)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "timeseries_comparison.pdf"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(true_np[:, 0], true_np[:, 1], "g-", label="True Trajectory", linewidth=2)
    ax.plot(pred_np[:, 0], pred_np[:, 1], "b--", label="Predicted Trajectory", linewidth=2)
    ax.scatter(
        true_y0[0, 0].cpu(), true_y0[0, 1].cpu(), color="red", s=40, marker="*", label="Initial Point"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Phase Portrait: True vs Best Model")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "phase_portrait_comparison.pdf"))
    plt.close(fig)


def save_error_analysis(true_dynamics, true_y0, model, device):
    t_ext = torch.linspace(0.0, 35.0, 1400, device=device)
    with torch.no_grad():
        true_y_ext = odeint(true_dynamics, true_y0, t_ext, method="dopri5", rtol=1e-5, atol=1e-7)
        pred_y_ext = odeint(model, true_y0, t_ext, method="dopri5", rtol=1e-5, atol=1e-7)

    err = torch.abs(pred_y_ext - true_y_ext).cpu().numpy()[:, 0, :]
    err_norm = np.sqrt(err[:, 0] ** 2 + err[:, 1] ** 2)
    t_np = t_ext.cpu().numpy()

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(t_np, err_norm, color="purple", linewidth=2)
    ax.axvline(25.0, color="red", linestyle="--", linewidth=1.5, label="Training End")
    ax.fill_between(
        t_np,
        err_norm.min(),
        err_norm.max(),
        where=(t_np >= 25.0),
        color="yellow",
        alpha=0.2,
        label="Extrapolation",
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Absolute Error")
    ax.set_title("Prediction Error Over Time (Best Model)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "error_analysis.pdf"))
    plt.close(fig)


def save_architecture_comparison(results):
    arch = results["architectures"]
    names = [a["name"] for a in arch]
    params = np.array([a["params"] for a in arch], dtype=float)
    losses = np.array([a["best_test_loss"] for a in arch], dtype=float)
    drifts = np.array([a["drift_ratio"] for a in arch], dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Left plot: scatter plot with smart label positioning to avoid overlap
    ax1.scatter(params, losses, c="tab:blue", edgecolors="black", s=50, zorder=5)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Number of Parameters")
    ax1.set_ylabel("Best Test Loss (MSE)")
    ax1.set_title("Architecture Performance")
    ax1.grid(True, which="both", alpha=0.3)

    # Manual annotation offsets to avoid overlapping labels
    # Format: (x_offset, y_offset, ha, va)
    label_offsets = {
        "Compact-32-L2": (8, 5, "left", "bottom"),
        "Compact-64-L3": (-60, 15, "left", "bottom"),
        "Compact-96-L3": (8, -12, "left", "top"),
        "Residual-64": (8, 8, "left", "bottom"),
        "TimeAware-64": (-70, -15, "left", "top"),
        "Residual-64 (Extended)": (8, 5, "left", "bottom"),
    }

    for name, p, l in zip(names, params, losses):
        offsets = label_offsets.get(name, (8, 0, "left", "center"))
        ax1.annotate(
            name,
            xy=(p, l),
            xytext=(offsets[0], offsets[1]),
            textcoords="offset points",
            fontsize=8,
            ha=offsets[2],
            va=offsets[3],
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.8),
            arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5, lw=0.5),
        )

    # Right plot: horizontal bar chart for drift ratios
    y_pos = np.arange(len(names))
    ax2.barh(y_pos, drifts, color="skyblue", edgecolor="steelblue")
    ax2.axvline(1.0, color="green", linestyle="--", linewidth=1.5, label="No Drift")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlabel("Drift Ratio (Late / Early Error)")
    ax2.set_title("Trajectory Drift Comparison")
    ax2.grid(True, axis="x", alpha=0.3)
    ax2.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "architecture_comparison.pdf"))
    plt.close(fig)


def save_parameter_efficiency(results):
    arch = results["architectures"]
    names = [a["name"] for a in arch]
    params = np.array([a["params"] for a in arch], dtype=float)
    losses = np.array([a["best_test_loss"] for a in arch], dtype=float)

    fig, ax = plt.subplots(figsize=(5, 3))

    colors = []
    sizes = []
    for name in names:
        if "Residual-64 (Extended)" in name:
            colors.append("tab:green")
            sizes.append(80)
        else:
            colors.append("tab:blue")
            sizes.append(40)

    ax.scatter(params, losses, c=colors, s=sizes, edgecolors="black", alpha=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Parameters")
    ax.set_ylabel("Best Test Loss (MSE)")
    ax.set_title("Parameter Efficiency")
    ax.grid(True, which="both", alpha=0.3)

    for p, l, name in zip(params, losses, names):
        if "Residual-64 (Extended)" in name:
            ax.annotate(
                name,
                (p, l),
                textcoords="offset points",
                xytext=(6, -10),
                fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.6),
            )

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "parameter_efficiency.pdf"))
    plt.close(fig)


def main() -> None:
    results, model, true_dynamics, true_y0, t, true_y, pred_y_best, device = load_results_and_model()

    save_spiral_figures(true_y, t, true_y0)
    save_training_convergence(results)
    save_timeseries_and_phase(true_y, pred_y_best, t, true_y0)
    save_error_analysis(true_dynamics, true_y0, model, device)
    save_architecture_comparison(results)
    save_parameter_efficiency(results)

    print("All figures written to:", os.path.abspath(FIG_DIR))


if __name__ == "__main__":
    main()
