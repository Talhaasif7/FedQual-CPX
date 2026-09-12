"""
Script to generate clean, high-resolution publication methodology diagrams for FedQual-CPX.
Generates:
1. architecture_overview.png: High-level abstract system flow of FedQual-CPX.
2. detection_and_adaptation_flow.png: Detailed sequential decision & change-point tracking module.
"""

from __future__ import annotations
import shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_architecture_overview(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Colors
    bg_server = "#f0f4f8"
    border_server = "#1f4e79"
    box_blue = "#d9e1f2"
    box_green = "#e2efda"
    box_orange = "#fce4d6"
    box_yellow = "#fff2cc"
    text_dark = "#1a1a1a"

    # Title Banner
    ax.text(50, 96, "FedQual-CPX System Architecture", ha="center", va="center",
            fontsize=14, fontweight="bold", color=border_server)

    # 1. Central Server Boundary Box
    server_rect = patches.FancyBboxPatch((4, 38), 92, 54, boxstyle="round,pad=1.5",
                                         facecolor=bg_server, edgecolor=border_server, linewidth=1.8, linestyle="--")
    ax.add_patch(server_rect)
    ax.text(8, 89, "Central Federated Server (Round t)", fontsize=11, fontweight="bold", color=border_server)

    # Server Components
    # Global Model
    ax.add_patch(patches.FancyBboxPatch((8, 68), 22, 14, boxstyle="round,pad=0.8",
                                        facecolor=box_blue, edgecolor="#2f5597", linewidth=1.4))
    ax.text(19, 76, "Global Model", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(19, 71, "w_t in R^d", ha="center", va="center", fontsize=9, style="italic")

    # Client History & Observation Mask
    ax.add_patch(patches.FancyBboxPatch((36, 68), 26, 14, boxstyle="round,pad=0.8",
                                        facecolor=box_yellow, edgecolor="#b28900", linewidth=1.4))
    ax.text(49, 76, "Observation Tracker", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(49, 71, "Missing-Value Handling\n(Unselected != 0)", ha="center", va="center", fontsize=8.5)

    # Sequential CUSUM + MAD
    ax.add_patch(patches.FancyBboxPatch((68, 68), 24, 14, boxstyle="round,pad=0.8",
                                        facecolor=box_orange, edgecolor="#c65911", linewidth=1.4))
    ax.text(80, 76, "Drift Tracking", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(80, 71, "CUSUM + Robust MAD\nGlobal Drift Rate D_t", ha="center", va="center", fontsize=8.5)

    # Dynamic Exploration Controller
    ax.add_patch(patches.FancyBboxPatch((36, 44), 26, 16, boxstyle="round,pad=0.8",
                                        facecolor=box_green, edgecolor="#385723", linewidth=1.4))
    ax.text(49, 54, "Adaptive Controller", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(49, 48, "Exploration eps_t in [0.05, 0.50]\nStaleness + Drift Rate", ha="center", va="center", fontsize=8.5)

    # Selection Cohorts
    ax.add_patch(patches.FancyBboxPatch((8, 44), 22, 16, boxstyle="round,pad=0.8",
                                        facecolor="#e7e6e6", edgecolor="#595959", linewidth=1.4))
    ax.text(19, 54, "Selection Cohort", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(19, 48, "K_exploit = (1-eps_t)K\nK_explore = eps_t*K", ha="center", va="center", fontsize=8.5)

    # 2. Edge Client Population (Bottom)
    client_rect = patches.FancyBboxPatch((4, 4), 92, 26, boxstyle="round,pad=1.5",
                                         facecolor="#fbfbfb", edgecolor="#7f7f7f", linewidth=1.5)
    ax.add_patch(client_rect)
    ax.text(8, 27, "Edge Client Population (N Clients, Non-Stationary Distributions)", fontsize=11, fontweight="bold", color="#333333")

    # Selected Clients
    ax.add_patch(patches.FancyBboxPatch((10, 8), 36, 14, boxstyle="round,pad=0.8",
                                        facecolor=box_green, edgecolor="#385723", linewidth=1.4))
    ax.text(28, 16, "Selected Clients (Cohort S_t, |S_t|=K)", ha="center", va="center", fontsize=9.5, fontweight="bold")
    ax.text(28, 11, "Local Training: w_{i,t} in R^d\nUtility: u_{i,t} = L(w_t) - L(w_{i,t})", ha="center", va="center", fontsize=8.5)

    # Unselected Clients
    ax.add_patch(patches.FancyBboxPatch((54, 8), 38, 14, boxstyle="round,pad=0.8",
                                        facecolor="#f2f2f2", edgecolor="#a6a6a6", linewidth=1.2, linestyle=":") )
    ax.text(73, 16, "Unselected Clients (N - K Clients)", ha="center", va="center", fontsize=9.5, color="#595959")
    ax.text(73, 11, "No observation (Utility is strictly missing)\nStaleness counter increments", ha="center", va="center", fontsize=8.5, color="#595959")

    # Connecting Arrows
    arrow_kw = dict(arrowstyle="->", lw=1.6, color="#1f4e79")
    # Downward: Broadcast
    ax.annotate("", xy=(19, 23), xytext=(19, 44), arrowprops=arrow_kw)
    ax.text(20.5, 33.5, "Broadcast w_t\n& Selection S_t", fontsize=8.5, color="#1f4e79", fontweight="bold")

    # Upward: Local Updates & Utility
    ax.annotate("", xy=(49, 44), xytext=(36, 23), arrowprops=dict(arrowstyle="->", lw=1.6, color="#385723"))
    ax.text(43.5, 32, "Upload w_{i,t}, u_{i,t}", fontsize=8.5, color="#385723", fontweight="bold")

    # Server Internal Connections
    ax.annotate("", xy=(68, 75), xytext=(62, 75), arrowprops=dict(arrowstyle="->", lw=1.3, color="#595959"))
    ax.annotate("", xy=(49, 68), xytext=(49, 60), arrowprops=dict(arrowstyle="<-", lw=1.3, color="#595959"))
    ax.annotate("", xy=(30, 52), xytext=(36, 52), arrowprops=dict(arrowstyle="<-", lw=1.3, color="#595959"))
    ax.annotate("", xy=(19, 68), xytext=(19, 60), arrowprops=dict(arrowstyle="<-", lw=1.3, color="#595959"))

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Diagram 1] Generated: {output_path}")


def draw_detection_and_adaptation_flow(output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    border_color = "#203764"
    box_blue = "#d9e1f2"
    box_green = "#e2efda"
    box_orange = "#fce4d6"
    box_yellow = "#fff2cc"

    ax.text(50, 95, "Sequential CUSUM Tracking and Adaptive Selection Flow", ha="center", va="center",
            fontsize=13, fontweight="bold", color=border_color)

    # Step 1: Raw Utility
    ax.add_patch(patches.FancyBboxPatch((4, 52), 18, 26, boxstyle="round,pad=0.8",
                                        facecolor=box_blue, edgecolor="#2f5597", linewidth=1.5))
    ax.text(13, 71, "Step 1: Utility", ha="center", va="center", fontsize=9.5, fontweight="bold")
    ax.text(13, 62, "Raw Loss Gain:\nu_{i,t} = L_global\n        - L_local", ha="center", va="center", fontsize=8.5)
    ax.text(13, 55, "u_{i,t} in R", ha="center", va="center", fontsize=8, style="italic")

    # Step 2: Robust Normalization
    ax.add_patch(patches.FancyBboxPatch((27, 52), 20, 26, boxstyle="round,pad=0.8",
                                        facecolor=box_yellow, edgecolor="#b28900", linewidth=1.5))
    ax.text(37, 71, "Step 2: Robust MAD", ha="center", va="center", fontsize=9.5, fontweight="bold")
    ax.text(37, 63, "Causal Median:\nmu_{i,t} = med(H_i)\nMAD_{i,t} = med(|u-mu|)", ha="center", va="center", fontsize=8.5)
    ax.text(37, 55, "Scaled & Clipped:\nu~_{i,t} in [-3, 3]", ha="center", va="center", fontsize=8.5, color="#7030a0", fontweight="bold")

    # Step 3: Sequential CUSUM
    ax.add_patch(patches.FancyBboxPatch((52, 52), 22, 26, boxstyle="round,pad=0.8",
                                        facecolor=box_orange, edgecolor="#c65911", linewidth=1.5))
    ax.text(63, 71, "Step 3: CUSUM Test", ha="center", va="center", fontsize=9.5, fontweight="bold")
    ax.text(63, 63, "S_t^+ = max(0, S^+ + u~ - d/2)\nS_t^- = max(0, S^- - u~ - d/2)", ha="center", va="center", fontsize=8.2)
    ax.text(63, 55, "Test: max(S^+, S^-) >= h", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#c00000")

    # Step 4: Decision & Bonus
    ax.add_patch(patches.FancyBboxPatch((79, 52), 17, 26, boxstyle="round,pad=0.8",
                                        facecolor=box_green, edgecolor="#385723", linewidth=1.5))
    ax.text(87.5, 71, "Step 4: Drift Flag", ha="center", va="center", fontsize=9.5, fontweight="bold")
    ax.text(87.5, 63, "Reset S_t = 0\nApply Bonus beta_c\nIncrement D_t", ha="center", va="center", fontsize=8.5)
    ax.text(87.5, 55, "Alert Triggered", ha="center", va="center", fontsize=8, color="#385723", fontweight="bold")

    # Step 5: Adaptive Exploration Feedback (Bottom Box)
    ax.add_patch(patches.FancyBboxPatch((20, 10), 60, 26, boxstyle="round,pad=1.0",
                                        facecolor="#f2f4f7", edgecolor="#203764", linewidth=1.6))
    ax.text(50, 30, "Step 5: Dynamic Exploration Controller (Global Feedback)", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#203764")
    ax.text(50, 22, "eps_t = clip(eps_base + gamma_d * D_t + gamma_u * U_t, 0.05, 0.50)", ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="#1f4e79")
    ax.text(50, 15, "Balances Exploitation (K_exploit with bonus beta_c) & Exploration (K_explore by staleness/uncertainty)",
            ha="center", va="center", fontsize=8.5, color="#333333")

    # Horizontal Flow Arrows
    arrow_kw = dict(arrowstyle="->", lw=1.6, color="#203764")
    ax.annotate("", xy=(27, 65), xytext=(22, 65), arrowprops=arrow_kw)
    ax.annotate("", xy=(52, 65), xytext=(47, 65), arrowprops=arrow_kw)
    ax.annotate("", xy=(79, 65), xytext=(74, 65), arrowprops=arrow_kw)

    # Feedback Arrows to Bottom
    ax.annotate("", xy=(70, 36), xytext=(87.5, 52),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#c65911", connectionstyle="arc3,rad=0.2"))
    ax.annotate("", xy=(13, 52), xytext=(30, 36),
                arrowprops=dict(arrowstyle="<-", lw=1.5, color="#385723", connectionstyle="arc3,rad=0.2"))

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Diagram 2] Generated: {output_path}")


def main() -> None:
    results_dir = Path("results/figures")
    paper_dir = Path("paper/figures")

    fig1_path = results_dir / "architecture_overview.png"
    fig2_path = results_dir / "detection_and_adaptation_flow.png"

    draw_architecture_overview(fig1_path)
    draw_detection_and_adaptation_flow(fig2_path)

    # Sync to paper/figures
    shutil.copy2(fig1_path, paper_dir / fig1_path.name)
    shutil.copy2(fig2_path, paper_dir / fig2_path.name)
    print(f"[Sync] Copied diagrams to {paper_dir}/")


if __name__ == "__main__":
    main()
