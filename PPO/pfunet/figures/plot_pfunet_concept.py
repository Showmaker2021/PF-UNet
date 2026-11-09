import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.ndimage import gaussian_filter

# Publication styling conventions from figures4papers
PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "blue_light": "#EBF2FA",
    "green_1": "#E6F4EA",
    "green_2": "#AADCA9",
    "green_3": "#34A853",
    "red_strong": "#C5221F",
    "red_light": "#FCE8E6",
    "neutral_dark": "#202124",
    "neutral_light": "#F8F9FA",
    "neutral_border": "#DADCE0",
    "accent_orange": "#F2994A",
    "teal": "#12B5CB",
    "purple": "#785EF0",
}

def apply_style():
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 13
    plt.rcParams['axes.linewidth'] = 2.0
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['svg.fonttype'] = 'none'

def plot_pfunet_overall_concept(save_path='figures/pfunet_overall_concept.png'):
    apply_style()
    
    # 4-column layout: State -> Splatting (Priority Field) -> U-Net Reasoning -> Action Readout
    fig, axes = plt.subplots(1, 4, figsize=(28, 6.2))
    plt.subplots_adjust(left=0.04, right=0.98, top=0.86, bottom=0.15, wspace=0.42)
    
    ax1, ax2, ax3, ax4 = axes
    
    # -------------------------------------------------------------
    # (a) Discrete State -> 2D Coordinates (ST, W)
    # -------------------------------------------------------------
    np.random.seed(42)
    st_vals = np.array([0.15, 0.25, 0.45, 0.55, 0.70, 0.82])
    w_vals  = np.array([0.85, 0.65, 0.70, 0.40, 0.50, 0.20])
    durations = np.array([35, 55, 45, 25, 40, 30])
    labels = [r"$O_{1,1}$", r"$O_{2,1}$", r"$O_{1,2}$", r"$O_{3,1}$", r"$O_{2,2}$", r"$O_{3,2}$"]
    colors = [PALETTE["blue_main"], PALETTE["accent_orange"], PALETTE["blue_main"],
              PALETTE["teal"], PALETTE["accent_orange"], PALETTE["teal"]]
    
    # Precedence arrows between same job
    ax1.annotate("", xy=(st_vals[2], w_vals[2]), xytext=(st_vals[0], w_vals[0]),
                 arrowprops=dict(arrowstyle="->", color=PALETTE["blue_main"], lw=2.0, ls="--"))
    ax1.annotate("", xy=(st_vals[4], w_vals[4]), xytext=(st_vals[1], w_vals[1]),
                 arrowprops=dict(arrowstyle="->", color=PALETTE["accent_orange"], lw=2.0, ls="--"))
    ax1.annotate("", xy=(st_vals[5], w_vals[5]), xytext=(st_vals[3], w_vals[3]),
                 arrowprops=dict(arrowstyle="->", color=PALETTE["teal"], lw=2.0, ls="--"))

    for x, y, dur, lbl, c in zip(st_vals, w_vals, durations, labels, colors):
        ax1.scatter(x, y, s=dur*16, color=c, edgecolors='black', linewidth=1.8, zorder=5, alpha=0.9)
        ax1.text(x, y + 0.05, lbl, fontsize=12, fontweight='bold', ha='center', va='bottom', zorder=6)
    
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xlabel(r"Earliest Start Time $ST_{j,i}$ (norm)", fontsize=14, fontweight='bold', labelpad=8)
    ax1.set_ylabel(r"Work Remaining $W_{j,i}$ (norm)", fontsize=14, fontweight='bold', labelpad=8)
    ax1.set_title("(a) State in Continuous $\mathbb{R}^2$", fontsize=15, fontweight='bold', pad=14)
    
    # Annotation box
    ax1.text(0.05, 0.08, "Operations mapped to\n$(ST, W)$ coordinates\nNode size $\propto$ Duration", 
             fontsize=11, style='italic', bbox=dict(boxstyle='round,pad=0.5', facecolor=PALETTE["neutral_light"], edgecolor=PALETTE["neutral_border"]))

    # -------------------------------------------------------------
    # (b) Continuous Priority Density Field (Bilinear Splatting)
    # -------------------------------------------------------------
    grid_size = 64
    density_map = np.zeros((grid_size, grid_size))
    for x, y, dur in zip(st_vals, w_vals, durations):
        gx = int(np.clip(x * (grid_size - 1), 0, grid_size - 1))
        gy = int(np.clip(y * (grid_size - 1), 0, grid_size - 1))
        density_map[gy, gx] += dur
        
    smoothed_field = gaussian_filter(density_map, sigma=4.2)
    smoothed_field /= (smoothed_field.max() + 1e-8)
    
    im2 = ax2.imshow(smoothed_field, origin='lower', extent=[0, 1, 0, 1], 
                     cmap='Blues', aspect='auto', interpolation='bilinear')
    
    ax2.contour(np.linspace(0, 1, grid_size), np.linspace(0, 1, grid_size), smoothed_field, 
                levels=5, colors=PALETTE["blue_secondary"], alpha=0.5, linewidths=1.4)
    
    # Query candidate markers
    for x, y in zip(st_vals[:3], w_vals[:3]):
        ax2.scatter(x, y, s=55, color='white', edgecolors=PALETTE["red_strong"], linewidth=2.0, zorder=6)
        
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel(r"Start Time Dimension ($X$)", fontsize=14, fontweight='bold', labelpad=8)
    ax2.set_ylabel(r"Work Dimension ($Y$)", fontsize=14, fontweight='bold', labelpad=8)
    ax2.set_title("(b) 2D Priority Field Canvas", fontsize=15, fontweight='bold', pad=14)
    
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("Continuous Density Field", fontsize=11, fontweight='bold')
    cbar2.ax.tick_params(labelsize=10)
    
    # -------------------------------------------------------------
    # (c) Multi-Scale Spatial Reasoning (U-Net + Machine Attention)
    # -------------------------------------------------------------
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')
    ax3.set_title("(c) U-Net Spatial Reasoning", fontsize=15, fontweight='bold', pad=14)
    
    # Encoder 1
    enc1 = FancyBboxPatch((0.5, 6.5), 1.6, 2.5, boxstyle="round,pad=0.2",
                          facecolor=PALETTE["blue_light"], edgecolor=PALETTE["blue_main"], linewidth=2)
    ax3.add_patch(enc1)
    ax3.text(1.3, 7.75, "ConvBlock\n64x64", ha='center', va='center', fontsize=11, fontweight='bold', color=PALETTE["blue_main"])

    # Down 1
    ax3.annotate("", xy=(2.7, 5.8), xytext=(2.2, 7.0),
                 arrowprops=dict(arrowstyle="->", color=PALETTE["red_strong"], lw=2.2))
    ax3.text(2.6, 6.6, "/2 Pool", fontsize=10, fontweight='bold', color=PALETTE["red_strong"])

    # Encoder 2
    enc2 = FancyBboxPatch((2.7, 4.0), 1.6, 2.2, boxstyle="round,pad=0.2",
                          facecolor=PALETTE["blue_light"], edgecolor=PALETTE["blue_main"], linewidth=2)
    ax3.add_patch(enc2)
    ax3.text(3.5, 5.1, "ConvBlock\n32x32", ha='center', va='center', fontsize=11, fontweight='bold', color=PALETTE["blue_main"])

    # Bottleneck + Machine Cross-Attention
    bneck = FancyBboxPatch((4.0, 1.2), 2.0, 2.2, boxstyle="round,pad=0.2",
                           facecolor=PALETTE["green_1"], edgecolor=PALETTE["green_3"], linewidth=2.5)
    ax3.add_patch(bneck)
    ax3.text(5.0, 2.3, "Bottleneck +\nMachine\nAttention", ha='center', va='center', fontsize=11, fontweight='bold', color=PALETTE["green_3"])

    ax3.annotate("", xy=(4.0, 2.5), xytext=(3.8, 3.8),
                 arrowprops=dict(arrowstyle="->", color=PALETTE["red_strong"], lw=2.2))

    # Decoder 2
    dec2 = FancyBboxPatch((5.7, 4.0), 1.6, 2.2, boxstyle="round,pad=0.2",
                          facecolor=PALETTE["blue_light"], edgecolor=PALETTE["blue_secondary"], linewidth=2)
    ax3.add_patch(dec2)
    ax3.text(6.5, 5.1, "UpBlock\n32x32", ha='center', va='center', fontsize=11, fontweight='bold', color=PALETTE["blue_secondary"])

    ax3.annotate("", xy=(6.2, 3.8), xytext=(5.7, 2.8),
                 arrowprops=dict(arrowstyle="->", color=PALETTE["green_3"], lw=2.2))

    # Skip Connection
    ax3.annotate("", xy=(5.6, 5.1), xytext=(4.4, 5.1),
                 arrowprops=dict(arrowstyle="->", color=PALETTE["accent_orange"], lw=2.2, ls="--"))
    ax3.text(5.0, 5.4, "Skip Conn", ha='center', va='bottom', fontsize=10, fontweight='bold', color=PALETTE["accent_orange"])

    # Decoder 1
    dec1 = FancyBboxPatch((7.8, 6.5), 1.6, 2.5, boxstyle="round,pad=0.2",
                          facecolor=PALETTE["blue_light"], edgecolor=PALETTE["blue_secondary"], linewidth=2)
    ax3.add_patch(dec1)
    ax3.text(8.6, 7.75, "UpBlock\n64x64\n(Field)", ha='center', va='center', fontsize=11, fontweight='bold', color=PALETTE["blue_secondary"])

    ax3.annotate("", xy=(7.7, 7.0), xytext=(7.4, 5.8),
                 arrowprops=dict(arrowstyle="->", color=PALETTE["green_3"], lw=2.2))

    ax3.text(5.0, 0.4, r"Receptive field scales with $O(M \cdot H W)$ vs $O(N^2)$ Graph", 
             ha='center', va='center', fontsize=11, style='italic',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFFBEA', edgecolor=PALETTE["accent_orange"]))

    # -------------------------------------------------------------
    # (d) Continuous-to-Discrete Readout & Action Selection
    # -------------------------------------------------------------
    candidates = [r"$(O_{1,1}, M_1)$", r"$(O_{1,1}, M_2)$", r"$(O_{2,1}, M_1)$", r"$(O_{3,1}, M_3)$"]
    probs = [0.68, 0.14, 0.12, 0.06]
    bar_colors = [PALETTE["blue_main"], PALETTE["blue_secondary"], PALETTE["neutral_dark"], PALETTE["neutral_border"]]
    
    bars = ax4.bar(range(len(candidates)), probs, color=bar_colors, edgecolor='black', linewidth=1.8, width=0.55)
    
    for bar, p in zip(bars, probs):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2.0, height + 0.02, f"{p*100:.1f}%",
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax4.annotate("Selected Action\n(Bilinear Readout)", xy=(0, 0.70), xytext=(1.0, 0.86),
                 arrowprops=dict(arrowstyle="->", color=PALETTE["red_strong"], lw=2.0),
                 fontsize=11, fontweight='bold', color=PALETTE["red_strong"])

    ax4.set_xticks(range(len(candidates)))
    ax4.set_xticklabels(candidates, fontsize=11, fontweight='bold', rotation=18)
    ax4.set_ylim(0, 1.05)
    ax4.set_ylabel(r"Policy Probability $\pi(a \mid s)$", fontsize=14, fontweight='bold', labelpad=8)
    ax4.set_title(r"(d) Bilinear Readout $\to$ Action", fontsize=15, fontweight='bold', pad=14)

    # -------------------------------------------------------------
    # Inter-panel Flow Arrows using precise subplot bbox coordinates
    # -------------------------------------------------------------
    fig.canvas.draw()
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    cbar_pos = cbar2.ax.get_position()
    pos3 = ax3.get_position()
    pos4 = ax4.get_position()

    arrow_props = dict(arrowstyle="-|>", mutation_scale=16, color=PALETTE["blue_main"], lw=2.2)
    
    # Between (a) and (b)
    x1_mid = 0.5 * (pos1.x1 + pos2.x0)
    fig.add_artist(FancyArrowPatch((x1_mid - 0.018, 0.50), (x1_mid + 0.018, 0.50), transform=fig.transFigure, **arrow_props))
    fig.text(x1_mid, 0.53, "Bilinear\nSplatting", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=PALETTE["blue_main"])

    # Between (b) [after colorbar] and (c)
    x2_mid = 0.5 * (cbar_pos.x1 + pos3.x0)
    fig.add_artist(FancyArrowPatch((x2_mid - 0.018, 0.50), (x2_mid + 0.018, 0.50), transform=fig.transFigure, **arrow_props))
    fig.text(x2_mid, 0.53, "Continuous\nField $[M,H,W]$", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=PALETTE["blue_main"])

    # Between (c) and (d)
    x3_mid = 0.5 * (pos3.x1 + pos4.x0)
    fig.add_artist(FancyArrowPatch((x3_mid - 0.018, 0.50), (x3_mid + 0.018, 0.50), transform=fig.transFigure, **arrow_props))
    fig.text(x3_mid, 0.53, "Bilinear\nQuerying", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=PALETTE["blue_main"])

    # -------------------------------------------------------------
    # Export PNG & PDF
    # -------------------------------------------------------------
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=300)
    pdf_path = save_path.replace('.png', '.pdf')
    fig.savefig(pdf_path, dpi=300)
    plt.close(fig)
    print(f"Successfully generated:\n- PNG: {save_path}\n- PDF: {pdf_path}")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_png = os.path.join(script_dir, 'figures', 'pfunet_overall_concept.png')
    plot_pfunet_overall_concept(output_png)
