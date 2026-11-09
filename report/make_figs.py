import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 8.5, "legend.fontsize": 7.5, "legend.frameon": False})
BLUE, ORANGE, GREY = "#0072B2", "#D55E00", "#777777"

# ---- Figure: hard part captured, final 30-seed run (synth_final, tuned models)
ds = ["D1\nsimilarity", "D2\nmidpoint", "D3\ncentre"]
sa_g, dh_g = [35.5, 21.6, 32.3], [3.7, 9.8, 7.3]
sa_b, dh_b = [0.9, 16.8, 26.4], [-0.8, 8.1, 5.7]
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.3), sharey=True)
for ax, sa, dh, t in [(axs[0], sa_g, dh_g, "Gaussian kernel"), (axs[1], sa_b, dh_b, "Box kernel")]:
    x = np.arange(3); w = 0.36
    b1 = ax.bar(x - w/2, sa, w, color=BLUE, label="Self-attention (tuned)")
    b2 = ax.bar(x + w/2, dh, w, color=ORANGE, label="DeepHalo, DHQ (tuned)")
    for b, v in list(zip(b1, sa)) + list(zip(b2, dh)):
        ax.text(b.get_x() + b.get_width()/2, max(v, 0) + 0.8, f"{v}", ha="center", fontsize=7)
    ax.axhline(0, color="black", lw=0.8); ax.set_xticks(x); ax.set_xticklabels(ds); ax.set_title(t)
axs[0].set_ylabel("hard part captured (%)\n0 = mean+sd pool, 100 = oracle")
axs[1].legend(loc="upper left"); axs[0].set_ylim(-3, 40)
fig.tight_layout(); fig.savefig("figs/hard_part.pdf"); plt.close(fig)

# ---- Figure: training-size sweep (synth_size + synth_final)
n = np.array([11e3, 30e3, 100e3, 285e3, 1e6]); lab = ["11k", "30k", "100k", "285k", "1M"]
P = {"D2 (order 2, Gaussian)": ([-8.1, -2.5, 6.0, 21.6, 53.1], [-15.5, -12.8, 2.2, 9.8, 21.8]),
     "D3 (order 3, Gaussian)": ([-14.1, -1.8, 20.8, 32.3, 45.4], [-48.1, -24.2, -4.3, 7.3, 20.4]),
     "D2b (order 2, box)": ([-8.9, -2.9, 4.1, 16.8, 41.2], [-15.3, -12.4, 2.0, 8.1, 15.8]),
     "D3b (order 3, box)": ([-15.8, -1.2, 16.6, 26.4, 38.2], [-45.6, -24.5, -4.6, 5.7, 18.3])}
fig, axs = plt.subplots(1, 5, figsize=(6.6, 2.1))
ax = axs[0]
ax.plot(n, [70.0, 84.5, 93.0, 96.6, 98.5], "-o", color=BLUE, ms=3, label="self-attention")
ax.plot(n, [56.6, 79.4, 92.4, 96.8, 98.7], "-o", color=ORANGE, ms=3, label="DeepHalo (DHQ)")
ax.plot(n, [82.8, 89.2, 94.3, 96.8, 98.2], "--o", color=GREY, ms=3, label="mean+sd pool")
ax.set_title("D0 (LCL)"); ax.set_ylabel("% captured"); ax.set_ylim(50, 101)
for ax, (t, (sa, dh)) in zip(axs[1:], P.items()):
    ax.axhline(0, color=GREY, ls="--", lw=1)
    ax.plot(n, sa, "-o", color=BLUE, ms=3); ax.plot(n, dh, "-o", color=ORANGE, ms=3)
    ax.set_title(t.replace(" (", "\n("), fontsize=7.5); ax.set_ylim(-50, 60)
    if ax is not axs[1]: ax.set_yticklabels([])
for ax in axs:
    ax.set_xscale("log"); ax.set_xticks(n); ax.set_xticklabels(lab, fontsize=6.5, rotation=45); ax.minorticks_off()
axs[0].set_title("D0 (LCL)\ngap to twin closed", fontsize=7.5)
fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
fig.text(0.5, 0.13, "training slates (log scale)", ha="center", fontsize=7.5)
fig.tight_layout(rect=(0, 0.15, 1, 1)); fig.savefig("figs/size_sweep.pdf"); plt.close(fig)

# ---- Figure: depth sweep, full self-attention (synth_depth)
L = [1, 2, 3, 4]
full = {2: [13.6, 21.3, 23.9, 23.5], 3: [18.1, 30.5, 30.0, 32.2], 4: [21.6, 40.7, 42.1, 42.3], 5: [22.8, 40.6, 41.8, 42.1]}
soft = {2: [-15.4, -0.9, 3.2, 3.7], 3: [-78.6, -25.4, -5.0, 4.2], 4: [-213.6, -46.5, -33.8, -10.5], 5: [-348.8, -78.9, -57.5, -42.1]}
fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.2))
shades = {2: 0.35, 3: 0.55, 4: 0.75, 5: 1.0}
for k in full:
    axs[0].plot(L, full[k], "-o", color=BLUE, alpha=shades[k], ms=3, label=f"order {k}")
    axs[1].plot(L, soft[k], "-o", color=ORANGE, alpha=shades[k], ms=3, label=f"order {k}")
axs[0].set_title("Full self-attention (about 26k parameters)"); axs[1].set_title("Attention only, softmax (about 16k parameters)")
for ax in axs:
    ax.set_xticks(L); ax.set_xlabel("number of layers L"); ax.axhline(0, color=GREY, ls="--", lw=1)
axs[0].set_ylabel("hard part captured (%)"); axs[0].legend(ncol=2, loc="lower right"); axs[1].legend(ncol=2, loc="lower right")
fig.tight_layout(); fig.savefig("figs/depth.pdf"); plt.close(fig)
print("ok")
