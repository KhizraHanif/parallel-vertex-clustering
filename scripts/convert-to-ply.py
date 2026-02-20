# --------------------------------------------------------------
#  Figure 1.2 – Vertex Clustering (CORRECTED to match new image)
# --------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

# ------------------------------------------------------------------
# 1. Original vertices (exactly as in the new figure)
# ------------------------------------------------------------------
pts = np.array([
    [0.0, 0.0],    # 1
    [0.8, -0.8],   # 2
    [3.0, 0.0],    # 3
    [1.5, 0.2],    # 4
    [1.2, 1.8]     # 5  ← outside both clusters
])
labels = ['1', '2', '3', '4', '5']

# ------------------------------------------------------------------
# 2. Cluster representatives (red circles) – manually placed
# ------------------------------------------------------------------
left_rep  = np.array([1.0, 0.0])   # rep for {1,2,4}
right_rep = np.array([2.5, 0.0])   # rep for {3}

reps = [left_rep, right_rep]
rep_labels = ['A', 'B']  # optional

# ε = radius of clustering cells
epsilon = 1.2

# ------------------------------------------------------------------
# 3. Assign vertices to clusters (distance ≤ ε from rep)
# ------------------------------------------------------------------
clusters = {0: [], 1: []}  # 0: left, 1: right
for i, p in enumerate(pts):
    dist_left  = np.linalg.norm(p - left_rep)
    dist_right = np.linalg.norm(p - right_rep)
    if dist_left <= epsilon:
        clusters[0].append(i)
    if dist_right <= epsilon:
        clusters[1].append(i)

# Note: point 5 is in neither → correctly excluded

# ------------------------------------------------------------------
# 4. Plot
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4))

# --- Original points (black) ---
ax.scatter(pts[:, 0], pts[:, 1], c='black', s=80, zorder=5)
for i, (x, y) in enumerate(pts):
    ax.text(x, y - 0.18, labels[i], ha='center', va='top', fontsize=12, fontweight='bold')

# --- Cluster representatives (red with black edge) ---
ax.scatter(*left_rep,  c='red', s=120, edgecolors='black', linewidth=1.5, zorder=6)
ax.scatter(*right_rep, c='red', s=120, edgecolors='black', linewidth=1.5, zorder=6)

# --- Dashed red circles (radius = ε) centered on representatives ---
ax.add_patch(Circle(left_rep,  epsilon, color='red', ls='--', fill=False, lw=1.5))
ax.add_patch(Circle(right_rep, epsilon, color='red', ls='--', fill=False, lw=1.5))

# --- Black arrows: original vertex → its representative ---
arrowprops = dict(arrowstyle='->', lw=1.2, color='black')

# Left cluster: 1,2,4 → left_rep
for i in clusters[0]:
    if i != 4:  # avoid overlap
        ax.annotate('', xy=left_rep, xytext=pts[i],
                    arrowprops=arrowprops, zorder=4)

# Right cluster: 3 → right_rep
for i in clusters[1]:
    ax.annotate('', xy=right_rep, xytext=pts[i],
                arrowprops=arrowprops, zorder=4)

# --- ε arrows (from rep outward, horizontal) ---
ax.annotate('', xy=(left_rep[0] - epsilon, left_rep[1]),
            xytext=left_rep, arrowprops=dict(arrowstyle='<->', lw=1.2, color='black'))
ax.annotate('', xy=(right_rep[0] + epsilon, right_rep[1]),
            xytext=right_rep, arrowprops=dict(arrowstyle='<->', lw=1.2, color='black'))

# --- ε labels ---
ax.text(left_rep[0] - epsilon - 0.1, left_rep[1], r'$\varepsilon$', 
        fontsize=14, color='black', ha='right', va='center')
ax.text(right_rep[0] + epsilon + 0.1, right_rep[1], r'$\varepsilon$', 
        fontsize=14, color='black', ha='left', va='center')

# --- Layout ---
ax.set_xlim(-0.8, 4.2)
ax.set_ylim(-1.5, 2.2)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout()
plt.savefig("vertex_clustering_corrected.png", dpi=500, bbox_inches='tight')
plt.show()