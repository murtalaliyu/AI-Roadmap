import numpy as np
from photos import X                        # 600 photos
import matplotlib.pyplot as plt

k = 3                                       # number of groups
rng = np.random.default_rng(17)
best = np.inf                               # no try yet
cluster_colors = np.array(["#e45756", "#54a24b", "#4c78a8"])
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")


def show_clusters(g, C, title):
    ax.clear()
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=cluster_colors[g], s=12)
    ax.scatter(
        C[:, 0], C[:, 1], C[:, 2],
        c=cluster_colors, marker="X", edgecolors="black", s=180,
    )
    ax.set(xlim=(0, 255), ylim=(0, 255), zlim=(0, 255))
    ax.set_xlabel("Red")
    ax.set_ylabel("Green")
    ax.set_zlabel("Blue")
    ax.set_title(title)
    fig.canvas.draw_idle()
    plt.pause(0.05)

for start in range(10):                     # ten random starts
    C = X[rng.choice(len(X), k, replace=False)]
    iteration = 0
    while True:
        iteration += 1
        old = C.copy()                      # remember centers
        d = ((X[:, None] - C) ** 2).sum(2)
        g = d.argmin(1)                     # nearest center
        
        for j in range(k):
            C[j] = X[g == j].mean(0)
        show_clusters(g, C, f"Start {start + 1}/10, iteration {iteration}")
        if (C == old).all():
            break

    s = d.min(1).sum()                      # total spread
    if s < best:
        best, keep, best_C = s, g.copy(), C.copy()

show_clusters(keep, best_C, "Best clustering")
plt.ioff()
print(np.bincount(keep))                    # photos per group
plt.show()

while True:
    try:
        rgb = np.array(
            [float(channel.strip()) for channel in input("Enter RGB values (R,G,B; 0-255): ").split(",")]
        )
        if rgb.shape != (3,) or not np.isfinite(rgb).all():
            raise ValueError
        if ((rgb < 0) | (rgb > 255)).any():
            raise ValueError
        break
    except ValueError:
        print("Please enter exactly three numbers from 0 to 255, separated by commas.")

distances = ((best_C - rgb) ** 2).sum(axis=1)
cluster = distances.argmin()
group_names = np.array(["red", "green", "blue"])
group = group_names[best_C[cluster].argmax()]
print(f"RGB {rgb.tolist()} belongs to the {group} group (cluster {cluster}).")
