import numpy as np

def _make_spirals(samples_per_class, rng):
    radius = np.linspace(0.05, 1.0, samples_per_class)
    angle = 4 * np.pi * radius
    noise = rng.normal(0, 0.02, (2, samples_per_class, 2))

    arms = []
    labels = []
    for class_index, offset in enumerate((0.0, np.pi)):
        points_angle = angle + offset
        points = np.column_stack(
            (radius * np.cos(points_angle), radius * np.sin(points_angle))
        )
        arms.append(points + noise[class_index])
        labels.append(np.full((samples_per_class, 1), 1 if class_index else -1))

    features = np.concatenate(arms)
    targets = np.concatenate(labels)
    order = rng.permutation(len(features))
    features = features[order]
    targets = targets[order]
    features = np.column_stack((features, np.ones(len(features))))
    return features, targets


_rng = np.random.default_rng(17)
X, y = _make_spirals(100, _rng)
new, truth = _make_spirals(100, _rng)
print("new:", new, "\ntruth:", truth)
