import json
from pathlib import Path

import numpy as np

with (Path(__file__).parent / "colors.json").open(encoding="utf-8") as file:
    color_data = json.load(file)

samples_per_group = color_data["samples_per_group"]
colors = []
labels = []

for label, color_range in color_data["groups"].items():
    shades = np.linspace(
        color_range["start_rgb"],
        color_range["end_rgb"],
        samples_per_group,
        dtype=np.float32,
    )
    colors.append(shades)
    labels.extend([label] * samples_per_group)

X = np.concatenate(colors, axis=0)
labels = np.array(labels)

__all__ = ["X", "labels"]
