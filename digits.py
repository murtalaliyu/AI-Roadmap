import random

SIZE = 16

DIGIT_TEMPLATES = {
    0: [
        "01110",
        "10001",
        "10001",
        "10001",
        "01110",
    ],
    1: [
        "00100",
        "01100",
        "00100",
        "00100",
        "01110",
    ],
    2: [
        "01110",
        "00001",
        "01110",
        "10000",
        "11111",
    ],
    3: [
        "11110",
        "00001",
        "00010",
        "00001",
        "11110",
    ],
    4: [
        "10001",
        "10001",
        "11111",
        "00001",
        "00001",
    ],
    5: [
        "11111",
        "10000",
        "11110",
        "00001",
        "11110",
    ],
    6: [
        "01110",
        "10000",
        "11110",
        "10001",
        "01110",
    ],
    7: [
        "11111",
        "00001",
        "00010",
        "00100",
        "00100",
    ],
    8: [
        "01110",
        "10001",
        "01110",
        "10001",
        "01110",
    ],
    9: [
        "01110",
        "10001",
        "11111",
        "00001",
        "01110",
    ],
}


def _expand_template(template):
    grid = [[0 for _ in range(SIZE)] for _ in range(SIZE)]
    for y in range(SIZE):
        for x in range(SIZE):
            sx = min(len(template[0]) - 1, int(x * len(template[0]) / SIZE))
            sy = min(len(template) - 1, int(y * len(template) / SIZE))
            grid[y][x] = 1 if template[sy][sx] == "1" else 0
    return grid


def _add_noise(grid, noise_level=0.02):
    noisy = [row[:] for row in grid]
    for y in range(SIZE):
        for x in range(SIZE):
            if random.random() < noise_level:
                noisy[y][x] = 1 - noisy[y][x]
    return noisy


def _shift(grid, max_shift=0):
    shifted = [[0 for _ in range(SIZE)] for _ in range(SIZE)]
    dx = random.randint(-max_shift, max_shift)
    dy = random.randint(-max_shift, max_shift)

    for y in range(SIZE):
        for x in range(SIZE):
            nx = x - dx
            ny = y - dy
            if 0 <= nx < SIZE and 0 <= ny < SIZE:
                shifted[y][x] = grid[ny][nx]
    return shifted


def _flatten(grid):
    return [pixel / 1.0 for row in grid for pixel in row]


def _build_digit_dataset(samples_per_digit, test_samples_per_digit):
    train = []
    test = []

    for digit, template in DIGIT_TEMPLATES.items():
        base = _expand_template(template)

        for _ in range(samples_per_digit):
            variant = _shift(_add_noise(base, noise_level=0.02), max_shift=0)
            train.append((_flatten(variant), digit))

        for _ in range(test_samples_per_digit):
            variant = _shift(_add_noise(base, noise_level=0.04), max_shift=0)
            test.append((_flatten(variant), digit))

    return train, test


train, test = _build_digit_dataset(50, 10)

__all__ = ["train", "test"]
