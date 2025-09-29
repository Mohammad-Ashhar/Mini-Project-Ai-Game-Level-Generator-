"""
Example external ML model module.
Provide a generate(genre,width,height,difficulty,seed) function.
This simple example slightly transforms the procedural grid to mimic inference.
Replace this with your PyTorch/TensorFlow inference logic later.
"""

import random
from typing import List

def generate(genre: str, width: int, height: int, difficulty: int, seed: str):
    # Simple deterministic PRNG based on seed
    try:
        seed_int = int(seed) if seed and str(seed).isdigit() else None
    except Exception:
        seed_int = None
    rnd = random.Random(seed_int if seed_int is not None else hash(seed))

    # A very small example: generate a checker-ish floor for testing
    # We'll produce a grid that differs from the procedural generator so you can see a change.
    width = max(8, min(300, int(width)))
    height = max(6, min(120, int(height)))
    grid = [[0 for _ in range(width)] for _ in range(height)]

    # make alternating floor rows and sprinkle treasures
    for y in range(height):
        if y % 3 == 0:
            for x in range(width):
                grid[y][x] = 1  # floor/ground
        else:
            # sparse floor tiles
            for x in range(0, width, 7):
                if rnd.random() < 0.8:
                    grid[y][x] = 1
    # Put a few treasures and enemies depending on difficulty
    num_treasures = max(1, difficulty)
    for _ in range(num_treasures):
        x = rnd.randint(0, width-1)
        y = rnd.randint(0, height-1)
        if grid[y][x] == 0:
            grid[y][x] = 3
    # enemies
    num_enemies = max(1, difficulty//1)
    for _ in range(num_enemies):
        x = rnd.randint(0, width-1)
        y = rnd.randint(0, height-1)
        if grid[y][x] == 0:
            grid[y][x] = 4
    return grid
