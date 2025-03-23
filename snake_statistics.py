"""
Modified Snake Eater for Statistical Testing with Max Score Tracking and Box Plots
Runs headless simulations and gathers score statistics (final and maximum) for each (model, dataset) combination.
Generates box plots for the score distributions per model.
"""

import pygame
import sys, time, random, heapq, os, math, statistics
import matplotlib.pyplot as plt

# Import the Weka interface (assumed available)
from wekaI import Weka

# -----------------------------
# Helper Functions (Game Logic)
# -----------------------------
def get_instance_attributes(game):
    """
    Returns a list of numeric attributes representing the current game state.
    """
    head_x, head_y = game.snake_pos
    food_x, food_y = game.food_pos

    # Binary flags indicating the relative food position
    food_left = head_x - food_x if food_x < head_x else 0
    food_right = food_x - head_x if food_x > head_x else 0
    food_up = head_y - food_y if food_y < head_y else 0
    food_down = food_y - head_y if food_y > head_y else 0

    # Check occupancy for adjacent cells.
    def occupied(cell):
        if cell[0] < 0 or cell[0] >= FRAME_SIZE_X or cell[1] < 0 or cell[1] >= FRAME_SIZE_Y:
            return 1
        return 1 if (cell in game.snake_body and cell != game.snake_body[-1]) else 0

    left_occ  = occupied([head_x - 10, head_y])
    up_occ    = occupied([head_x, head_y - 10])
    right_occ = occupied([head_x + 10, head_y])
    down_occ  = occupied([head_x, head_y + 10])

    # Only 12 attributes are used for prediction.
    return [
        head_x,
        head_y,
        food_x,
        food_y,
        food_left,
        food_up,
        food_right,
        food_down,
        left_occ,
        up_occ,
        right_occ,
        down_occ
    ]

def get_instance_attributes_extended(game):
    """
    Returns a list of numeric attributes representing the current game state,
    including distances to the nearest obstacle in the four cardinal directions.
    """
    head_x, head_y = game.snake_pos
    food_x, food_y = game.food_pos

    # Compute food direction differences (non-negative)
    food_left  = head_x - food_x if food_x < head_x else 0
    food_right = food_x - head_x if food_x > head_x else 0
    food_up    = head_y - food_y if food_y < head_y else 0
    food_down  = food_y - head_y if food_y > head_y else 0

    # Check occupancy for adjacent cells
    def occupied(cell):
        if cell[0] < 0 or cell[0] >= FRAME_SIZE_X or cell[1] < 0 or cell[1] >= FRAME_SIZE_Y:
            return 1
        return 1 if (cell in game.snake_body and cell != game.snake_body[-1]) else 0

    left_occ  = occupied([head_x - 10, head_y])
    up_occ    = occupied([head_x, head_y - 10])
    right_occ = occupied([head_x + 10, head_y])
    down_occ  = occupied([head_x, head_y + 10])

    # Helper: compute distance to nearest obstacle in a given direction.
    def distance_in_direction(dx, dy):
        distance = 0
        current = [head_x, head_y]
        while True:
            current[0] += dx
            current[1] += dy
            distance += 10  # step size
            # Check if out of bounds.
            if current[0] < 0 or current[0] >= FRAME_SIZE_X or current[1] < 0 or current[1] >= FRAME_SIZE_Y:
                break
            # Check collision with snake body (excluding tail)
            if current in game.snake_body and current != game.snake_body[-1]:
                break
        return distance

    # Compute distances in each cardinal direction.
    dist_left = distance_in_direction(-10, 0)
    dist_up = distance_in_direction(0, -10)
    dist_right = distance_in_direction(10, 0)
    dist_down = distance_in_direction(0, 10)

    # Return the extended list of attributes.
    return [
        head_x, head_y,
        food_x, food_y,
        food_left, food_up, food_right, food_down,
        left_occ, up_occ, right_occ, down_occ,
        dist_left, dist_up, dist_right, dist_down
    ]

def hilbert_index(x, y, order):
    """
    Returns the Hilbert index for given coordinates (used in the heuristic).
    """
    index = 0
    n = 1 << order  # 2^order
    s = n >> 1
    while s:
        rx = 1 if (x & s) else 0
        ry = 1 if (y & s) else 0
        index += s * s * ((3 * rx) ^ ry)
        if ry == 0:
            if rx == 1:
                x = n - 1 - x
                y = n - 1 - y
            x, y = y, x
        s //= 2
    return index

def astar(start, goal, obstacles, grid_width, grid_height, cell_size=10, alpha=0.001, order=6, epsilon=0.1):
    """
    A* pathfinder with a Hilbert-based heuristic.
    """
    start = tuple(start)
    goal = tuple(goal)
    obstacles_set = set(tuple(o) for o in obstacles)

    def to_grid(coord):
        return (coord[0] // cell_size, coord[1] // cell_size)

    def hilbert_heuristic(a, b):
        manhattan = (abs(a[0] - b[0]) + abs(a[1] - b[1])) / cell_size
        a_grid = to_grid(a)
        b_grid = to_grid(b)
        h_index_a = hilbert_index(a_grid[0], a_grid[1], order)
        h_index_b = hilbert_index(b_grid[0], b_grid[1], order)
        hilbert_diff = abs(h_index_a - h_index_b)
        return manhattan + alpha * hilbert_diff

    def heuristic(a, b):
        # Add a small random noise (epsilon) to break ties.
        return hilbert_heuristic(a, b) + random.uniform(-epsilon, epsilon)

    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), start))
    came_from = {}
    g_score = {start: 0}
    closed_set = set()

    while open_set:
        current_f, current = heapq.heappop(open_set)
        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(list(current))
                current = came_from[current]
            path.append(list(start))
            path.reverse()
            return path
        closed_set.add(current)
        for dx, dy in [(0, -cell_size), (0, cell_size), (-cell_size, 0), (cell_size, 0)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if (neighbor[0] < 0 or neighbor[0] >= grid_width or
                neighbor[1] < 0 or neighbor[1] >= grid_height):
                continue
            if neighbor in obstacles_set:
                continue
            if neighbor in closed_set:
                continue
            tentative_g = g_score[current] + 1
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, neighbor))
    return []

def is_safe_move(game, next_cell):
    if next_cell == game.food_pos:
        new_body = [next_cell] + game.snake_body[:]
    else:
        new_body = [next_cell] + game.snake_body[:-1]
    new_head = next_cell
    new_tail = new_body[-1]
    new_obstacles = new_body[1:]
    if new_tail in new_obstacles:
        new_obstacles.remove(new_tail)
    path_to_tail = astar(new_head, new_tail, new_obstacles, FRAME_SIZE_X, FRAME_SIZE_Y, cell_size=10)
    return len(path_to_tail) > 0

def move_tutorial_1(game):
    """
    Determines the next move based on a hierarchy of strategies.
    First, if the snake's score is far below its max score, it forces a suicidal move
    by heading toward its second segment. Otherwise, it computes a candidate path to the food
    using A*; if that move is safe, it is chosen. If not, it evaluates all possible moves
    and selects the safest move based on Manhattan distance to the food. As a last resort,
    it attempts to follow its tail.
    """
    # Suicide logic: force collision if score is far below max.
    if game.max_score - game.score >= 2 * FRAME_SIZE_X:
        if len(game.snake_body) > 1:
            dx = game.snake_body[1][0] - game.snake_pos[0]
            dy = game.snake_body[1][1] - game.snake_pos[1]
            if dx == 10:
                return 'RIGHT'
            elif dx == -10:
                return 'LEFT'
            elif dy == 10:
                return 'DOWN'
            elif dy == -10:
                return 'UP'
        else:
            return game.direction

    obstacles = game.snake_body[1:]
    path = astar(game.snake_pos, game.food_pos, obstacles, FRAME_SIZE_X, FRAME_SIZE_Y, cell_size=10)
    if len(path) >= 2:
        candidate_next = path[1]
        if is_safe_move(game, candidate_next):
            dx = candidate_next[0] - game.snake_pos[0]
            dy = candidate_next[1] - game.snake_pos[1]
            if dx == 10:
                return 'RIGHT'
            elif dx == -10:
                return 'LEFT'
            elif dy == 10:
                return 'DOWN'
            elif dy == -10:
                return 'UP'
    safe_moves = []
    directions = [('UP', (0, -10)), ('DOWN', (0, 10)), ('LEFT', (-10, 0)), ('RIGHT', (10, 0))]
    for direction, (dx, dy) in directions:
        next_cell = [game.snake_pos[0] + dx, game.snake_pos[1] + dy]
        if (next_cell[0] < 0 or next_cell[0] >= FRAME_SIZE_X or
            next_cell[1] < 0 or next_cell[1] >= FRAME_SIZE_Y):
            continue
        if next_cell in game.snake_body[:-1]:
            continue
        if is_safe_move(game, next_cell):
            safe_moves.append((direction, next_cell))
    if safe_moves:
        def manhattan(cell):
            return abs(cell[0] - game.food_pos[0]) + abs(cell[1] - game.food_pos[1])
        best_move = min(safe_moves, key=lambda m: manhattan(m[1]))
        return best_move[0]
    tail = game.snake_body[-1]
    path_to_tail = astar(game.snake_pos, tail, obstacles, FRAME_SIZE_X, FRAME_SIZE_Y, cell_size=10)
    if len(path_to_tail) >= 2:
        next_cell = path_to_tail[1]
        dx = next_cell[0] - game.snake_pos[0]
        dy = next_cell[1] - game.snake_pos[1]
        if dx == 10:
            return 'RIGHT'
        elif dx == -10:
            return 'LEFT'
        elif dy == 10:
            return 'DOWN'
        elif dy == -10:
            return 'UP'
    return game.direction

# -----------------------------
# Game State and Simulation
# -----------------------------
# Global game board dimensions
FRAME_SIZE_X = 420
FRAME_SIZE_Y = 420

class GameState:
    def __init__(self, FRAME_SIZE):
        self.snake_pos = [100, 50]
        self.snake_body = [[100, 50], [90, 50], [80, 50]]
        self.food_pos = [random.randrange(1, (FRAME_SIZE[0] // 10)) * 10,
                         random.randrange(1, (FRAME_SIZE[1] // 10)) * 10]
        self.food_spawn = True
        self.direction = 'RIGHT'
        self.score = 0
        self.max_score = 0  # Tracks the highest score reached during the simulation.

def run_simulation(model_path, dataset_path, max_steps=25000, function=1):
    """
    Runs a single simulation of the snake game using the given model and dataset.
    Tracks both the final score and the maximum score achieved.
    Returns a tuple: (final_score, max_score).
    """
    game = GameState((FRAME_SIZE_X, FRAME_SIZE_Y))
    steps = 0
    while steps < max_steps:
        # Get instance attributes for the current state.
        if function == 1:
            x = get_instance_attributes(game)
        else:
            x = get_instance_attributes_extended(game)
        # Predict the next move using the given model and dataset.
        if model_path != "move_tutorial":
            weka_direction = weka.predict(model_path, x, dataset_path)
            game.direction = weka_direction
        else:
            game.direction = move_tutorial_1(game)

        # Update snake position based on the predicted direction.
        if game.direction == "UP":
            game.snake_pos[1] -= 10
        elif game.direction == "DOWN":
            game.snake_pos[1] += 10
        elif game.direction == "LEFT":
            game.snake_pos[0] -= 10
        elif game.direction == "RIGHT":
            game.snake_pos[0] += 10

        # Update snake body: insert new head position.
        game.snake_body.insert(0, list(game.snake_pos))
        # Check if food is eaten.
        if game.snake_pos[0] == game.food_pos[0] and game.snake_pos[1] == game.food_pos[1]:
            game.score += 100
            game.food_spawn = False
        else:
            game.snake_body.pop()  # Remove tail.
            game.score -= 1  # Penalize time-steps.

        # Update max score if current score exceeds it.
        if game.score > game.max_score:
            game.max_score = game.score

        # Food respawn logic.
        if not game.food_spawn:
            new_food_pos = [random.randrange(1, (FRAME_SIZE_X // 10)) * 10,
                            random.randrange(1, (FRAME_SIZE_Y // 10)) * 10]
            while new_food_pos in game.snake_body:
                new_food_pos = [random.randrange(1, (FRAME_SIZE_X // 10)) * 10,
                                random.randrange(1, (FRAME_SIZE_Y // 10)) * 10]
            game.food_pos = new_food_pos
        game.food_spawn = True

        # Collision detection (with walls or self)
        if game.snake_pos[0] < 0 or game.snake_pos[0] >= FRAME_SIZE_X:
            break
        if game.snake_pos[1] < 0 or game.snake_pos[1] >= FRAME_SIZE_Y:
            break
        collision = False
        for block in game.snake_body[1:]:
            if game.snake_pos[0] == block[0] and game.snake_pos[1] == block[1]:
                collision = True
                break
        if collision:
            break

        if game.score < -100:
            break

        steps += 1

    return game.score, game.max_score

# -----------------------------
# Statistical Testing Functions
# -----------------------------
def run_statistical_tests(model_paths, dataset_paths, runs_per_combination=20, functios=[1]):
    """
    Runs multiple simulation trials for each (model, dataset) pair.
    Collects both final and maximum scores, then computes statistics and prints results.
    Returns a dictionary with results for each model.
    """
    if len(model_paths) != len(functios):
        functios = [1] * len(model_paths)
    results = {}
    for i, (model_path, dataset_path) in enumerate(zip(model_paths, dataset_paths)):
        print("Running simulation for model {}".format(model_path))
        final_scores = []
        max_scores = []
        for j in range(runs_per_combination):
            print(f"  Run {j+1}/{runs_per_combination}")
            if functios:
                final, max_val = run_simulation(model_path, dataset_path, function=functios[i])
            else:
                final, max_val = run_simulation(model_path, dataset_path)
            final_scores.append(final)
            max_scores.append(max_val)
        # Compute statistics for final scores.
        avg_final = statistics.mean(final_scores)
        var_final = statistics.variance(final_scores) if len(final_scores) > 1 else 0
        n = len(final_scores)
        stdev_final = math.sqrt(var_final)
        try:
            from scipy.stats import t
            ci_final = t.ppf(0.975, n - 1) * stdev_final / math.sqrt(n)
        except ImportError:
            ci_final = 1.96 * stdev_final / math.sqrt(n)
        lower_final = avg_final - ci_final
        upper_final = avg_final + ci_final

        # Compute statistics for max scores.
        avg_max = statistics.mean(max_scores)
        var_max = statistics.variance(max_scores) if len(max_scores) > 1 else 0
        stdev_max = math.sqrt(var_max)
        try:
            from scipy.stats import t
            ci_max = t.ppf(0.975, n - 1) * stdev_max / math.sqrt(n)
        except ImportError:
            ci_max = 1.96 * stdev_max / math.sqrt(n)
        lower_max = avg_max - ci_max
        upper_max = avg_max + ci_max

        results[f"Model {i+1}"] = {
            "model_path": model_path,
            "dataset_path": dataset_path,
            "avg_final_score": avg_final,
            "var_final_score": var_final,
            "95% CI final": (lower_final, upper_final),
            "final_scores": final_scores,
            "avg_max_score": avg_max,
            "var_max_score": var_max,
            "95% CI max": (lower_max, upper_max),
            "max_scores": max_scores
        }
        print(f"Results for Model {i+1}:")
        print(f"  Model: {model_path}")
        print(f"  Dataset: {dataset_path}")
        print(f"  Average Final Score: {avg_final}")
        print(f"  Final Score Variance: {var_final}")
        print(f"  95% CI Final: ({lower_final}, {upper_final})")
        print(f"  Average Max Score: {avg_max}")
        print(f"  Max Score Variance: {var_max}")
        print(f"  95% CI Max: ({lower_max}, {upper_max})")
        print("-" * 40)
    return results

# -----------------------------
# Main Block
# -----------------------------
if __name__ == "__main__":
    # Initialize pygame in headless mode.
    pygame.init()

    # Start the Weka JVM
    weka = Weka()
    weka.start_jvm()

    # Define your list of models and the associated training datasets.
    # Ensure that the indices match (i.e. model_paths[i] uses dataset_paths[i]).
    model_paths = [
        "./Final_models/binary/j48.model",
        "./Final_models/extended/j48.model",
        "./Final_models/extended/ibk_5.model",
        #"./Final_models/binary/ibk_10.model",
        #"./Final_models/binary/logistic.model"#,
        #"./Final_models/binary/random.model"
    ]

    fuctions = [1, 2, 2]

    model_names = ["j48", "extended j48", "extended ibk 5"]

    dataset_paths = [
        "./Arffs/noiceless_pruned.arff",
        "./Arffs/clean_extended_pruned.arff",
        "./Arffs/clean_extended_pruned.arff"
    ]

    # Set the number of simulation runs for statistical significance.
    runs_per_combination = 20

    # Run statistical tests for each model/dataset pair.
    stats_results = run_statistical_tests(model_paths, dataset_paths, runs_per_combination, functios=fuctions)

    # -----------------------------
    # Generate Box Plots for the Models
    # -----------------------------
    model_labels = [f"Model {model_names[i]}" for i in range(len(model_paths))]

    # Extract final and max scores for each model.
    final_scores_data = [stats_results[f"Model {i+1}"]["final_scores"] for i in range(len(model_paths))]
    max_scores_data = [stats_results[f"Model {i+1}"]["max_scores"] for i in range(len(model_paths))]

    # Create two subplots: one for final scores, one for maximum scores.
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].boxplot(final_scores_data, )
    axes[0].set_title("Final Scores per Model")
    axes[0].set_xticklabels(model_labels)
    axes[0].set_ylabel("Score")
    axes[0].axhline(y=0, color='black', linestyle='--')

    axes[1].boxplot(max_scores_data)
    axes[1].set_title("Maximum Scores per Model")
    axes[1].set_xticklabels(model_labels)
    axes[1].set_ylabel("Score")
    axes[1].axhline(y=0, color='black', linestyle='--')

    plt.suptitle("Score Distributions for Each Model")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Optionally, stop the JVM if your Weka interface supports it.
    # weka.stop_jvm()