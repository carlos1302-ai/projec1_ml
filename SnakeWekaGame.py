"""
Snake Eater
Made with PyGame
Machine Learning Classes - University Carlos III de Madrid
Carlos Barboza - 100472143
Lucas Monzón - 100473232
"""

import pygame, sys, time, random
import heapq, atexit, os
from wekaI import Weka


def get_instance_attributes(game):
    head_x, head_y = game.snake_pos
    food_x, food_y = game.food_pos

    # Numeric distances
    food_left = head_x - food_x if food_x < head_x else 0
    food_right = food_x - head_x if food_x > head_x else 0
    food_up = head_y - food_y if food_y < head_y else 0
    food_down = food_y - head_y if food_y > head_y else 0

    # Occupancy as numeric 0 or 1
    def occupied(cell):
        if cell[0] < 0 or cell[0] >= FRAME_SIZE_X or cell[1] < 0 or cell[1] >= FRAME_SIZE_Y:
            return 1
        return 1 if (cell in game.snake_body and cell != game.snake_body[-1]) else 0

    left_occ  = occupied([head_x - 10, head_y])
    up_occ    = occupied([head_x, head_y - 10])
    right_occ = occupied([head_x + 10, head_y])
    down_occ  = occupied([head_x, head_y + 10])

    score = game.score
    manhattan_distance = abs(head_x - food_x) + abs(head_y - food_y)

    # New helper: compute distance to obstacle in a given direction
    def distance_in_direction(dx, dy):
        distance = 0
        current = [head_x, head_y]
        while True:
            current[0] += dx
            current[1] += dy
            distance += 10  # step size
            # Check for wall collision
            if current[0] < 0 or current[0] >= FRAME_SIZE_X or current[1] < 0 or current[1] >= FRAME_SIZE_Y:
                break
            # Check for collision with snake body (exclude tail, similar to occupied())
            if current in game.snake_body and current != game.snake_body[-1]:
                break
        return distance

    # Compute distances in each cardinal direction
    dist_left = distance_in_direction(-10, 0)
    dist_up = distance_in_direction(0, -10)
    dist_right = distance_in_direction(10, 0)
    dist_down = distance_in_direction(0, 10)

    # Return 14 values (the 15th attribute is the class, appended in predict())
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
        down_occ#,
        #dist_left,
        #dist_up,
        #dist_right,
        #dist_down
        #score,
        #manhattan_distance
    ]


# Global buffer for ARFF log entries.
log_buffer = []

def get_arff_instance(game):
    """
    Returns a string in ARFF format representing the current game state.
    """
    head_x, head_y = game.snake_pos
    food_x, food_y = game.food_pos

    # Compute distances to the food (only positive differences, otherwise 0)
    food_left  = head_x - food_x if food_x < head_x else 0
    food_right = food_x - head_x if food_x > head_x else 0
    food_up    = head_y - food_y if food_y < head_y else 0
    food_down  = food_y - head_y if food_y > head_y else 0

    # Check occupancy for adjacent cells
    def occupied(cell):
        if cell[0] < 0 or cell[0] >= FRAME_SIZE_X or cell[1] < 0 or cell[1] >= FRAME_SIZE_Y:
            return 1
        return 1 if (cell in game.snake_body and cell != game.snake_body[-1]) else 0

    left_cell  = [head_x - 10, head_y]
    up_cell    = [head_x, head_y - 10]
    right_cell = [head_x + 10, head_y]
    down_cell  = [head_x, head_y + 10]

    left_occ  = occupied(left_cell)
    up_occ    = occupied(up_cell)
    right_occ = occupied(right_cell)
    down_occ  = occupied(down_cell)

    direction = game.direction
    score = game.score
    manhattan_distance = abs(head_x - food_x) + abs(head_y - food_y)

    next_head = [head_x, head_y]
    if direction == "UP":
        next_head[1] -= 10
    elif direction == "DOWN":
        next_head[1] += 10
    elif direction == "LEFT":
        next_head[0] -= 10
    elif direction == "RIGHT":
        next_head[0] += 10

    if next_head == game.food_pos:
        next_score = score + 100
    else:
        next_score = score - 1

    instance = f"{head_x},{head_y},{food_x},{food_y},{food_left},{food_up},{food_right},{food_down},{left_occ},{up_occ},{right_occ},{down_occ},{score},{next_head},{manhattan_distance},{direction}"
    return instance

# Modified flush_logs() with the updated ARFF header.
def flush_logs():
    """
    Writes buffered ARFF log entries to 'game_log.arff'.
    If the file is new, writes the header with the updated attributes.
    """
    filename = "game_log.arff"
    if not os.path.isfile(filename):
        header = """@relation snake_game

@attribute head_x numeric
@attribute head_y numeric
@attribute food_x numeric
@attribute food_y numeric
@attribute food_left numeric
@attribute food_up numeric
@attribute food_right numeric
@attribute food_down numeric
@attribute left_occ {0,1}
@attribute up_occ {0,1}
@attribute right_occ {0,1}
@attribute down_occ {0,1}
@attribute score numeric
@attribute next_score numeric
@attribute manhattan_distance numeric
@attribute direction {UP,DOWN,LEFT,RIGHT}

@data
"""
        with open(filename, "w") as f:
            f.write(header)
    if log_buffer:
        with open(filename, "a") as f:
            f.write("\n".join(log_buffer) + "\n")
        log_buffer.clear()


atexit.register(flush_logs)

# DIFFICULTY settings (frames per second)
DIFFICULTY = 0

# Game board dimensions
FRAME_SIZE_X = 420
FRAME_SIZE_Y = 420

# Colors (R, G, B)
BLACK   = pygame.Color(51, 51, 51)
WHITE   = pygame.Color(255, 255, 255)
RED     = pygame.Color(204, 51, 0)
GREEN   = pygame.Color(0, 255, 0)
GRAY    = pygame.Color(128, 128, 128)
BLUE    = pygame.Color(0, 51, 102)
LIGHT   = pygame.Color(200, 200, 200)
ORANGE  = pygame.Color(255, 165, 0)  # Snake's head color

# GAME STATE CLASS
class GameState:
    def __init__(self, FRAME_SIZE):
        self.snake_pos = [100, 50]
        self.snake_body = [[100, 50], [90, 50], [80, 50]]
        self.food_pos = [random.randrange(1, (FRAME_SIZE[0] // 10)) * 10,
                         random.randrange(1, (FRAME_SIZE[1] // 10)) * 10]
        self.food_spawn = True
        self.direction = 'RIGHT'
        self.change_to = self.direction
        self.score = 0
        self.max_score = 0  # Running highest score so far
        self.max_at = 0

    def __str__(self):
        food_str = '[' + ';'.join(map(str, self.food_pos)) + ']'
        head_str = '[' + ';'.join(map(str, self.snake_pos)) + ']'
        body_str = get_double_list_str(self.snake_body)
        return f"{food_str},{self.direction},{head_str},{body_str}"

def get_double_list_str(lst):
    s = "["
    for i in lst:
        s += '[' + ';'.join(map(str, i)) + '];'
    s = s[:-1] + ']'
    return s

def game_over(game):
    flush_logs()
    print("Game Over! Score:", game.score)
    my_font = pygame.font.SysFont('times new roman', 90)
    game_over_surface = my_font.render('YOU DIED', True, WHITE)
    game_over_rect = game_over_surface.get_rect()
    game_over_rect.midtop = (FRAME_SIZE_X / 2, FRAME_SIZE_Y / 4)
    game_window.fill(BLUE)
    game_window.blit(game_over_surface, game_over_rect)
    show_score(game, 0, WHITE, 'times', 20)
    pygame.display.flip()
    time.sleep(3)
    pygame.quit()
    sys.exit()

def show_score(game, choice, color, font, size):
    score_font = pygame.font.SysFont(font, size)
    score_surface = score_font.render('Score : ' + str(game.score), True, color)
    score_rect = score_surface.get_rect()
    if choice == 1:
        score_rect.midtop = (FRAME_SIZE_X / 8, 15)
    else:
        score_rect.midtop = (FRAME_SIZE_X / 2, FRAME_SIZE_Y / 1.25)
    game_window.blit(score_surface, score_rect)

def move_keyboard(game, event):
    change_to = game.direction
    if event.type == pygame.KEYDOWN:
        if (event.key == pygame.K_UP or event.key == ord('w')) and game.direction != 'DOWN':
            change_to = 'UP'
        if (event.key == pygame.K_DOWN or event.key == ord('s')) and game.direction != 'UP':
            change_to = 'DOWN'
        if (event.key == pygame.K_LEFT or event.key == ord('a')) and game.direction != 'RIGHT':
            change_to = 'LEFT'
        if (event.key == pygame.K_RIGHT or event.key == ord('d')) and game.direction != 'LEFT':
            change_to = 'RIGHT'
    return change_to

def hilbert_index(x, y, order):
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
    Improved A* where each move costs 1 point.
    This way, the total path cost equals the number of moves,
    which reflects the -1 point per time-step penalty in the game.
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
        return hilbert_heuristic(a, b) + random.uniform(-epsilon, epsilon)
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), start))
    came_from = {}
    g_score = {start: 0}
    closed_set = set()
    while open_set:
        current_f, current = heapq.heappop(open_set)
        if current == goal:
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

def compute_heuristic(cell, food, cell_size=10, alpha=0.001, order=6, epsilon=0.0):
    manhattan = abs(cell[0] - food[0]) + abs(cell[1] - food[1])
    a_grid = (cell[0] // cell_size, cell[1] // cell_size)
    b_grid = (food[0] // cell_size, food[1] // cell_size)
    h_index_a = hilbert_index(a_grid[0], a_grid[1], order)
    h_index_b = hilbert_index(b_grid[0], b_grid[1], order)
    hilbert_diff = abs(h_index_a - h_index_b)
    return manhattan + alpha * hilbert_diff

def print_state(game):
    print("--------GAME STATE--------")
    print("FrameSize:", FRAME_SIZE_X, FRAME_SIZE_Y)
    print("Direction:", game.direction)
    print("Snake X:", game.snake_pos[0], ", Snake Y:", game.snake_pos[1])
    print("Snake Body:", game.snake_body)
    print("Food X:", game.food_pos[0], ", Food Y:", game.food_pos[1])
    print("Score:", game.score)

def print_line_data(game, filename="game_log.arff"):
    """
    Returns a string in ARFF format representing the game state.
    Attributes include:
      head_x, head_y,
      food_x, food_y,
      snake_length,
      seg2_x, seg2_y, seg3_x, seg3_y,
      tail_x, tail_y,
      left_occ, up_occ, right_occ, down_occ,
      direction (numeric: UP=0, RIGHT=1, DOWN=2, LEFT=3)
    """
    return get_arff_instance(game)




# Initialization
weka = Weka()
weka.start_jvm()


check_errors = pygame.init()
if check_errors[1] > 0:
    print(f"[!] Had {check_errors[1]} errors when initialising game, exiting...")
    sys.exit(-1)
else:
    print("[+] Game successfully initialised")

game_window = pygame.display.set_mode((FRAME_SIZE_X, FRAME_SIZE_Y))
pygame.display.set_caption("Snake Eater - Machine Learning (UC3M)")
fps_controller = pygame.time.Clock()

game = GameState((FRAME_SIZE_X, FRAME_SIZE_Y))

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            flush_logs()
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                flush_logs()
                pygame.event.post(pygame.event.Event(pygame.QUIT))
            # Uncomment for manual control:
            game.direction = move_keyboard(game, event)

    # Uncomment the next line to use the AI move function.
    x = get_instance_attributes(game)
    #print("X:", x)
    weka_direction = weka.predict("./Final_models/heuristic_trained/j48.model", x, "./Arffs/tutorial_pruned_cleaned.arff")
    #print(weka_direction)
    game.direction = weka_direction
    #game.direction = move_tutorial_1(game)

    # Log the current game state in ARFF format.
    log_buffer.append(print_line_data(game))
    if len(log_buffer) > 100:
        flush_logs()
        pass

    if game.direction == "UP":
        game.snake_pos[1] -= 10
    if game.direction == "DOWN":
        game.snake_pos[1] += 10
    if game.direction == "LEFT":
        game.snake_pos[0] -= 10
    if game.direction == "RIGHT":
        game.snake_pos[0] += 10

    game.snake_body.insert(0, list(game.snake_pos))
    if game.snake_pos[0] == game.food_pos[0] and game.snake_pos[1] == game.food_pos[1]:
        game.score += 100
        game.food_spawn = False
    else:
        game.snake_body.pop()
        game.score -= 1

    if not game.food_spawn:
        new_food_pos = [random.randrange(1, (FRAME_SIZE_X // 10)) * 10,
                        random.randrange(1, (FRAME_SIZE_Y // 10)) * 10]
        while new_food_pos in game.snake_body:
            new_food_pos = [random.randrange(1, (FRAME_SIZE_X // 10)) * 10,
                            random.randrange(1, (FRAME_SIZE_Y // 10)) * 10]
        game.food_pos = new_food_pos
    game.food_spawn = True

    if game.score > game.max_score:
        game.max_score = game.score

    game_window.fill(BLUE)
    pygame.draw.rect(game_window, GREEN, pygame.Rect(game.snake_body[0][0], game.snake_body[0][1], 10, 10))
    for pos in game.snake_body[1:]:
        pygame.draw.rect(game_window, LIGHT, pygame.Rect(pos[0], pos[1], 10, 10))
    pygame.draw.rect(game_window, RED, pygame.Rect(game.food_pos[0], game.food_pos[1], 10, 10))

    if game.snake_pos[0] < 0 or game.snake_pos[0] > FRAME_SIZE_X - 10:
        game_over(game)
    if game.snake_pos[1] < 0 or game.snake_pos[1] > FRAME_SIZE_Y - 10:
        game_over(game)
    for block in game.snake_body[1:]:
        if game.snake_pos[0] == block[0] and game.snake_pos[1] == block[1]:
            game_over(game)

    show_score(game, 1, WHITE, "consolas", 15)
    pygame.display.update()
    fps_controller.tick(DIFFICULTY)

