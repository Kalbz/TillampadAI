import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import time


# Constants
DIRECTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
POP_SIZE = 500
MUTATION_RATE = 0.05
CROSSOVER_RATE = 0.7
MAX_GEN = 500
MAX_STEPS = 100
ELITISM_RATE = 0.7  # 90% chance to keep the best path unchanged


# Representation: A path is a list of directions

def random_step():
    return random.choice(DIRECTIONS)


def random_path():
    return [random_step() for _ in range(MAX_STEPS)]

def fitness(path):
    x, y = 0, 0
    penalty = 0
    for i in range(len(path) - 1):
        dx, dy = path[i]
        next_dx, next_dy = path[i + 1]
        
        # Check for redundant moves
        if dx == -next_dx and dy == -next_dy:
            penalty += 500 # Penalty for redundant moves
            
        if is_valid_move(x, y, dx, dy) and is_valid_position(y + dy, x + dx):
            x += dx
            y += dy
        else:
            penalty += 10000  # Penalty for invalid moves
            
    distance_to_goal = abs(len(maze[0]) - 2 - x) + abs(len(maze) - 1 - y)
    if (y, x) == (len(maze) - 1, len(maze[0]) - 2):
        return 10000  # Big bonus for reaching the goal
    return 1 / (distance_to_goal + penalty)



def is_valid_position(y, x):
    return 0 <= y < len(maze) and 0 <= x < len(maze[0])


def repair_path(path):
    x, y = 0, 0
    repaired_path = []
    for dx, dy in path:
        if is_valid_move(x, y, dx, dy) and is_valid_position(y + dy, x + dx):
            x += dx
            y += dy
            repaired_path.append((dx, dy))
        else:
            # If the move is invalid, try to move towards the goal
            goal_x, goal_y = len(maze[0]) - 2, len(maze) - 1
            if goal_x > x and is_valid_move(x, y, 1, 0):
                repaired_path.append((1, 0))
                x += 1
            elif goal_y > y and is_valid_move(x, y, 0, 1):
                repaired_path.append((0, 1))
                y += 1
            else:
                # If we can't move towards the goal, just append a stationary move
                repaired_path.append((0, 0))

    while len(repaired_path) < MAX_STEPS:
        repaired_path.append((0, 0))
    return repaired_path


def is_valid_move(x, y, dx, dy):
    if dx == 1 and maze[y][x]['right'] == 0:
        return True
    if dx == -1 and maze[y][x]['left'] == 0:
        return True
    if dy == 1 and maze[y][x]['bottom'] == 0:
        return True
    if dy == -1 and maze[y][x]['top'] == 0:
        return True
    return False


def crossover(parent1, parent2):
    crossover_point = random.randint(0, MAX_STEPS - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child



def mutate(path):
    for i in range(len(path)):
        if random.random() < MUTATION_RATE:
            path[i] = random.choice(DIRECTIONS)
    return repair_path(path)  # Repair the path after mutation

def path_to_positions(path):
    x, y = 0, 0
    positions = [(y, x)]
    for dx, dy in path:
        if is_valid_move(x, y, dx, dy):
            x += dx
            y += dy
            positions.append((y, x))
    return positions


def ga():
    population = [random_path() for _ in range(POP_SIZE)]
    best_path = None
    best_fitness = 0
    stagnant_generations = 0
    previous_best_fitness = 0

    # Initialize the plot
    plt.ion()
    fig, ax = draw_maze(maze)
    line, = ax.plot([], [], color='b', linewidth=2)

    for generation in range(MAX_GEN):
        population.sort(key=fitness, reverse=True)
        current_best_fitness = fitness(population[0])
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_path = population[0]
            
            # Update the plot with the best path of this generation
            path_positions = path_to_positions(best_path)
            x_data = [x + 0.5 for _, x in path_positions]
            y_data = [y + 0.5 for y, _ in path_positions]
            line.set_data(x_data, y_data)
            fig.canvas.flush_events()
            time.sleep(0.1)

        if best_fitness == 10000:
            plt.ioff()
            return path_to_positions(best_path)
        new_population = []
        if random.random() < ELITISM_RATE:
            if random.random() < 0.05:  # 5% chance to replace the best path with a random one
                new_population.append(random_path())
            else:
                new_population.append(best_path)
        if best_fitness == previous_best_fitness:
            stagnant_generations += 1
        else:
            stagnant_generations = 0

        if stagnant_generations > 50:  # If no improvement for 50 generations, stop
            break

        previous_best_fitness = best_fitness


        while len(new_population) < POP_SIZE:
            parent1 = random.choice(population[:50])  # Select from the top 50
            parent2 = random.choice(population[:50])
            child = crossover(parent1, parent2)
            child = mutate(child)
            # Extend the path if it's shorter than MAX_STEPS
            while len(child) < MAX_STEPS:
                child.append((0, 0))
            new_population.append(child)
        population = new_population
    plt.ioff()
    return path_to_positions(best_path)

TOP, RIGHT, BOTTOM, LEFT = 'top', 'right', 'bottom', 'left'




# Create a default cell with no walls
default_cell = {'top': 0, 'right': 0, 'bottom': 0, 'left': 0}

# Create a 7x20 maze of default cells
maze = [[default_cell.copy() for _ in range(20)] for _ in range(7)]

# Add the top walls for the first row
for cell in maze[0]:
    cell['top'] = 1

# Add the bottom walls for the last row
for cell in maze[-1]:
    cell['bottom'] = 1

# Add the left wall for the first column
for row in maze:
    row[0]['left'] = 1

# Add the right wall for the last column
for row in maze:
    row[-1]['right'] = 1

def add_wall(maze, row, col, direction):
    maze[row][col][direction] = 1

def choose_algorithm():
    while True:
        choice = input("Which algorithm would you like to use? (BFS/DFS): ").lower()
        if choice in ['bfs', 'dfs']:
            return choice
        else:
            print("Invalid choice. Please choose either 'BFS' or 'DFS'.")
def bfs(maze):
    start = (0, 0)
    goal = (len(maze) - 1, len(maze[0]) - 2)
    visited = set()
    queue = [start]
    came_from = {start: None}

    while queue:
        current = queue.pop(0)
        print(f"Processing cell: {current}")
        if current == goal:
            # Reconstruct the path
            path = []
            while current:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        visited.add(current)
        for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            y, x = current
            ny, nx = y + dy, x + dx
            if 0 <= ny < len(maze) and 0 <= nx < len(maze[0]) and (ny, nx) not in visited:
                can_move = False
                if dx == 1 and maze[y][x]['right'] == 0:
                    can_move = True
                elif dx == -1 and maze[ny][nx]['right'] == 0:
                    can_move = True
                elif dy == 1 and maze[y][x]['bottom'] == 0:
                    can_move = True
                elif dy == -1 and maze[ny][nx]['bottom'] == 0:
                    can_move = True

                if can_move:
                    print(f"Adding neighbor: {(ny, nx)}")
                    queue.append((ny, nx))
                    came_from[(ny, nx)] = current
                    visited.add((ny, nx))  # Mark the neighbor as visited immediately after adding it to the queue

    return None  # No path found

def dfs(maze):
    start = (0, 0)
    goal = (len(maze) - 1, len(maze[0]) - 2)
    visited = set()
    stack = [start]
    came_from = {start: None}

    while stack:
        current = stack.pop()  # Note: We use pop() without an index for DFS, unlike BFS
        print(f"Processing cell: {current}")
        if current == goal:
            # Reconstruct the path
            path = []
            while current:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        visited.add(current)
        for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            y, x = current
            ny, nx = y + dy, x + dx
            if 0 <= ny < len(maze) and 0 <= nx < len(maze[0]) and (ny, nx) not in visited:
                can_move = False
                if dx == 1 and maze[y][x]['right'] == 0:
                    can_move = True
                elif dx == -1 and maze[ny][nx]['right'] == 0:
                    can_move = True
                elif dy == 1 and maze[y][x]['bottom'] == 0:
                    can_move = True
                elif dy == -1 and maze[ny][nx]['bottom'] == 0:
                    can_move = True

                if can_move:
                    print(f"Adding neighbor: {(ny, nx)}")
                    stack.append((ny, nx))
                    came_from[(ny, nx)] = current
                    visited.add((ny, nx))  # Mark the neighbor as visited immediately after adding it to the stack

    return None  # No path found


def draw_maze(maze, path=None):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    
    for i, row in enumerate(maze):
        for j, cell in enumerate(row):
            if cell['top']:
                ax.plot([j, j+1], [i, i], color='k')
            if cell['bottom']:
                ax.plot([j, j+1], [i+1, i+1], color='k')
            if cell['left']:
                ax.plot([j, j], [i, i+1], color='k')
            if cell['right']:
                ax.plot([j+1, j+1], [i, i+1], color='k')

    # Draw the path, if provided
    if path:
        for i in range(len(path) - 1):
            y1, x1 = path[i]
            y2, x2 = path[i + 1]
            ax.plot([x1 + 0.5, x2 + 0.5], [y1 + 0.5, y2 + 0.5], color='b', linewidth=2)

    # Mark the goal
    goal_rect = patches.Rectangle((len(maze[0])-2, len(maze)-1), 1, 1, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(goal_rect)
                
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 7)
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()  # This will invert the y-axis so that (0,0) is at the top-left
    plt.axis('off')  # Turn off the axis
    plt.show()
    return fig, ax

def build_entire_maze():
    add_wall(maze, 0, 0, BOTTOM)
    add_wall(maze, 0, 1, BOTTOM)
    add_wall(maze, 0, 2, BOTTOM)
    add_wall(maze, 0, 4, BOTTOM)
    add_wall(maze, 0, 5, BOTTOM)
    add_wall(maze, 0, 7, BOTTOM)
    add_wall(maze, 0, 8, BOTTOM)
    add_wall(maze, 0, 10, BOTTOM)
    add_wall(maze, 0, 12, BOTTOM)
    add_wall(maze, 0, 13, BOTTOM)
    add_wall(maze, 0, 14, BOTTOM)
    add_wall(maze, 0, 15, BOTTOM)
    add_wall(maze, 0, 19, BOTTOM)
    add_wall(maze, 1, 1, BOTTOM)
    add_wall(maze, 1, 2, BOTTOM)
    add_wall(maze, 1, 3, BOTTOM)
    add_wall(maze, 1, 4, BOTTOM)
    add_wall(maze, 1, 6, BOTTOM)
    add_wall(maze, 1, 9, BOTTOM)
    add_wall(maze, 1, 11, BOTTOM)
    add_wall(maze, 1, 12, BOTTOM)
    add_wall(maze, 1, 15, BOTTOM)
    add_wall(maze, 1, 16, BOTTOM)
    add_wall(maze, 1, 18, BOTTOM)
    add_wall(maze, 2, 0, BOTTOM)
    add_wall(maze, 2, 1, BOTTOM)
    add_wall(maze, 2, 3, BOTTOM)
    add_wall(maze, 2, 5, BOTTOM)
    add_wall(maze, 2, 6, BOTTOM)
    add_wall(maze, 2, 7, BOTTOM)
    add_wall(maze, 2, 8, BOTTOM)
    add_wall(maze, 2, 10, BOTTOM)
    add_wall(maze, 2, 11, BOTTOM)
    add_wall(maze, 2, 12, BOTTOM)
    add_wall(maze, 2, 14, BOTTOM)
    add_wall(maze, 2, 15, BOTTOM)
    add_wall(maze, 2, 16, BOTTOM)
    add_wall(maze, 2, 17, BOTTOM)
    add_wall(maze, 2, 18, BOTTOM)
    add_wall(maze, 3, 2, BOTTOM)
    add_wall(maze, 3, 4, BOTTOM)
    add_wall(maze, 3, 5, BOTTOM)
    add_wall(maze, 3, 6, BOTTOM)
    add_wall(maze, 3, 7, BOTTOM)
    add_wall(maze, 3, 11, BOTTOM)
    add_wall(maze, 3, 13, BOTTOM)
    add_wall(maze, 3, 14, BOTTOM)
    add_wall(maze, 3, 15, BOTTOM)
    add_wall(maze, 3, 16, BOTTOM)
    add_wall(maze, 3, 18, BOTTOM)
    add_wall(maze, 3, 19, BOTTOM)
    add_wall(maze, 4, 1, BOTTOM)
    add_wall(maze, 4, 2, BOTTOM)
    add_wall(maze, 4, 3, BOTTOM)
    add_wall(maze, 4, 16, BOTTOM)
    add_wall(maze, 5, 2, BOTTOM)
    add_wall(maze, 5, 5, BOTTOM)
    add_wall(maze, 5, 6, BOTTOM)
    add_wall(maze, 5, 8, BOTTOM)
    add_wall(maze, 5, 9, BOTTOM)
    add_wall(maze, 5, 12, BOTTOM)
    add_wall(maze, 5, 13, BOTTOM)
    add_wall(maze, 5, 15, BOTTOM)
    add_wall(maze, 5, 16, BOTTOM)
    add_wall(maze, 5, 18, BOTTOM)

    add_wall(maze, 0, 6, RIGHT)
    add_wall(maze, 0, 11, RIGHT)
    add_wall(maze, 0, 16, RIGHT)
    add_wall(maze, 1, 3, RIGHT)
    add_wall(maze, 1, 5, RIGHT)
    add_wall(maze, 1, 7, RIGHT)
    add_wall(maze, 1, 9, RIGHT)
    add_wall(maze, 1, 10, RIGHT)
    add_wall(maze, 1, 13, RIGHT)
    add_wall(maze, 1, 16, RIGHT)
    add_wall(maze, 1, 17, RIGHT)
    add_wall(maze, 2, 2, RIGHT)
    add_wall(maze, 2, 6, RIGHT)
    add_wall(maze, 2, 8, RIGHT)
    add_wall(maze, 2, 13, RIGHT)
    add_wall(maze, 2, 18, RIGHT)
    add_wall(maze, 3, 1, RIGHT)
    add_wall(maze, 3, 3, RIGHT)
    add_wall(maze, 3, 8, RIGHT)
    add_wall(maze, 3, 9, RIGHT)
    add_wall(maze, 3, 12, RIGHT)
    add_wall(maze, 4, 0, RIGHT)
    add_wall(maze, 4, 3, RIGHT)
    add_wall(maze, 4, 5, RIGHT)
    add_wall(maze, 4, 7, RIGHT)
    add_wall(maze, 4, 8, RIGHT)
    add_wall(maze, 4, 9, RIGHT)
    add_wall(maze, 4, 10, RIGHT)
    add_wall(maze, 4, 12, RIGHT)
    add_wall(maze, 4, 14, RIGHT)
    add_wall(maze, 4, 17, RIGHT)
    add_wall(maze, 5, 0, RIGHT)
    add_wall(maze, 5, 3, RIGHT)
    add_wall(maze, 5, 4, RIGHT)
    add_wall(maze, 5, 6, RIGHT)
    add_wall(maze, 5, 7, RIGHT)
    add_wall(maze, 5, 9, RIGHT)
    add_wall(maze, 5, 10, RIGHT)
    add_wall(maze, 5, 11, RIGHT)
    add_wall(maze, 5, 13, RIGHT)
    add_wall(maze, 5, 14, RIGHT)
    add_wall(maze, 5, 16, RIGHT)
    add_wall(maze, 5, 18, RIGHT)
    add_wall(maze, 6, 0, RIGHT)
    add_wall(maze, 6, 2, RIGHT)
    add_wall(maze, 6, 4, RIGHT)
    add_wall(maze, 6, 10, RIGHT)
    add_wall(maze, 6, 13, RIGHT)
    add_wall(maze, 6, 17, RIGHT)


    

build_entire_maze()
def add_opposite_walls(maze):
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if maze[i][j]['right'] == 1 and j < len(maze[0]) - 1:
                maze[i][j + 1]['left'] = 1
            if maze[i][j]['bottom'] == 1 and i < len(maze) - 1:
                maze[i + 1][j]['top'] = 1

add_opposite_walls(maze)



def main():
    path = ga()
    print("Path:", path)
    draw_maze(maze, path)

if __name__ == "__main__":
    main()