import matplotlib.pyplot as plt
import matplotlib.patches as patches

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


def main():
    algorithm = choose_algorithm()
    if algorithm == 'bfs':
        path = bfs(maze)
    else:
        path = dfs(maze)

    if path:
        print(f"Path found using {algorithm.upper()}: {path}")
    else:
        print(f"No path found using {algorithm.upper()}.")

    draw_maze(maze, path)
    print(maze)
    while():
        print("sug mig")


if __name__ == "__main__":
    main()


