# -*- coding: utf-8 -*-
"""
circle-obstacle grid-map generation utility
Input: obstacle list (each obstacle is[x, y, radius]), map size
Output: grid-map numpy array, with optional visualization
"""
import numpy as np
import matplotlib.pyplot as plt



def calculate_distance(x1, y1, x2, y2):
    """Compute the distance between two points in grid cells"""
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def generate_grid_map(num_obstacles=8, map_size=100):
    """
    obstacles: list of [x, y, radius]
    map_size: map side length (number of grid cells)
    Return: grid_map (0-free cell, 1-obstacle)
    """

    np.random.seed(42)

    obstacle_params = []
    for i in range(num_obstacles):
        while True:
            x = np.random.uniform(15, 85)
            y = np.random.uniform(15, 85)
            radius = np.random.uniform(3, 8)
            start_dist = calculate_distance(x, y, 10.0, 10.0)
            target_dist = calculate_distance(x, y, 90.0, 90.0)
            if start_dist > radius + 10 and target_dist > radius + 10:
                obstacle_params.append([x, y, radius]) # generate obstacle circles, unused later
                break
    # generategrid map, obstacle is 1
    grid_map = np.zeros((map_size, map_size), dtype=np.uint8)
    yy, xx = np.meshgrid(np.arange(map_size), np.arange(map_size))
    for x, y, r in obstacle_params:
            mask = (xx - x)**2 + (yy - y)**2 <= r**2
            grid_map[mask] = 1

    return grid_map

def visualize_grid_map(grid_map, start=None, target=None, trajectory=None):
    plt.figure(figsize=(8,8))
    plt.imshow(grid_map.T, origin='lower', cmap='Reds', alpha=0.7)
    if start:
        plt.plot(start[0], start[1], 'go', markersize=10, label='Start')
    if target:
        plt.plot(target[0], target[1], 'g*', markersize=15, label='Target')
    if trajectory is not None:
        traj = np.array(trajectory)
        plt.plot(traj[:,0], traj[:,1], 'b-', linewidth=2, label='Trajectory')
        plt.plot(traj[-1,0], traj[-1,1], 'bo', markersize=8, label='Current Position')
    plt.xlim(0, grid_map.shape[0])
    plt.ylim(0, grid_map.shape[1])
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.title('Grid Map Visualization')
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    # exampleobstacle
    # obstacles = [
    #     [20, 40, 10],
    #     [30, 70, 8],
    #     [50, 30, 7],
    #     [60, 60, 6],
    #     [80, 20, 9],
    #     [70, 80, 7],
    #     [40, 50, 5],
    #     [60, 20, 8],
    # ]
    start = [10, 10]
    target = [90, 90]
    grid_map = generate_grid_map(num_obstacles=8, map_size=100)
    # Optional: simulated trajectory
    trajectory = [start, [15,15], [20,20], [25,25], [30,30], [35,35], [40,40], [45,45], [50,50], target]
    visualize_grid_map(grid_map, start, target, trajectory)
