#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
from timeit import timeit
from concurrent.futures import ThreadPoolExecutor
from mpl_toolkits.mplot3d import Axes3D
"""
Created on Wed Sep 25 10:25:14 2024

@author: Joseph Wang
"""
def plot_3d_animation(initial_grid, steps=50):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Create the initial plot
    grid = initial_grid.copy()
    frames = [grid.copy()]  # Store initial state
    
    # Generate all frames first
    for _ in range(steps):
        increment(grid, pop_grow, walk_func)
        frames.append(grid.copy())
    
    # Initialize meshgrid and surface BEFORE animation
    rows, cols = grid.shape
    x, y = np.meshgrid(np.arange(rows), np.arange(cols))
    surf = [ax.plot_surface(x, y, frames[0], cmap='viridis')]
    total_sum_text = fig.text(0.05, 0.95, "", transform=fig.transFigure)

    def update_with_sum(frame):
        surf[0].remove()
        surf[0] = ax.plot_surface(x, y, frames[frame], cmap='viridis')
        total_sum = np.sum(frames[frame])
        total_sum_text.set_text(f"Total Sum: {total_sum:.2f}")
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        return surf

    # Keep only one FuncAnimation
    anim = FuncAnimation(fig, update_with_sum, frames=len(frames), 
                        interval=200, blit=False)
    
    # Save animation as MP4
    anim.save('grid_evolution.mp4', writer='ffmpeg')
    plt.show()

def increment(grid, growth_function, walk_function):
    def logistic_growth():
        growth_grid = growth_function(grid)
        return growth_grid
        # for element in np.nditer(grid, op_flags=['readwrite']):
        #     element[...] += growth_function(element)

    def walk_grid():
        walk(grid, walk_function)
    walk_grid()
    grid += logistic_growth()
    #print(np.sum(grid))
    return grid
# want this to eventually take in two lambda functions

def walk(grid, walk_function):
    # grid_copy = np.copy(grid)
    # for (i, j), value in np.ndenumerate(grid):
    #     grid_copy[i, j] = walkfunction(value)
    grid_donating = walk_function(grid) #change the var name!
    for (i, j), value in np.ndenumerate(grid):
        L = adj(grid, i, j)
        for neighbor in L:
            grid[i, j] += (grid_donating[neighbor[0], neighbor[1]] / len(adj(grid, neighbor[0], neighbor[1])))
            #each grid cell gets a fraction of the walk function value of its neighbors, 
            # where the fraction is proportional to how many neighbors the neighbor has
    grid -= grid_donating
    return grid  # Return the updated grid

def adj(grid, x, y):
    neighbors = []
    rows, cols = grid.shape
    if x + 1 < rows:  # down
        neighbors.append([x+1, y])
    if x - 1 >= 0:    # up
        neighbors.append([x-1, y])
    if y + 1 < cols:  # right
        neighbors.append([x, y+1])
    if y - 1 >= 0:    # left
        neighbors.append([x, y-1])
    return neighbors

def log_growth(r, k, area = 1):
    log_func = lambda x: (r*x*(1-x/(k)))
    return log_func 
def walk_lambda(x):
    return x

def normalize(grid):
    grid_sum = np.sum(grid)
    for element in np.nditer(grid, op_flags=['readwrite']):
        element[...] /= grid_sum
    return grid

array_ones = np.ones((40, 40))
#creates 40x40 grid
for (i, j), value in np.ndenumerate(array_ones):
    array_ones[i, j] = np.random.uniform(0, 1)
    #assigns random values to each cell
    if(i > 20):
        array_ones[i, j] = 0
    #gives a lower half of the grid 0 for assymetry
    if(j > 20):
        array_ones[i, j] = 2
    #gives a right half of the grid 2 for assymmetry
#array_twos = array_ones.copy()
normalize(array_ones) # normalizes grid entries to sum to 1, so each can represent a density
#normalize(array_twos)
pop_grow = log_growth(0.5, 1, 1) # defines logistic growth function with growth rate 0.5 and carrying capacity 1 per cell
walk_func = lambda x: x*x # defines a migration function that is quadratic in the density, evenly spread out amongst neighbors
plot_3d_animation(array_ones, steps = 200) #animation with 100 increments

#Baby Optimization, trying to find best coefficients for a predetermined function shape
#e.g finding best coefficient for a linear walk function, quadratic, etc. 
#need to find a way to generalize the optimization process to multiple parameters. 
#Target Total Population(in terms of original)
# n = 200 
# coeff_one = 0.99
# coeff_two = 0.01
# max_iterations = 1000  # Set a maximum number of iterations
# iteration = 0

#checks to see if either increments have reached the target sum

# while coeff_one > coeff_two and iteration < max_iterations:
#     walk_func = lambda x: x * coeff_one 
#     walk_func1 = lambda x: x * coeff_two 

#     while array_ones.sum() < n and array_twos.sum() < n:
#         increment(array_ones, pop_grow, walk_func)
#         increment(array_twos, pop_grow, walk_func1)

#     if array_ones.sum() > n and array_twos.sum() > n:
#         if array_ones.sum() > array_twos.sum():
#             print("Walk1")
#             coeff_two += 0.01
#             print(coeff_two)
#         else:
#             print("Walk2")
#             coeff_one -= 0.01
#             print(coeff_one)
#     elif array_ones.sum() > n and array_twos.sum() < n:
#         print("Walk1")
#         coeff_two += 0.01
#         print(coeff_two)
#     elif array_ones.sum() < n and array_twos.sum() > n:
#         print("Walk2")
#         coeff_one -= 0.01
#         print(coeff_one)

#     iteration += 1

# if iteration == max_iterations:
#     print("Terminated due to reaching maximum iterations.")
#     print(coeff_one, coeff_two)
# else:
#     print(coeff_one)

