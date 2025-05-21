#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math
from timeit import timeit
from concurrent.futures import ThreadPoolExecutor
from mpl_toolkits.mplot3d import Axes3D
import rasterio
import copy
import time

"""
Created on Wed Sep 25 10:25:14 2024

@author: Joseph Wang
"""
class Graph:
    def __init__(self, vertices):
        self.vertices = vertices
        self.adj_list = {vertex: [] for vertex in vertices}

    def add_edge(self, u, v, weight=1):
         self.adj_list[u].append((v, weight))
         self.adj_list[v].append((u, weight))

    def get_neighbors(self, vertex):
        return self.adj_list[vertex]

    def get_vertices(self):
        return list(self.vertices)

    def get_edges(self):
        edges = []
        for vertex in self.vertices:
            for neighbor, weight in self.adj_list[vertex]:
                edges.append((vertex, neighbor, weight))
        return edges
    
    def __str__(self):
        graph_str = ""
        for vertex in self.vertices:
            graph_str += f"{vertex}: "
            for neighbor, weight in self.adj_list[vertex]:
                graph_str += f"({neighbor}, {weight}) "
            graph_str += "\n"
        return graph_str

    def journey(self, start, end):
        current = start
        while(current != end):
            neighbors = self.get_neighbors(current)

class Grid:
    def __init__(self, numpy_grid, walk_function = np.vectorize(lambda x: x), growth_function = np.vectorize(lambda x: 0)):
        self.numpy_grid = numpy_grid
        self.growth_function = growth_function
        self.walk_function = walk_function
    def increment(self):
        growth_grid = self.growth_function(self.numpy_grid)
        self.walk()
        self.numpy_grid += growth_grid
        return self.numpy_grid
    def walk(self, time):
        grid_donating = time*self.walk_function(self.numpy_grid)
        for (i, j), value in np.ndenumerate(self.numpy_grid):
            L = self.adj(i, j)
            for neighbor in L:
                self.numpy_grid[i, j] += (grid_donating[neighbor[0], neighbor[1]] / len(self.adj(neighbor[0], neighbor[1])))
        self.numpy_grid -= grid_donating
        return self.numpy_grid
    def adj(self, x, y):
        neighbors = []
        rows, cols = self.numpy_grid.shape
        if x + 1 < rows:  # down
            neighbors.append([x+1, y])
        if x - 1 >= 0:    # up
            neighbors.append([x-1, y])
        if y + 1 < cols:  # right
            neighbors.append([x, y+1])
        if y - 1 >= 0:    # left
            neighbors.append([x, y-1])
        return neighbors
    def normalize(self):
        grid_sum = np.sum(self.numpy_grid)
        self.numpy_grid /= grid_sum

class TwoGrid(Grid):
    def __init__(self, g1, g2, combine_growth_function1 = lambda xy: 0.5*(0.5*xy[0] - xy[0]*xy[1]), combine_growth_function2 = lambda ab: 0.5*(ab[0]*ab[1]-ab[1])):
        self.g1 = g1
        self.g2 = g2
        self.cgf1 = 0 if self.g1.numpy_grid.all() < 0 else combine_growth_function1
        self.cgf2 = 0 if self.g1.numpy_grid.all() < 0 else combine_growth_function2
    def increment(self, time):
        # predator_past_grid = np.copy(self.g2.numpy_grid)
        L = [self.g1.numpy_grid, self.g2.numpy_grid]
        growth_grid1 = self.cgf1(L)
        growth_grid2 = self.cgf2(L)
        self.walk(time)
        self.g1.numpy_grid += time*growth_grid1
        self.g2.numpy_grid += time*growth_grid2
        return (self)
    def walk(self, time):
        self.g1.walk(time)
        self.g2.walk(time)
    def print(self):
        print("prey")
        print(self.g1.numpy_grid)
        print("predator")
        print(self.g2.numpy_grid)
    def animate(self, time, frames=1, interval=100):
        fig = plt.figure(figsize=(10, 5))
        ax_prey = fig.add_subplot(121, projection='3d')
        ax_predator = fig.add_subplot(122, projection='3d')

        ax_prey.set_title("Prey Grid")
        ax_predator.set_title("Predator Grid")

        x = np.arange(self.g1.numpy_grid.shape[1])
        print(self.g1.numpy_grid.shape[1])  # Corrected for full grid
        y = np.arange(self.g1.numpy_grid.shape[0])
        self.g1.numpy_grid.shape[0]  # Corrected for full grid
        X, Y = np.meshgrid(x, y)

        def update(_):
            self.increment(time)
            ax_prey.clear()
            ax_predator.clear()

            ax_prey.set_title("Prey Grid")
            ax_predator.set_title("Predator Grid")

            ax_prey.set_xlim(0, self.g1.numpy_grid.shape[1]-1)  # Corrected for full grid
            ax_prey.set_ylim(0, self.g1.numpy_grid.shape[0]-1)  # Corrected for full grid
            ax_prey.set_zlim(0, 2)

            ax_predator.set_xlim(0, self.g2.numpy_grid.shape[1]-1)  # Corrected for full grid
            ax_predator.set_ylim(0, self.g2.numpy_grid.shape[0]-1)  # Corrected for full grid
            ax_predator.set_zlim(0, 1)
            self.print()
            ax_prey.plot_surface(X, Y, self.g1.numpy_grid, cmap='Blues')
            ax_predator.plot_surface(X, Y, self.g2.numpy_grid, cmap='Reds')
            
        ani = FuncAnimation(fig, update, frames=frames, interval=interval)
        plt.show()
class FireNode: 
    def __init__(self, firestate, counter): 
        self.firestate = 0
        self.count = 0
        self.terrain_score = 1
class FireModel:
    def __init__(self, size):
        self.fire_grid = np.array([[FireNode(0, 0) for _ in range(size)] for _ in range(size)])
        self.stochastic_grid = np.zeros((size, size))
    def setscore(self, i, j, score):
        self.fire_grid[i, j].terrain_score = score
    def increment(self, wind_function):
        for (i, j), value in np.ndenumerate(self.stochastic_grid):
            if (self.fire_grid[i, j].firestate == 1):
                self.fire_grid[i, j].count += 1
                if self.fire_grid[i, j].count >= 7: 
                    self.fire_grid[i, j].firestate = -1
            random_counter = np.random.uniform(0, 1)
            if random_counter < value:
                if(self.fire_grid[i, j].firestate == 0):
                    self.fire_grid[i, j].firestate = 1
            shift_matrix_x = np.zeros((3, 3))
            shift_matrix_y = np.zeros((3, 3))
        x = wind_function[0] * math.cos(wind_function[1])
        y = wind_function[0] * math.sin(wind_function[1])
        x_index = 0
        y_index = 0
        xExistence = False
        yExistence = False
        if (x > 0):
            x_index = (1, 2)
            shift_matrix_x[1, 2] = x
            shift_matrix_x[1, 1] = 1
            xExistence = True
        elif(x < 0):
            x_index = (1, 0)
            shift_matrix_x[1, 0] = x
            shift_matrix_x[1, 1] = 1   
            xExistence = True 
        if(y > 0):
            y_index = (0, 1)
            shift_matrix_y[0, 1] = y
            shift_matrix_y[1, 1] = 1
            yExistence = True
        elif(y < 0):
            y_index = (2, 1)
            shift_matrix_y[2, 1] = y
            shift_matrix_y[1, 1] = 1
            yExistence = True
        list_shift = []
        for(i, j), value in np.ndenumerate(self.fire_grid):
            if(value.firestate > 0):
                shift = np.array([[0, 0, 0], [0, value.firestate, 0], [0, 0, 0]])
                if (xExistence == True):
                    shift_matrix_x[x_index] /= value.terrain_score
                    shift = np.matmul(shift, shift_matrix_x)
                if (yExistence == True):
                    shift_matrix_y[y_index] /= value.terrain_score
                    shift = np.matmul(shift_matrix_y, shift)
                shift = shift / np.sum(shift)
                list_shift.append((shift, i, j))
        self.stochastic_grid = np.zeros_like(self.stochastic_grid)
        for shift, i, j in list_shift: 
            rows, cols = self.stochastic_grid.shape
            if i - 1 < 0 and j - 1 < 0:  # Top-left corner
                self.stochastic_grid[i:i+2, j:j+2] += shift[1:3, 1:3]
            elif i - 1 < 0 and j + 1 >= cols:  # Top-right corner
                self.stochastic_grid[i:i+2, j-1:j+1] += shift[1:3, 0:2]
            elif i + 1 >= rows and j - 1 < 0:  # Bottom-left corner
                self.stochastic_grid[i-1:i+1, j:j+2] += shift[0:2, 1:3]
            elif i + 1 >= rows and j + 1 >= cols:  # Bottom-right corner
                self.stochastic_grid[i-1:i+1, j-1:j+1] += shift[0:2, 0:2]
            elif i - 1 < 0:  # Top edge
                self.stochastic_grid[i:i+2, j-1:j+2] += shift[1:3, 0:3]
            elif i + 1 >= rows:  # Bottom edge
                self.stochastic_grid[i-1:i+1, j-1:j+2] += shift[0:2, 0:3]
            elif j - 1 < 0:  # Left edge
                self.stochastic_grid[i-1:i+2, j:j+2] += shift[0:3, 1:3]
            elif j + 1 >= cols:  # Right edge
                self.stochastic_grid[i-1:i+2, j-1:j+1] += shift[0:3, 0:2]
            else:  # General case
                self.stochastic_grid[i-1:i+2, j-1:j+2] += shift
    def print(self):
        firestate_array = np.array([[node.firestate for node in row] for row in self.fire_grid])
        print(firestate_array)
        print("---------------------")
    def forecast(self, num_samples, wind_function, depth):
        samples = np.zeros((self.fire_grid.shape[0], self.fire_grid.shape[1]))
        fire_matrix = copy.deepcopy(self.fire_grid)
        for j in range(num_samples):
            for i in range(depth): 
                self.increment(wind_function(i))
            samplex = np.array([[1 if node.firestate in [1, -1] else 0 for node in row] for row in self.fire_grid])
            samples += samplex
            self.fire_grid = copy.deepcopy(fire_matrix)
            self.stochastic_grid -= self.stochastic_grid
        samples /= num_samples
        roundsamples = np.round(samples, 3)
        plot(self, roundsamples)
        return roundsamples
    
def plot(firemodel, samples):
    terrain_scores = np.array([[node.terrain_score for node in row] for row in firemodel.fire_grid])

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Terrain Score Grid
    ax1 = axes[0]
    im1 = ax1.imshow(terrain_scores, cmap='Blues', interpolation='nearest')
    ax1.set_title('Terrain Score Grid')
    ax1.set_xlabel('X-axis')
    ax1.set_ylabel('Y-axis')
    fig.colorbar(im1, ax=ax1, label='Terrain Score')

    # Probability Grid
    ax2 = axes[1]
    im2 = ax2.imshow(samples, cmap='Reds', interpolation='nearest')
    ax2.set_title('Probability Grid')
    ax2.set_xlabel('X-axis')
    ax2.set_ylabel('Y-axis')
    fig.colorbar(im2, ax=ax2, label='Probability')

    plt.tight_layout()
    plt.show()

vertices = ['A', 'B', 'C', 'D', 'E']

# graph = Graph(vertices)

# graph.add_edge('A', 'B', 4)
# graph.add_edge('A', 'C', 2)
# graph.add_edge('B', 'E', 3)
# graph.add_edge('C', 'D', 2)
# graph.add_edge('C', 'E', 5)
# graph.add_edge('D', 'E', 1)

# print("Graph:")
# print(graph)

# print("Vertices:", graph.get_vertices())
# print("Edges:", graph.get_edges())
# print("Neighbors of C:", graph.get_neighbors('C'))

# wind_list = [[20, 0], [10, 0], [3, 0]]
wind_function1 = lambda x: [1, x]
fire_matrix = FireModel(10)
for i in range(10):
    fire_matrix.setscore(i, 8, 2)
    fire_matrix.setscore(3, i, 10)
fire_matrix.fire_grid[5, 5].firestate = 1
fire_matrix.forecast(1000, wind_function1, 10)

# print(f"Execution time: {end_time - start_time} seconds")

# samples = np.zeros((10, 10))
# num_samples = 3000  # Number of samples for the sampling distribution

# start_time = time.time()

# for _ in range(num_samples):
#     fire_matrix = FireModel(10)  # Reset the fire model for each sample
#     for i in range(10):
#         fire_matrix.setscore(i, 4, 2)
#         fire_matrix.setscore(i, 3, 10)
#     fire_matrix.fire_grid[5, 5].firestate = 1  # Initialize fire at the center
#     for i in range(10):  # Simulate 10 increments
#         # fire_matrix.increment(wind_function2(i))
#         fire_matrix.increment(wind_function1(i))
#     samplex = np.array([[1 if node.firestate in [1, -1] else 0 for node in row] for row in fire_matrix.fire_grid])
#     samples += samplex

# samples /= num_samples
# samples = np.round(samples, 3)
# print(samples)

# end_time = time.time()
# print(f"Execution time: {end_time - start_time} seconds")

predator = np.ones((4, 4))
prey = np.ones((4, 4))
walkfunc = lambda x: 0.5*x
grid1 = Grid(predator, walkfunc)
grid2 = Grid(prey, walkfunc)
lotka = TwoGrid(grid1, grid2)
lotka.animate(time=0.05, frames=200, interval=50)