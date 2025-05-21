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