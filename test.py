import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

# Assume you have a 2D numpy array where each cell is a coordinate and its value represents the region it belongs to
grid = np.random.choice([0, 1, 2, 3], size=(1000, 1000))  # Randomly assign regions for this example

# Use Sobel filter to find edges (boundaries between regions)
sx = ndimage.sobel(grid, axis=0, mode='constant')
sy = ndimage.sobel(grid, axis=1, mode='constant')
edges = np.hypot(sx, sy)

# Plot the edges
plt.imshow(edges, cmap='gray')
plt.show()

