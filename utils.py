import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageOps

def load_img(filename):
    img = Image.open(filename)
    img = ImageOps.grayscale(img)

    np_img = np.array(img)
    np_img = ~np_img
    np_img[np_img>0] = 1
    plt.set_cmap('binary')
    plt.imshow(np_img)

    # Save image
    np.save('map.npy', np_img)

def read_img(filename):
    grid = np.load(filename)
    plt.imshow(grid)
    plt.tight_layout()
    plt.show()

def map_to_grid(cell_x, cell_y, map):

    size_y, size_x = map.shape

    # new grid dimension
    n = size_x // cell_x
    m = size_y // cell_y
    
    # reshape to grid and value of the grid cell is the grid center
    grid_map = np.zeros((n, m))
    for i in range (n-1):
        for j in range (m-1):
            # Determine the center pixel of the current block
            center_y = j * cell_y + cell_y // 2
            center_x = i * cell_x + cell_x // 2

            # Assign the value of the center pixel to the grid cell
            grid_map[i, j] = map[center_y, center_x]
    
    return grid_map

def grid_point_to_map(cell_x, cell_y, grid_coordinate):
    map_x = grid_coordinate[0]*cell_x + cell_x//2
    map_y = grid_coordinate[1]*cell_y + cell_y//2

    return [map_x, map_y]