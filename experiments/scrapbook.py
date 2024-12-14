import grid

# Bresenham line

grid_factory = grid.GridFactory(10, 10)
grid = grid_factory.generate_random_line()

print(grid)