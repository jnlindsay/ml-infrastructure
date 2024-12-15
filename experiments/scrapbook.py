import grid
from trainer import GridAutoencoderTrainer

########################### BRESENHAM LINE ################################

# grid_factory = grid.GridFactory(10, 10)
# grid = grid_factory.generate_random_line()

# print(grid)

########################### GRID AUTOENCODER ##############################

gridAutoencoderTrainer = GridAutoencoderTrainer()
gridAutoencoderTrainer.train()
gridAutoencoderTrainer.demonstrate()

###########################################################################