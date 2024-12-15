import grid
from trainer import GridAutoencoderTrainer

########################### BRESENHAM LINE ################################

# grid_factory = grid.GridFactory(10, 10)
# grid = grid_factory.generate_random_line()

# print(grid)

########################### GRID AUTOENCODER ##############################

gridAutoencoderTrainer = GridAutoencoderTrainer(10, 10)
gridAutoencoderTrainer.train("random_lines", 1000, 20, force_retrain=True)
gridAutoencoderTrainer.train("random", 1000, 10, force_retrain=True)
gridAutoencoderTrainer.train("random_lines", 1000, 20, force_retrain=True)
gridAutoencoderTrainer.train("random", 1000, 10, force_retrain=True)
gridAutoencoderTrainer.train("random_lines", 1000, 50, force_retrain=True)
gridAutoencoderTrainer.demonstrate()

###########################################################################