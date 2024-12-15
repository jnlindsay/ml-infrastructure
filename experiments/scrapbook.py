import grid
from trainer import GridAutoencoderTrainer

########################### BRESENHAM LINE ################################

# grid_factory = grid.GridFactory(10, 10)
# grid = grid_factory.generate_random_line()

# print(grid)

########################### GRID AUTOENCODER ##############################

gridAutoencoderTrainer = GridAutoencoderTrainer(10, 10)

training_phases = [
    GridAutoencoderTrainer.TrainingPhase("random_lines", 1000, 20),
    GridAutoencoderTrainer.TrainingPhase("random", 1000, 10),
    GridAutoencoderTrainer.TrainingPhase("random_lines", 1000, 20),
    GridAutoencoderTrainer.TrainingPhase("random", 1000, 10),
    GridAutoencoderTrainer.TrainingPhase("random_lines", 1000, 50)
]
gridAutoencoderTrainer.train(training_phases)
gridAutoencoderTrainer.demonstrate()

###########################################################################