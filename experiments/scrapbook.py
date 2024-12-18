import torch
import grid
from trainer import GridAutoencoderTrainer, GridCounterTrainer

########################### BRESENHAM LINE ################################

# grid_factory = grid.GridFactory(5, 5)
# grid = grid_factory.generate_random_line(mixin_amount=0.1)

# print(grid)

########################### GRID AUTOENCODER ##############################

# gridAutoencoderTrainer = GridAutoencoderTrainer(10, 10)

# training_phases = [
#     GridAutoencoderTrainer.TrainingPhase("random_lines", 1000, 20),
#     GridAutoencoderTrainer.TrainingPhase("random", 1000, 5),
#     GridAutoencoderTrainer.TrainingPhase("random_lines_mixin_0.1", 1000, 50),
#     GridAutoencoderTrainer.TrainingPhase("random_lines", 1000, 20),
#     GridAutoencoderTrainer.TrainingPhase("random_lines", 1000, 1000)
# ]
# gridAutoencoderTrainer.train(training_phases, force_retrain=False)
# gridAutoencoderTrainer.demonstrate()

############################## GRID COUNTER ###############################

gridCounterTrainer = GridCounterTrainer(32, 64, 10)
gridCounterTrainer.demonstrate()

