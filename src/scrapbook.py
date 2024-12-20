import torch
from utilities import grid
from trainers.trainer import GridAutoencoderTrainer, GridCounterTrainer

########################### BRESENHAM LINE ################################

# num_dots = 10

# grid_factory = grid.GridFactory(10, 10)
# for _ in range(10000):
#     grid = grid_factory.generate_random_spaced_dots(num_dots=num_dots)
#     assert(torch.sum(grid == 1.0).item() == num_dots)

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
# gridCounterTrainer.train(force_retrain=True)
gridCounterTrainer.demonstrate()