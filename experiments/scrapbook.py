import torch
import grid
from trainer import GridAutoencoderTrainer
from models import GridCounter

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
# gridAutoencoderTrainer.train(training_phases, force_retrain=True)
# gridAutoencoderTrainer.demonstrate()

############################## GRID COUNTER ###############################

input_size = 32 # dimension of CNN output
hidden_size = 64
grid_size = 10

model = GridCounter(input_size, hidden_size, grid_size)

grid = torch.randn((2, 1, grid_size, grid_size)) # batch size of 2

final_count, final_mask = model(grid)
print("Final Count:", final_count)
print("Final Mask Memory:", final_mask.view(-1, grid_size, grid_size))