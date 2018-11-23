import os
import torch
import torch.utils.data
import pickle
import numpy as np
import random

from input_pipeline import CustomDataPreprocessorForCNN, CustomDatasetForCNN

# Data preprocessor.
processor = CustomDataPreprocessorForCNN(test_data_sets = [2], dev_fraction = 0.1, forcePreProcess=False)

# Processed datasets. (training/dev/test)
train_set = CustomDatasetForCNN(processor.processed_train_data_file)
dev_set = CustomDatasetForCNN(processor.processed_dev_data_file)
test_set = CustomDatasetForCNN(processor.processed_dev_data_file)

# Use DataLoader object to load data. Note batch_size=1 is necessary since each datum has different rows (i.e. number of pedestrians).
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=1, shuffle=True)
dev_loader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1, shuffle=True)

# Each example is a pair of (input_sequence, prediction_sequence). Each sequence is a torch tensor (matrix) where each row corresponds to 
# either x or y position of a pedestrian and each column corresponds to a frame.
print(next(iter(train_loader)))

print("  ")
print("enter for loop")
print("  ")
i = 0
for batch_idx, (data, target) in enumerate(train_loader):
	print(batch_idx)
	# print(data)
	print(list(data.size()))
	print(list(target.size()))
	print(" ")
	i = i + 1
	if i > 1:
		break



# argument list
# embedding_size = 32
# batch_size = 32
# learning_rate = 0.001
# optimizer = Adam
# loss = L2 loss
# dropout = 0.5
# input_size: so confused about padding embedding layer. The size depends on the # of pedestrains???
# output_size = pred_seq_length ?? 