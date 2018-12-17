'''
input_pipeline_fill_0_mix_all_data: 

Before:
--> Dumping dev data with size 7167 to pickle file
--> Dumping test data with size 7167 to pickle file
--> Dumping train data with size 57336 to pickle file

After deleting zero rows:
--> Dumping dev data with size 7141 to pickle file
--> Dumping test data with size 7148 to pickle file
--> Dumping train data with size 57171 to pickle file
'''

import os
import sys
import torch
import torch.utils.data
import pickle
import numpy as np
import random
from tqdm import tqdm


model_n = input('Please enter a model # (e.g. 1, 2, 3...): ')
if model_n == str(1):
    print('Dataset: normal, mix_all_data')
    from input_pipeline_mix_all_data import CustomDataPreprocessorForCNN, CustomDatasetForCNN
elif model_n == str(2):
    print('Dataset: normal, specify_test_set')
    from input_pipeline import CustomDataPreprocessorForCNN, CustomDatasetForCNN
elif model_n == str(3):
    print('Dataset: fill_0, mix_all_data')
    from input_pipeline_fill_0_mix_all_data import CustomDataPreprocessorForCNN, CustomDatasetForCNN
elif model_n == str(4):
    print('Dataset: fill_0, specify_test_set ')
    from input_pipeline_fill_0 import CustomDataPreprocessorForCNN, CustomDatasetForCNN
elif model_n == str(5):
    print('Dataset: individual, mix_all_data')
    from input_pipeline_individual_pedestrians_mix_all_data import CustomDataPreprocessorForCNN, CustomDatasetForCNN
elif model_n == str(6):
    print('Dataset: individual, specify_test_set')
    from input_pipeline_individual_pedestrians import CustomDataPreprocessorForCNN, CustomDatasetForCNN
else:
    sys.exit('Execution stopped 1: please check and re-run')
     
print(' ')
answer = None
while answer not in ('y', 'n'):
    answer = input('Do you want to force preprocessing data? (y/n)')
    if answer == 'y':
        print('Confirmed forcePreProcess=True')
        if_force_preprocess = True
    elif answer == 'n':
        print('Confirmed forcePreProcess=False')
        if_force_preprocess = False
    else:
        print('Please enter y or n')

# Data preprocessor.
if model_n == str(2):
    processor = CustomDataPreprocessorForCNN(forcePreProcess=if_force_preprocess, test_data_sets=[30,35], dev_ratio_to_test_set = 0.8, augmentation=True, pre_dir='model_'+model_n)
elif model_n == str(1) or model_n == str(3) or model_n == str(5):
    processor = CustomDataPreprocessorForCNN(dev_ratio=0.1, test_ratio=0.1, forcePreProcess=if_force_preprocess, augmentation=True, pre_dir='model_'+model_n)
elif model_n == str(4) or model_n == str(6):
    processor = CustomDataPreprocessorForCNN(forcePreProcess=if_force_preprocess, test_data_sets=[30,35], dev_ratio_to_test_set = 0.5, augmentation=True, pre_dir='model_'+model_n)
else:
    sys.exit('Execution stopped 2: please check and re-run')


# Processed datasets. (training/dev/test)
train_set = CustomDatasetForCNN(processor.processed_train_data_file)
dev_set = CustomDatasetForCNN(processor.processed_dev_data_file)
test_set = CustomDatasetForCNN(processor.processed_test_data_file)


def nonzero_row_index(inp): # inp is a 2m X T tensor
    if inp.size()[0]%2 == 1:
        sys.exit('Execution stopped at nonzero_row_index(): Error: 2m % 2 == 0 is not satisfied')
    sr = torch.sum(inp, dim=1) # size = [nrow, 1] # nrow must be an even number
    sr_ind_tensor = (sr != 0).nonzero() # nonzero index
    ind_in_1Darray =  sr_ind_tensor.numpy().ravel() # index in 1Darray
    insert_at = []
    insert_value = []
    append_value = False
    for i in range(ind_in_1Darray.shape[0]):
        if ind_in_1Darray[i]%2 == 0: # an even index means an x coordinate
            if i == (ind_in_1Darray.shape[0] - 1):
                append_value = True
            elif ind_in_1Darray[i+1] != (ind_in_1Darray[i] + 1):
                insert_at.append(i+1)
                insert_value.append(ind_in_1Darray[i] + 1)
        else: # an odd index means a y coordinate
            if ind_in_1Darray[i-1] != (ind_in_1Darray[i] - 1):
                insert_at.append(i)
                insert_value.append(ind_in_1Darray[i] - 1)
    new_ind = np.insert(ind_in_1Darray, insert_at, insert_value)
    if append_value:
        new_ind = np.append(new_ind, new_ind[-1]+1)
    return new_ind

def delete_all_zero_rows(dataset_pair): # any data set from CustomDatasetForCNN 
    new_data_pair = []
    for input_output_pair in tqdm(dataset_pair):
        data_input, data_output = input_output_pair[0], input_output_pair[1]
        nonzero_ind = np.intersect1d(nonzero_row_index(data_input), nonzero_row_index(data_output))
        if nonzero_ind.size == 0: # if there is no nonzero rows
            continue
        else:
            new_data_pair.append((data_input[nonzero_ind,:], data_output[nonzero_ind,:]))
    return new_data_pair


if (model_n == str(3) or model_n == str(4)) and if_force_preprocess:
    print('Deleting all-0 rows and re-dumping...')
    new_dev_set = delete_all_zero_rows(dev_set)
    print('--> Dumping dev data with size ' + str(len(new_dev_set)) + ' to pickle file')
    f_dev = open(processor.processed_dev_data_file, 'wb')
    pickle.dump(new_dev_set, f_dev, protocol=2)
    f_dev.close()

    new_test_set = delete_all_zero_rows(test_set)
    print('--> Dumping test data with size ' + str(len(new_test_set)) + ' to pickle file')
    f_test = open(processor.processed_test_data_file, 'wb')
    pickle.dump(new_test_set, f_test, protocol=2)
    f_test.close()

    new_train_set = delete_all_zero_rows(train_set)
    print('--> Dumping train data with size ' + str(len(new_train_set)) + ' to pickle file')
    f_train = open(processor.processed_train_data_file, 'wb')
    pickle.dump(new_train_set, f_train, protocol=2)
    f_train.close()

print('--> Saving the scaling factors (' +str(processor.scale_factor_x)+' and '+str(processor.scale_factor_y)+') to pickle file')
f_scaling = open(os.path.join(processor.data_dir, "scaling_factors.cpkl"), 'wb')
pickle.dump((processor.scale_factor_x, processor.scale_factor_y), f_scaling, protocol=2)
f_scaling.close()

print("Train set number of examples: {}".format(len(train_set)))
print("Dev set size number of examples: {}".format(len(dev_set)))
print("Test set size number of examples: {}".format(len(test_set)))
