'''
write something
'''

import os
import sys
import torch
import torch.utils.data
import pickle
import numpy as np
import random
import time
import timeit
from tqdm import tqdm
import argparse
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from try_model import CNNTrajNet

def preparing_dataset(if_training, model_n):
    answer = None
    while answer not in ('y', 'n'):
        answer = input('\nWhich input-output sequence do you want to use? "y" for (8, 12),  "n" for (5, 5)')
        if answer == 'y':
            print('Confirmed (8, 12)')
            in_out_seq = (8, 12)
        elif answer == 'n':
            print('Confirmed (5, 5)')
            in_out_seq = (5, 5)
        else:
            print('Please enter y or n')
         
    answer = None
    while answer not in ('y', 'n'):
        answer = input('\nDo you want to force preprocessing data? (y/n)')
        if answer == 'y':
            print('Confirmed forcePreProcess=True')
            if_force_preprocess = True
        elif answer == 'n':
            print('Confirmed forcePreProcess=False')
            if_force_preprocess = False
        else:
            print('Please enter y or n')

    # Data preprocessor.
    pre_dir_path='.' if if_training else 'model_'+model_n
    if model_n == str(2):
        processor = CustomDataPreprocessorForCNN(input_seq_length=in_out_seq[0], pred_seq_length=in_out_seq[1], forcePreProcess=if_force_preprocess, test_data_sets=[30,35], dev_ratio_to_test_set = 0.8, augmentation=True, pre_dir=pre_dir_path)
    elif model_n == str(1) or model_n == str(3) or model_n == str(5):
        processor = CustomDataPreprocessorForCNN(input_seq_length=in_out_seq[0], pred_seq_length=in_out_seq[1], dev_ratio=0.1, test_ratio=0.1, forcePreProcess=if_force_preprocess, augmentation=True, pre_dir=pre_dir_path)
    elif model_n == str(4) or model_n == str(6):
        processor = CustomDataPreprocessorForCNN(input_seq_length=in_out_seq[0], pred_seq_length=in_out_seq[1], forcePreProcess=if_force_preprocess, test_data_sets=[30,35], dev_ratio_to_test_set = 0.5, augmentation=True, pre_dir=pre_dir_path)
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

    print('Saving the scaling factors and the global minimum values in x and y to pickle files...')
    print('Scaling factors: ' +str(processor.scale_factor_x)+' and '+str(processor.scale_factor_y))
    print('Global minimums: ' +str(processor.x_global_min)+' and '+str(processor.y_global_min))
    f_scaling = open(os.path.join(processor.data_dir, "scaling_factor_global_min.cpkl"), 'wb')
    pickle.dump((processor.scale_factor_x, processor.scale_factor_y, processor.x_global_min, processor.y_global_min), f_scaling, protocol=2)
    f_scaling.close()
    print("Train set number of examples: {}".format(len(train_set)))
    print("Dev set size number of examples: {}".format(len(dev_set)))
    print("Test set size number of examples: {}".format(len(test_set)))

    return (processor.scale_factor_x, processor.scale_factor_y, processor.x_global_min, processor.y_global_min)



def pred_traj_for_all(model, device, model_n, scale_min, test_batch_size): # model_n: a string number
    test_data_dir = 'model_'+model_n+'/data/test/'
    log_dir = 'model_'+model_n+'/data/pred_test/' # log out predicted sequences in a new directory all of txt files
    log_traj_dir = 'model_'+model_n+'/data/pred_test_traj/' # save trajectory sequences in a new directory all of pkl files
    dataset_dir_names = ['biwi/', 'crowds/', 'stanford/']
    # biwi_files = os.listdir(test_data_dir+'biwi/')
    # crowds_files = os.listdir(test_data_dir+'crowds/')
    # stanford_files = os.listdir(test_data_dir+'stanford/')
    # biwi_files = [os.path.join(test_data_dir+'biwi/', _path) for _path in os.listdir(test_data_dir+'biwi/')]
    # crowds_files = [os.path.join(test_data_dir+'crowds/', _path) for _path in os.listdir(test_data_dir+'crowds/')]
    # stanford_files = [os.path.join(test_data_dir+'stanford/', _path) for _path in os.listdir(test_data_dir+'stanford/')]
    # all_files = [biwi_files, crowds_files, stanford_files] # all test file paths as a list
    for i in range(len(dataset_dir_names)):
        files_in_one_dir = os.listdir(test_data_dir+dataset_dir_names[i])
        for j in range(len(files_in_one_dir)): # e.g number of txt files in '/stanford'
            curr_file = files_in_one_dir[j]
            txt_path = os.path.join(test_data_dir+dataset_dir_names[i], curr_file)
           

            trajnet_testset = preparing_trajnet_testset(txt_path, scale_min) # a single testset in one txt file
            trajnet_test_loader = torch.utils.data.DataLoader(dataset=trajnet_testset, batch_size=test_batch_size, shuffle=False)
            pred_traj_lists = pred_traj(test_batch_size, model, device, trajnet_test_loader, scale_min) # [pred_list, target_pred_pair_list] in numpy

            write_pred_txt(curr_file, txt_path, log_dir+dataset_dir_names[i], pred_traj_lists[0])
            with open(os.path.join(log_traj_dir, os.path.splitext(curr_file)[0]+'_traj.pkl'), 'wb') as f: 
                pickle.dump(pred_traj_lists[1], f)


def write_pred_txt(file_name, read_file_path, log_dir_path, log_content): # read and log files will be of the same names but in different directories
    log_file = open(os.path.join(log_dir_path, file_name), 'w')   # log_content: [np.array(2, T), np.array(2, T), ...] in physical coordinates
    line_n = -1 # line count
    pred_n = -1
    with open(read_file_path, 'r') as f:
        for line in f:
            line_n += 1
            if line_n%20 in range(0, 8):  # 0~7 # when in observation sequence
                log_file.write(line)
                if line_n%20 == 7:
                    pred_n += 1 # starts from 0
            else: # when in prediction sequence
                xy_array = log_content[pred_n] # np.array(2, T)
                ind = line_n%20 - 8 # 0~11
                line = line.strip().split(' ')
                log_file.write(line[0]+' '+line[1]+' '+str(xy_array[0,ind])+' '+str(xy_array[1,ind])+'\n')
    log_file.close()









def preparing_trajnet_testset(txt_path, scale_min):
    data = []
    test_obs = []
    c = -1 # line count
    with open(txt_path, 'r') as f:
        for line in f:
            c += 1
            if c%20 in range(0, 8):  # 0~7
                line = line.strip().split(' ')
                x = 1/scale_min[0]*(float(line[2]) - scale_min[2]) - 1.0
                y = 1/scale_min[1]*(float(line[3]) - scale_min[3]) - 1.0
                data.append([x, y])
                if c%20 == 7:
                    test_obs.append(torch.from_numpy(np.transpose(np.asarray(data)))) # [tensor(2, t), tensor(2, t), ...]
                    data = []
    return test_obs



def pred_traj(batch_size, model, device, test_loader, scale_min): # testing through forward propagation with the pre-trained model
    model.eval()
    target_pred_pair_list = []
    pred_list = []
    with torch.no_grad():
        for obs_seq in test_loader:
            obs_seq = obs_seq.to(device)
            pred = torch.squeeze(model(obs_seq), 2) # 1 X 2m X T or batch X 2 X T
            traj_lists = traj_items(batch_size, obs_seq, pred, scale_min)
            pred_list.extend(traj_lists[0])
            target_pred_pair_list.extend(traj_lists[1])
    return [pred_list, target_pred_pair_list] # a list of numpy arrays/tuple


def traj_items(batch_size, data, output, scale_min):
    if batch_size == 1:
        output_array = torch.squeeze(output)
        one_pred_list = [convert_normalized_to_physical(output_array, scale_min).cpu().numpy()]
        item_list = [(torch.squeeze(data).cpu().numpy(), output_array.cpu().numpy())]
        return [one_pred_list, item_list]
    else:
        data_2D = torch.squeeze(torch.cat(torch.split(data, 1, dim=0), 1)) # data: batch X 2 X T => data_2D: batch*2 X T
        output_2D = torch.squeeze(torch.cat(torch.split(output, 1, dim=0), 1)) # output: batch X 2 X T => output_2D: batch*2 X T
        # process for pred_list
        one_pred_list = []
        one_pred_2D_cluster = torch.split(output_2D, 2, dim=0)
        for i in range(len(one_pred_2D_cluster)):
            one_pred_list.append(convert_normalized_to_physical(one_pred_2D_cluster[i], scale_min).cpu().numpy()) # [np.array(2, T), np.array(2, T), ...]
        # process for target_pred_pair_list
        if batch_size < 6:
            item_list = [(data_2D.cpu().numpy(), output_2D.cpu().numpy())]
        else: 
            item_list = []
            data_2D_cluster = torch.split(data_2D, 5*2, dim=0) # 5 pedestrians in one trajectory plot ("5" is just arbitrarily chosen, can be any number)
            output_2D_cluster = torch.split(output_2D, 5*2, dim=0)
            for i in range(len(data_2D_cluster)):
                item_list.append((data_2D_cluster[i].cpu().numpy(), output_2D_cluster[i].cpu().numpy()))
        return [one_pred_list, item_list]

def convert_normalized_to_physical(seq, scale_min): # sequence in the form of a torch tensor 2 X T
    x_row = (seq[0,:]+1.0)*scale_min[0] + scale_min[2] # convert the normalized traj back to its physical form, dimension is unchanged
    y_row = (seq[1,:]+1.0)*scale_min[1] + scale_min[3]

    return torch.stack((x_row, y_row))
    # return torch.cat((x_row, y_row), 1)

def main():


    print('Please select from the following 6 datasets:')
    print('(1) normal, mix_all_data')
    print('(2) normal, specify_test_set')
    print('(3) fill_0, mix_all_data')
    print('(4) fill_0, specify_test_set ')
    print('(5) individual, mix_all_data')
    print('(6) individual, specify_test_set')
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

    # model_n = str(1)


    with open(os.path.join('model_'+model_n+'/', 'train_config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)
    if (saved_args.input_size != 8) or (saved_args.output_size != 12):
        sys.exit('Execution stopped: test and pre-trained model input/output sequence dimensions dismatch')

    parser = argparse.ArgumentParser(description='PyTorch CNNTrajNet dataset preparation')
    parser.add_argument('--if_training', type=bool, default=False) 
    parser.add_argument('--test_batch_size', type=int, default=100) 
    args = parser.parse_args()

    train_preprocess_path = 'model_'+model_n+'/data/train/processed/'
    scaling_factor_global_min_file = os.path.join(train_preprocess_path, 'scaling_factor_global_min.cpkl')
    if os.path.isfile(scaling_factor_global_min_file):
        with open(scaling_factor_global_min_file, 'rb') as f:
            scale_min = pickle.load(f)
        print("The scaling factors: {}, {}".format(scale_min[0], scale_min[1]))
        print("The global minimums: {}, {}".format(scale_min[2], scale_min[3]))
    else:
        scale_min = preparing_dataset(args.if_training, model_n)

    model_path = 'model_'+model_n+'/pretrained_models/'
    if os.path.isdir(model_path):
        model_filenames = os.listdir(model_path)
        model_file_paths = [os.path.join(model_path, file_name) for file_name in model_filenames]
    else:
        sys.exit('Execution stopped: pre-trained model directory does not exist at: '+model_path)


    use_cuda = not saved_args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(saved_args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    model = CNNTrajNet(saved_args).to(device)


# be sure to only have ONE model in the models folder, or predictions from the next model will overwrite!
    for checkpoint_path in model_file_paths:
        if os.path.splitext(checkpoint_path)[1] != '.tar':
            continue
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        pred_traj_for_all(model, device, model_n, scale_min, args.test_batch_size)


    


if __name__ == '__main__':
    main()
