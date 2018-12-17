import os
import sys
import torch
import torch.utils.data
import pickle
import numpy as np
import random
import time
import timeit
import argparse
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from try_model import CNNTrajNet, reshape_output, displacement_error, final_displacement_error, time_elapsed, rescaled_for_loss, traj_items
'''
Model numbering:
(1) normal, mix_all_data               - done with wrong logged losses
(2) normal, specify_test_set           - done with wrong logged losses
(3) fill_0, mix_all_data               - done with wrong logged losses - started Sat night, finished Sun night
(4) fill_0, specify_test_set           - done: started Sun night, finished Mon morning
(5) individual, mix_all_data           - done: started Sun night, finished Mon midnight - needs to multiply loss by 100
(6) individual, specify_test_set       - training on AWS halfway, started Mon midnight; re-training on laptop, Mon noon

path example: try_cnn_rerun/model_1/old_save/model_[2]_51.tar
              try_cnn_rerun/rerun_for_dev.py

'''

def calc_loss(args, model, device, dev_loader, x_scal, y_scal, epoch, save_directory):
    model.eval()
    loss_func = nn.MSELoss()
    dev_loss = 0
    disp_error = 0
    fina_disp_error = 0
    target_pred_pair_list = []
    with torch.no_grad():
        for data, target in dev_loader:
            data, target = data.to(device), target.to(device)
            target = target.float()
            pred = torch.squeeze(model(data), 2) # 1 X 2m X T or batch X 2 X T
            loss = loss_func(rescaled_for_loss(pred, x_scal, y_scal), rescaled_for_loss(target, x_scal, y_scal))  # careful to take scale from args
            dev_loss += loss.item() # sum up batch loss
            disp_error += displacement_error(reshape_output(pred,x_scal, y_scal, mode ='disp'), reshape_output(target, x_scal, y_scal, mode ='disp')).item()
            fina_disp_error += final_displacement_error(reshape_output(pred, x_scal, y_scal, mode ='f_disp'), reshape_output(target, x_scal, y_scal, mode ='f_disp')).item()
            target_pred_pair_list.extend(traj_items(args.batch_size, data, target, pred))
    dev_loss /= len(dev_loader)
    disp_error /= len(dev_loader)     
    fina_disp_error /= len(dev_loader)   
    with open(os.path.join(save_directory, 'dev_trajectories_'+str(epoch)+'.pkl'), 'wb') as f: 
        pickle.dump(target_pred_pair_list, f)
    return [dev_loss, disp_error, fina_disp_error]


def train_loss_epoch(epoch, detailed_train_loss, n): # n is the averged over divider
    return np.sum(detailed_train_loss[n*(epoch-1):n*(epoch-1)+4,2])/n



def main():
    model_n = input('Please enter a model # (e.g. 1, 2, 3...): ') # output a string
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

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CNNTrajNet')
    parser.add_argument('--epoch_num', type=int, default=500, 
                        help='number of epochs to train')
    rerun_args = parser.parse_args()

    start = time.time()
    
    rerun_save_directory = 'model_'+model_n+'/save/'
    rerun_log_directory = 'model_'+model_n+'/log/'


    rerun_old_save_directory = 'model_'+model_n+'/old_save/'
    rerun_old_log_directory = 'model_'+model_n+'/old_log/'


    with open(os.path.join(rerun_old_save_directory, 'train_config.pkl'), 'rb') as f:   
        saved_args = pickle.load(f)

    use_cuda = not saved_args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(saved_args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    model = CNNTrajNet(saved_args).to(device)



    # Data preprocessor.
    if model_n == str(2):
        processor = CustomDataPreprocessorForCNN(forcePreProcess=False, test_data_sets=[30,35], dev_ratio_to_test_set = 0.8, augmentation=True, pre_dir='model_'+model_n)
    elif model_n == str(1) or model_n == str(3) or model_n == str(5):
        processor = CustomDataPreprocessorForCNN(dev_ratio=0.1, test_ratio=0.1, forcePreProcess=False, augmentation=True, pre_dir='model_'+model_n)
    elif model_n == str(4) or model_n == str(6):
        processor = CustomDataPreprocessorForCNN(forcePreProcess=False, test_data_sets=[30,35], dev_ratio_to_test_set = 0.5, augmentation=True, pre_dir='model_'+model_n)
    else:
        sys.exit('Execution stopped: please check and re-run')

    # Processed datasets. (training/dev/test)
    # train_set = CustomDatasetForCNN(processor.processed_train_data_file)
    dev_set = CustomDatasetForCNN(processor.processed_dev_data_file)
    # train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=saved_args.batch_size, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=saved_args.batch_size, shuffle=True)

    with open(os.path.join(processor.data_dir, 'scaling_factors.cpkl'), 'rb') as f:
        x_y_scale = pickle.load(f)
    # find the last epoch number
    loss_file_name = os.path.join(rerun_old_log_directory, 'train_errors_per_epoch_excluding_testset_'+str(saved_args.testset)+'.txt')
    if os.path.isfile(loss_file_name):
        all_loss = np.loadtxt(loss_file_name, delimiter=',')
        end_epoch = all_loss.shape[0] 
        print('Re-running for dev errors till Epoch {}...'.format(end_epoch))
    else:
        sys.exit('Execution stopped: Error: cannot find train_errors_per_epoch_excluding_testset_'+str(saved_args.testset)+'.txt')

    # retrieve the detailed train losses and averaged over the whole epoch
    detailed_loss_file_name = os.path.join(rerun_old_log_directory, 'train_errors_for_every_'+str(saved_args.log_interval)+'th_batch_excluding_testset_'+str(saved_args.testset)+'.txt')
    if os.path.isfile(detailed_loss_file_name):
        detailed_train_loss = np.loadtxt(detailed_loss_file_name, delimiter=',')
        count_epoch = 1; train_average_over = 0
        while count_epoch!=2:
            count_epoch = detailed_train_loss[train_average_over,0]
            train_average_over += 1
        train_average_over -= 1
    else:
        sys.exit('Execution stopped: Error: cannot find the detailed log file')

    # start iterations
    averg_epoch_n = 20
    accum_dev_loss = np.zeros((averg_epoch_n, 3))
    log_file = open(os.path.join(rerun_log_directory, 'train_errors_per_epoch_excluding_testset_'+str(saved_args.testset)+'.txt'), 'w')
    for epoch in range(1, end_epoch + 1):
        # Get the checkpoint path
        checkpoint_path = os.path.join(rerun_old_save_directory, 'model_'+str(saved_args.testset)+'_'+str(epoch)+'.tar')
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['state_dict'])  # will state dict be re-written in each epoch iteration??????????
        else:
            sys.exit('Execution stopped: model for test set '+str(saved_args.testset)+', epoch '+str(epoch)+' does not exist for loading')

        curr_dev_losses = calc_loss(saved_args, model, device, dev_loader, x_y_scale[0], x_y_scale[1], epoch, rerun_save_directory)

        # average out the dev_loss over averg_epoch_n epochs
        if epoch < averg_epoch_n + 1:
            for i in range(3):
                accum_dev_loss[epoch-1, i] = curr_dev_losses[i]
            dev_losses = list(np.sum(accum_dev_loss, axis=0)/epoch)
        else:
            accum_dev_loss = np.delete(accum_dev_loss, 0, 0)  # delete the first object on axis = 0 (delete a row)
            accum_dev_loss = np.append(accum_dev_loss, [curr_dev_losses], axis=0) # append at the end
            dev_losses = list(np.sum(accum_dev_loss, axis=0)/averg_epoch_n)

        train_losses = np.sum(detailed_train_loss[train_average_over*(epoch-1):train_average_over*(epoch-1)+4,2])/train_average_over
        log_file.write(str(epoch)+','+str(train_losses)+',' + str(dev_losses[0])+','+str(dev_losses[1])+','+str(dev_losses[2])+'\n')  #--str(train_losses)!!!!!
        # log_file.write(str(epoch)+','+ str(dev_losses[0])+','+str(dev_losses[1])+','+str(dev_losses[2])+'\n')  #--str(train_losses)!!!!!
        print('Epoch {}: Train loss: {:.5f}, Dev loss: {:.8f}, disp error: {:.4f}, final disp error: {:.4f}'.format(
        epoch, train_losses, dev_losses[0], dev_losses[1], dev_losses[2]))

    log_file.close()
    print('Finished re-running for dev errors; time elapsed: {}'.format(time_elapsed(time.time() - start)))


   

if __name__ == '__main__':
    main()
