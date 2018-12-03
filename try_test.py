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
from input_pipeline import CustomDataPreprocessorForCNN, CustomDatasetForCNN
from try_model import CNNTrajNet, reshape_output, displacement_error, final_displacement_error, time_elapsed

# test_log_file format: example #, test_loss, disp_error, fina_disp_error
# Note: last row is the average loss

# test_results.pkl format: target and pred pair in a tuple: [(2m X T, 2m X T), (2m X T, 2m X T),...]

# maybe can specificy epoch_num in the file names?

def test(args, model, device, test_loader):
    model.eval()
    loss_func = nn.MSELoss()
    test_loss = 0
    disp_error = 0
    fina_disp_error = 0
    pred_target_pair_list = []
    test_log_directory = 'log/'
    test_log_file = open(os.path.join(test_log_directory, 'test_errors_wi_testset_'+str(args.testset)+'.txt'), 'w')
    i = 0
    # losses = []
    with torch.no_grad():
        for data, target in test_loader:
            i += 1
            data, target = data.to(device), target.to(device)
            target = target.float()
            pred = torch.squeeze(model(data), 2) # 1 X 2m X T
            target_pred_pair_list.append((torch.squeeze(target).cpu().numpy(), torch.squeeze(pred).cpu().numpy())) # 2m X T, save target and pred pair

            current_test_loss = loss_func(pred, target).item() # sum up batch loss
            current_disp_error = displacement_error(reshape_output(pred, mode ='disp'), reshape_output(target, mode ='disp')).item()
            current_fina_disp_error = final_displacement_error(reshape_output(pred, mode ='f_disp'), reshape_output(target, mode ='f_disp')).item()
            test_loss += current_test_loss
            disp_error += current_disp_error
            fina_disp_error += current_fina_disp_error

            test_log_file.write(str(i)+','+str(current_test_loss)+','+str(current_disp_error)+','+str(current_fina_disp_error)+'\n')
            # losses.append(test_loss)
    test_loss /= len(test_loader.dataset)
    disp_error /= len(test_loader.dataset)
    fina_disp_error /= len(test_loader.dataset)
    test_log_file.write('average,'+str(test_loss)+','+str(disp_error)+','+str(fina_disp_error)+'\n')
    print('\nTest set: Average loss: {:.4f}, disp error: {:.4f}, final disp error: {:.4f}\n'.format(
        test_loss, disp_error, fina_disp_error))
    with open(os.path.join(test_log_directory, 'test_results_wi_testset_'+str(args.testset)+'.pkl'), 'wb') as f: # format: [(2m X T, 2m X T), (2m X T, 2m X T),...]
        pickle.dump(target_pred_pair_list, f)
    test_log_file.close()
    return [test_loss, disp_error, fina_disp_error]


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CNNTrajNet')
    parser.add_argument('--epoch_num', type=int, default=100, 
                        help='number of epochs to train')
    test_args = parser.parse_args()

    start = time.time()
    
    save_directory = 'save/'
    with open(os.path.join(save_directory, 'train_config.pkl'), 'rb') as f:
        saved_args = pickle.load(f)

    use_cuda = not saved_args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(saved_args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    model = CNNTrajNet(saved_args).to(device)

    # Get the checkpoint path
    checkpoint_path = os.path.join(save_directory, 'model_'+str(saved_args.testset)+'_'+str(test_args.epoch_num)+'.tar')
    if os.path.isfile(checkpoint_path):
        print('Loading checkpoint')
        checkpoint = torch.load(checkpoint_path)
        # model_iteration = checkpoint['iteration']
        model_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print('Loaded checkpoint at epoch', model_epoch)
    else:
        sys.exit('Execution stopped: model for test set '+str(saved_args.testset)+', epoch '+str(test_args.epoch_num)+' does not exist for loading')

    # Test data loading
    processor = CustomDataPreprocessorForCNN(input_seq_length=saved_args.input_size, pred_seq_length=saved_args.output_size, test_data_sets = saved_args.testset, dev_ratio_to_test_set = saved_args.dev_ratio, forcePreProcess=saved_args.forcePreProcess)
    test_set = CustomDatasetForCNN(processor.processed_test_data_file)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=saved_args.batch_size, shuffle=True)

    test_losses = test(saved_args, model, device, test_loader)    # didn't use test_losses for anything for now
    elapsed_time = time.time() - start
    if elapsed_time < 1.0:
        print('Time elapsed less than a second')
    else:
        print('Time elapsed: {}'.format(time_elapsed(elapsed_time)))

   

if __name__ == '__main__':
    main()
