"""

Kian's advice:
1. definitely train longer
2. tune regularization/dropout at different layers in network; try different combinations
3. check training set bad examples, does it correspond to any dev set error. 
5. mix some test sets wtih training. I'm thinking maybe try several sets at a time as testing sets, and to be mixed with train sets. 
maybe test on a Stanford dataset?? since the majority of data is from Stanford
6. learning rate decay....just try hard try different hyperparameters. 

decoupling pedestrian
just for debugging, mixing all data before seperating

[normal, fill_0, individual] x [mix_all_data, specify_test_set]

(1) normal, mix_all_data               - done with wrong logged losses
(2) normal, specify_test_set           - done with wrong logged losses
(3) fill_0, mix_all_data               - done with wrong logged losses - started Sat night, finished Sun night
(4) fill_0, specify_test_set           - done: started Sun night, finished Mon morning
(5) individual, mix_all_data           - done: started Sun night, finished Mon midnight - needs to multiply loss by 100 - only runs 3 hours - fixed
(6) individual, specify_test_set       - training on AWS halfway, started Mon midnight; re-training on laptop, Mon noon


12/06 Things to do:
2. do resume training, plotting loss curve based on a certain trained model in the middle of training
5. After decoupling pedestrains. change kernel size to 2 and stride of 2   （??? what does it mean）

documentation & analysis
1. e.g analyze if trying permute make a difference. Just a bunch of tries.



12/9 NOTE:

1. current loss is for each pedistrain in a sequence of T. 

2. training before 12/9 Sunday night (affecting (1)(2)(3)): 
(a) has wrong disp and final_disp errors, as well as train and dev losses, which are incorrectly divided by m^2 (only need to be divided by m once); 
values will be  * inconsistent *  with previous trainings. 

(b) Luckily, log_detailed_file still has correct train losses for individual epoches; also dev losses can be corrected by re-running all saved models. 


************************************************************************
------------------- PLEASE READ BEFORE TRAINING ------------------- 
************************************************************************
Specs: 
dev_ratio = 0.5 when test set can be selected; dev_ratio = test_ratio = 0.1 for mixed datasets. 
epochs = 500 # just want to train as long duration as possible; mannually stop (Ctl+C) anytime if needs to stop


current version:
lambda_param = 0.001
dropout_rate = 0.06
delete_all_zero_rows defaults to True
"""
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
# from input_pipeline import CustomDataPreprocessorForCNN, CustomDatasetForCNN

class CNNTrajNet(nn.Module):

    def __init__(self, args):   
        '''
        Initializer function
        args: Training arguments
        infer: Training or test time (true if test time)
        '''
        super(CNNTrajNet, self).__init__()
        self.args = args
        self.embedding_size = args.embedding_size
        self.input_size = args.input_size # means the input sequence length
        self.output_size = args.output_size

        self.input_embedding_layer = nn.Linear(1, self.embedding_size) # assume embedding_size = 32

        self.conv1 = nn.Conv2d(in_channels = self.input_size, out_channels = 2*self.input_size, kernel_size = 3, padding = (0,2), dilation=2)
        self.bn1 = nn.BatchNorm2d(2*self.input_size) # padding 1 to keep the same size
        self.conv2 = nn.Conv2d(in_channels = 2*self.input_size, out_channels = 4*self.input_size, kernel_size = 3, padding = (0,2), dilation=2)
        self.bn2 = nn.BatchNorm2d(4*self.input_size)
        self.conv3 = nn.Conv2d(in_channels = 4*self.input_size, out_channels = 6*self.input_size, kernel_size = 3, padding = (0,2), dilation=2)
        self.bn3 = nn.BatchNorm2d(6*self.input_size)
        self.conv4 = nn.Conv2d(in_channels = 6*self.input_size, out_channels = 8*self.input_size, kernel_size = 3, padding = (0,2), dilation=2)
        self.bn4 = nn.BatchNorm2d(8*self.input_size)

        # self.conv5 = nn.Conv2d(in_channels = 8*self.input_size, out_channels = 10*self.input_size, kernel_size = 3, padding = (0,2), dilation=2)
        # self.bn5 = nn.BatchNorm2d(10*self.input_size)

        self.interm_fc1 = nn.Linear(8*16*self.input_size, 8*self.input_size)
        # self.interm_fc1_bn = nn.BatchNorm2d(8*self.input_size)
        # self.interm_fc2 = nn.Linear(8*8*self.input_size, 8*self.output_size)
        self.output_fc = nn.Linear(8*self.input_size, self.output_size)

        # ReLU and dropout unit
        # self.relu = nn.ReLU()
        # self.conv2_drop = nn.Dropout2d()
        # self.dropout = nn.Dropout(args.dropout)
        self.dropout_rate = args.dropout_rate

    def forward(self, x):
        """
        input: assume x is input_sequence from one example of (input_sequence, prediction_sequence) 
        x inital size: 1 X 2m X t
        1 is the batch_size, m is the # of pedestrians in that one example, t is the input sequence length
        """
        x = x.float()
        x = torch.unsqueeze(x, 3)  # 1 X 2m X t_input X 1
        x = F.leaky_relu(self.input_embedding_layer(x)) # 1 X 2m X t_input X 32

        x = torch.transpose(x, 1, 3) # (N, H, C, W) = 1 X 32 X t X 2m
        x = torch.transpose(x, 1, 2) # (N, C, H, W) = 1 X t X 32 X 2m


        x = self.conv1(x) # or FF.leaky_relu(F.max_pool2d(self.bn1(x), kernel_size = 2, stride=1, padding = 1, dilation=2)) error of 0.08
        x = F.leaky_relu(self.bn1(x))  
        x = self.conv2(x)
        x = F.leaky_relu(self.bn2(x)) 
        x = self.conv3(x)
        x = F.leaky_relu(self.bn3(x))
        x = self.conv4(x)
        x = F.leaky_relu(self.bn4(x))   # (N, C, H, W) = 1 X 8t X 16 X 2m

        # x = self.conv5(x)
        # x = F.leaky_relu(self.bn5(x))   # (N, C, H, W) = 1 X 10t X 12 X 2m

        x = torch.cat(torch.split(x, 1, dim=1), 2) # (N, H, C, W) = 1 X 1 X 8t*16 X 2m

        x = torch.transpose(x, 2, 3) # (N, H, W, C) = 1 X 1 X 2m X 8t*16
        x = torch.transpose(x, 1, 2) # (N, W, H, C) = 1 X 2m X 1 X 8t*16

        x = self.interm_fc1(x) # (N, W, H, C) = 1 X 2m X 1 X 8t*8
        # x = self.interm_fc2(x) # (N, W, H, C) = 1 X 2m X 1 X 8t

        # # add batch norm and reulu to the itermediate fc layer
        # x = self.interm_fc1_bn(torch.transpose(x, 1, 3)) # (N, C, H, W) = 1 X 8t X 1 X 2m
        # x = F.leaky_relu(torch.transpose(x, 1, 3))

        # add dropout
        if self.dropout_rate != 0:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.output_fc(x) # (N, W, H, C) = 1 X 2m X 1 X T
        return F.leaky_relu(x)



def rescaled_for_loss(inpp,x_scaling_factor,y_scaling_factor): # careful it changes (x,y,x,y) to (x,x,y,y) # size is unchanged though
    return torch.cat((inpp[:, ::2]*x_scaling_factor, inpp[:, 1::2]*y_scaling_factor), 1)  # careful to take scale from args

def traj_items(batch_size, data, target, output):
    if batch_size == 1:
        return [(torch.squeeze(data).cpu().numpy(), torch.squeeze(target).cpu().numpy(), torch.squeeze(output).cpu().detach().numpy())]
    else:
        data_2D = torch.squeeze(torch.cat(torch.split(data, 1, dim=0), 1)) # data: batch X 2 X T => data_2D: batch*2 X T
        target_2D = torch.squeeze(torch.cat(torch.split(target, 1, dim=0), 1)) # target: batch X 2 X T => target_2D: batch*2 X T
        output_2D = torch.squeeze(torch.cat(torch.split(output, 1, dim=0), 1)) # output: batch X 2 X T => output_2D: batch*2 X T
        if batch_size < 6:
            return [(data_2D.cpu().numpy(), target_2D.cpu().numpy(), output_2D.cpu().detach().numpy())]
        else: 
            item_list = []
            data_2D_cluster = torch.split(data_2D, 5*2, dim=0) # 5 pedestrians in one trajectory plot ("5" is just arbitrarily chosen, can be any number)
            target_2D_cluster = torch.split(target_2D, 5*2, dim=0)
            output_2D_cluster = torch.split(output_2D, 5*2, dim=0)
            for i in range(len(data_2D_cluster)):
                item_list.append((data_2D_cluster[i].cpu().numpy(), target_2D_cluster[i].cpu().numpy(), output_2D_cluster[i].cpu().detach().numpy()))
            return item_list



def train(args, model, device, train_loader, optimizer, epoch, log_detailed_file, x_scal, y_scal):
    save_directory = 'save/'
    def checkpoint_path(epoch_num):
        return os.path.join(save_directory, 'model_'+str(args.testset)+'_'+str(epoch_num)+'.tar') # careful args.testset is a list..
    model.train()
    loss_func = nn.MSELoss()
    train_loss = 0
    target_pred_pair_list = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        target = target.float()
        optimizer.zero_grad()
        output = torch.squeeze(model(data), 2) # batch X 2m X T
        # m = target.size()[1]/2 # m = 1 for individual example case (namely when batch_size != 1). 
        loss = loss_func(rescaled_for_loss(output, x_scal, y_scal), rescaled_for_loss(target, x_scal, y_scal))  # careful to take scale from args
        
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        # target_pred_pair_list.append((torch.squeeze(data).cpu().numpy(), torch.squeeze(target).cpu().numpy(), torch.squeeze(output).cpu().detach().numpy())) 
        target_pred_pair_list.extend(traj_items(args.batch_size, data, target, output))



        # losses.append(loss.item())
        if args.verbose and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            log_detailed_file.write(str(epoch)+','+str(batch_idx * len(data))+','+str(loss.item())+'\n')

    train_loss /= len(train_loader) # changed from divded by n_examples
    print('average train loss for Epoch {} is: {:.8f}'.format(epoch, train_loss))


    with open(os.path.join(save_directory, 'train_trajectories_'+str(epoch)+'.pkl'), 'wb') as f: 
        pickle.dump(target_pred_pair_list, f)

    if epoch % args.save_every == 0:
        print('Saving model')
        # time.sleep(10) # to avoid failture in synchronizing before saving out a CUDA model
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path(epoch))
        print('model saved')

    return train_loss



def vali(args, model, device, dev_loader, x_scal, y_scal, epoch):
    save_directory = 'save/'
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


def reshape_output(s, x_scale, y_scale, mode ='disp'):
    """
    Input:  
    - s: (batch, 2 Ped #, seq_len) = 1 X 2m X T
    - mode ('disp' or 'f_disp'): select between shapes for disp_error and final_disp_error, repsecitvely
    Output: 
    - disp: (batch, seq_len, 2) = m X T X 2
    - fina_disp: m X 2
    """
    s_2D = torch.squeeze(torch.cat(torch.split(s, 1, dim=0), 1)) # data: batch X 2m X T => data_2D: batch*2m X T
    s_3D = torch.unsqueeze(s_2D, 0) #1 X batch*2m X T
    s_cluster = torch.split(s_3D, 2, dim=1) # (1 X 2 X T, 1 X 2 X T, 1 X 2 X T, ...) 
    s_cluster_cat = torch.cat(s_cluster,0) # m X 2 X T
    s_stack_trans = torch.transpose(s_cluster_cat, 1, 2) # m X T X 2
    x = torch.unsqueeze(s_stack_trans[:,:,0], 2) # m X T X 1
    y = torch.unsqueeze(s_stack_trans[:,:,1], 2)
    s_stack_trans_scaled = torch.cat((x*x_scale, y*y_scale), 2) # m X T X 2

    if  mode == 'disp':
        return s_stack_trans_scaled
    elif mode == 'f_disp':
        return torch.squeeze(torch.split(s_stack_trans_scaled, 1, dim=1)[-1], 1)



def displacement_error(pred_traj, pred_traj_gt, mode='average'):
    """
    Input:  
    - pred_traj: Tensor of shape (m, seq_len, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (m, seq_len, 2). Ground truth
    predictions.
    - mode: Can be sum or average
    Output:
    - loss: gives the eculidian displacement error
    """
    m, seq_len, _ = pred_traj.size()
    loss = pred_traj_gt - pred_traj
    loss = loss**2
    loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) # m
    loss = torch.sum(loss)/m # a scalar: loss for each person
    if mode == 'sum':
        return loss # a tensor
    elif mode == 'average':
        return loss/seq_len  # a tensor


def final_displacement_error(pred_pos, pred_pos_gt):
    """
    Input:
    - pred_pos: Tensor of shape (m, 2). Predicted last pos.
    - pred_pos_gt: Tensor of shape (m, 2). Groud truth
    Output:
    - loss: gives the eculidian displacement error
    """
    m, _ = pred_pos.size()
    loss = (pred_pos_gt - pred_pos)**2
    loss = torch.sqrt(loss.sum(dim=1)) # m
    loss = torch.sum(loss)/m # a scalar: loss for each person
    return loss  # a tensor

def adjust_learning_rate(optimizer, epoch, decay_rate, original_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = 1./(1+decay_rate*epoch)*original_lr

def time_elapsed(elapsed_seconds):
    seconds = int(elapsed_seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    periods = [('hours', hours), ('minutes', minutes), ('seconds', seconds)]
    return  ', '.join('{} {}'.format(value, name) for name, value in periods if value)


def main():
    print('Please select from the following 6 datasets (be consistent with preprocessing done in train.py):')
    print('(1) normal, mix_all_data')
    print('(2) normal, specify_test_set')
    print('(3) fill_0, mix_all_data')
    print('(4) fill_0, specify_test_set ')
    print('(5) individual, mix_all_data')
    print('(6) individual, specify_test_set')
    model_n = input('Please enter a dataset # (e.g. 1, 2, 3...): ') # output a string
    # import datapipeline
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
    # set default_batch_size
    if model_n == str(1) or model_n == str(2) or model_n == str(3) or model_n == str(4):
        default_batch_size = 1
    elif model_n == str(5) or model_n == str(6):
        default_batch_size = 100
    else:
        sys.exit('Execution stopped 2: please check and re-run')


    print(' ')
    in_out_squ_bool = None
    while in_out_squ_bool not in ('y', 'n'):
        in_out_squ_bool = input('Which input-output sequence do you want to use? "y" for (8, 12),  "n" for (5, 5)')
        if in_out_squ_bool == 'y':
            print('Confirmed (8, 12)')
            in_out_seq = (8, 12)
            default_epochs = 1500
        elif in_out_squ_bool == 'n':
            print('Confirmed (5, 5)')
            in_out_seq = (5, 5)
            default_epochs = 500
        else:
            print('Please enter y or n')

    if in_out_squ_bool=='y' and (model_n == str(5) or model_n == str(6)):
    	default_log_interval = 100
    else:
    	default_log_interval = 1000


    answer = None
    while answer not in ('y', 'n'):
        print('\n(1) Did you save the log/save files from the last train model?')
        print('\n(2) Did you run train_fill_0.py to delete all-0 rows and re-dump the preprocessed data WHILE you need to?')
        answer = input('(y/n)')
        if answer == 'y':
            print('Great')
        elif answer == 'n':
            print('Execution stopped:')
            print('(1) You are not supposed to lose/overwrite log/save files')
            sys.exit('(2) Please delete all-0 rows before training fill_0 data')
        else:
            print('Please enter y or n')


    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CNNTrajNet')
    parser.add_argument('--input_size', type=int, default=in_out_seq[0]) # input sequence length
    parser.add_argument('--output_size', type=int, default=in_out_seq[1]) # prediction sequence length
    parser.add_argument('--batch_size', type=int, default=default_batch_size,  #  PLEASE use a batch size a mutiplier of 5 (e.g. 5, 10, 15, 20, ...)
                        help='minibatch (default: 1)')
    parser.add_argument('--epochs', type=int, default=default_epochs, 
                        help='number of epochs to train')

    parser.add_argument('--dev_ratio', type=float, default=0.5,      # not using this arg for now
                        help='the ratio of dev set to test set')
    parser.add_argument('--testset', type=list, default=[2],     
                        help='test_data_sets (default: [2])')
    parser.add_argument('--forcePreProcess', type=bool, default=False,     
                        help='forcePreProcess (default: False)')
    # Frequency at which the model should be saved parameter
    parser.add_argument('--save_every', type=int, default=1,      # don't forget
                        help='save frequency')
    # Dropout probability parameter
    parser.add_argument('--dropout_rate', type=float, default=0.06,       # dropout 
                        help='dropout probability')
    # Dimension of the embeddings parameter
    parser.add_argument('--embedding_size', type=int, default=32,
                        help='Embedding dimension for the spatial coordinates')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                        help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, 
                        help='learning rate decay rate')
    # Lambda regularization parameter (L2)
    parser.add_argument('--lambda_param', type=float, default=0.001, #0.01
                        help='L2 regularization parameter')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=default_log_interval,
                        help='how many batches to wait before logging training status')
    # parser.add_argument('--delete_all_zero_rows', type=bool, default=True,         # done in train_fill_0.py
    #                     help='if needs to delete all zero rows')
    parser.add_argument('--verbose', type=bool, default=True,     
                        help='printing log')

    start = time.time()
    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)



    # Data preprocessor.
    if model_n == str(2):
        processor = CustomDataPreprocessorForCNN(input_seq_length=in_out_seq[0], pred_seq_length=in_out_seq[1], forcePreProcess=False, test_data_sets=[30,35], dev_ratio_to_test_set = 0.8, augmentation=True)
    elif model_n == str(1) or model_n == str(3) or model_n == str(5):
        processor = CustomDataPreprocessorForCNN(input_seq_length=in_out_seq[0], pred_seq_length=in_out_seq[1], dev_ratio=0.1, test_ratio=0.1, forcePreProcess=False, augmentation=True)
    elif model_n == str(4) or model_n == str(6):
        processor = CustomDataPreprocessorForCNN(input_seq_length=in_out_seq[0], pred_seq_length=in_out_seq[1], forcePreProcess=False, test_data_sets=[30,35], dev_ratio_to_test_set = 0.5, augmentation=True)
    else:
        sys.exit('Execution stopped 3: please check and re-run')


    # Processed datasets. (training/dev/test)
    print("Loading data from the pickle files. This may take a while...")
    train_set = CustomDatasetForCNN(processor.processed_train_data_file)
    dev_set = CustomDatasetForCNN(processor.processed_dev_data_file)
    test_set = CustomDatasetForCNN(processor.processed_test_data_file)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=True)

    print("Train set size (n_example/batch_size): {}".format(len(train_loader)))
    print("Dev set size (n_example/batch_size): {}".format(len(dev_loader)))
    print("Test set size (n_example/batch_size): {} (not used in training)".format(len(test_loader)))

    with open(os.path.join(processor.data_dir, 'scaling_factors.cpkl'), 'rb') as f:
        x_y_scale = pickle.load(f)
    print("The scaling factors: {}, {}".format(x_y_scale[0], x_y_scale[1]))
    
    # ---------  leave room for a resume option--------
    # add a resume option to continue training from a existing presaved model
    # ----------------------------------
    with open(os.path.join('save/', 'train_config.pkl'), 'wb') as f:
        pickle.dump(args, f)
        
    log_directory = 'log/'   # log_file format: epoch, average_train_loss, dev_loss, disp_error, final_disp_error
    log_file = open(os.path.join(log_directory, 'train_errors_per_epoch_excluding_testset_'+str(args.testset)+'.txt'), 'w')
    # log_detailed_file format: epoch, batch/example_index, train error for that batch/example at that epoch
    log_detailed_file = open(os.path.join(log_directory, 'train_errors_for_every_'+str(args.log_interval)+'th_batch_excluding_testset_'+str(args.testset)+'.txt'), 'w')


    device = torch.device("cuda" if use_cuda else "cpu")
    model = CNNTrajNet(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay=args.lambda_param)
    # optimizer = torch.optim.SGD(model.parameters(), lr= args.learning_rate, momentum=0.9, weight_decay=args.lambda_param)

    averg_epoch_n = 20
    accum_dev_loss = np.zeros((averg_epoch_n, 3))
    for epoch in range(1, args.epochs + 1):
        # adjust_learning_rate(optimizer, epoch, args.lr_decay_rate, args.learning_rate)
        # get train loss
        train_losses = train(args, model, device, train_loader, optimizer, epoch, log_detailed_file, x_y_scale[0], x_y_scale[1])   
        log_file.write(str(epoch)+','+str(train_losses)+',')
        # get dev loss
        curr_dev_losses = vali(args, model, device, dev_loader, x_y_scale[0], x_y_scale[1], epoch)  #[dev_error, disp_error, final_disp_error]
        # average out the dev_loss over averg_epoch_n epochs
        if epoch < averg_epoch_n + 1:
            for i in range(3):
                accum_dev_loss[epoch-1, i] = curr_dev_losses[i]
            dev_losses = list(np.sum(accum_dev_loss, axis=0)/epoch)
        else:
            accum_dev_loss = np.delete(accum_dev_loss, 0, 0)  # delete the first object on axis = 0 (delete a row)
            accum_dev_loss = np.append(accum_dev_loss, [curr_dev_losses], axis=0) # append at the end
            dev_losses = list(np.sum(accum_dev_loss, axis=0)/averg_epoch_n)
        log_file.write(str(dev_losses[0])+','+str(dev_losses[1])+','+str(dev_losses[2])+'\n')
        print('\nDev set: Average loss: {:.8f}, disp error: {:.4f}, final disp error: {:.4f}\n'.format(
        dev_losses[0], dev_losses[1], dev_losses[2]))


        print('finish epoch {}; time elapsed: {}'.format(epoch,  time_elapsed(time.time() - start)))
        print(' ')                 
    log_file.close()
    log_detailed_file.close()



if __name__ == '__main__':
    main()
