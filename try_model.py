"""
Experiments on the model:
Note 1: max pooling makes performance worse FF.leaky_relu(F.max_pool2d(self.bn1(x), kernel_size = ...)) 
kernel_size = 2, stride=1, padding = 1, dilation=2 => initial error: 0.08
kernel_size = 3, stride=1, padding = 1 => initial error: 0.14
Note 2: adding an extra fc at output doesn't help; adding a 5th conv layer worsens result; increasing conv channels worsens result
Note 3: a larger dev set portion results in worse train error => more training data could help
Note 4: dropout doesn't seem to help dev error but could worsen train performance
Note 5: changing learning rate from 0.001 to 0.0001 or adding learning rate decay doesn't seem to make a difference
Note 6: Adam is better than SGD
Note 7: adding batch norm and relu to intermediate fc layer doesn't help
Note 8: changing relu to leaky_relu improves a lot??? + dev error is much smaller than train error???
* current version is the best version *
"""
# ---------  things to do: --------
# 11/26 possible things to try:
# look at some resulted trajectories in plot
# ----------------------------------
import os
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

        self.input_embedding_layer = nn.Linear(1, self.embedding_size) # assume embedding_size = 24

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

        # # add dropout
        # if self.dropout_rate != 0:
        #     x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.output_fc(x) # (N, W, H, C) = 1 X 2m X 1 X T
        return F.leaky_relu(x)


def train(args, model, device, train_loader, optimizer, epoch, log_detailed_file):
    save_directory = 'save/'
    with open(os.path.join(save_directory, 'train_config.pkl'), 'wb') as f:
        pickle.dump(args, f)
    def checkpoint_path(epoch_num):
        return os.path.join(save_directory, 'model_'+str(args.testset)+'_'+str(epoch_num)+'.tar') # careful args.testset is a list..

    model.train()
    loss_func = nn.MSELoss()
    train_loss = 0
    # losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        target = target.float()
        optimizer.zero_grad()
        output = torch.squeeze(model(data), 2) # 1 X 2m X T
        loss = loss_func(output, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        # losses.append(loss.item())
        if args.verbose and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            log_detailed_file.write(str(epoch)+','+str(batch_idx * len(data))+','+str(loss.item())+'\n')

    train_loss /= len(train_loader.dataset)
    print('average train loss for Epoch {} is: {:.4f}'.format(epoch, train_loss))


    if epoch % args.save_every == 0:
        print('Saving model')
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, checkpoint_path(epoch))
        print('model saved')

    return train_loss



def vali(args, model, device, dev_loader):
    model.eval()
    loss_func = nn.MSELoss()
    dev_loss = 0
    disp_error = 0
    fina_disp_error = 0
    # losses = []
    with torch.no_grad():
        for data, target in dev_loader:
            data, target = data.to(device), target.to(device)
            target = target.float()
            pred = torch.squeeze(model(data), 2) # 1 X 2m X T
            dev_loss += loss_func(pred, target).item() # sum up batch loss
            disp_error += displacement_error(reshape_output(pred, mode ='disp'), reshape_output(target, mode ='disp')).item()
            fina_disp_error += final_displacement_error(reshape_output(pred, mode ='f_disp'), reshape_output(target, mode ='f_disp')).item()
            # losses.append(dev_loss)
            
    dev_loss /= len(dev_loader.dataset)
    disp_error /= len(dev_loader.dataset)
    fina_disp_error /= len(dev_loader.dataset)
    print('\nDev set: Average loss: {:.4f}, disp error: {:.4f}, final disp error: {:.4f}\n'.format(
        dev_loss, disp_error, fina_disp_error))
    return [dev_loss, disp_error, fina_disp_error]

def reshape_output(s, mode ='disp'):
    """
    Input:  
    - s: (batch, 2 Ped #, seq_len) = 1 X 2m X T
    - mode ('disp' or 'f_disp'): select between shapes for disp_error and final_disp_error, repsecitvely
    Output: 
    - disp: (batch, seq_len, 2) = m X T X 2
    - fina_disp: m X 2
    """
    # print(s.size())
    s_2D = torch.squeeze(s) # 2m X T  
    s_cluster = torch.split(s, 2, dim=1) # (m X T, m X T, m X T, ...)
    s_stack = torch.stack(s_cluster) # m X 1 X 2 X T
    s_stack = torch.squeeze(s_stack, 1) # m X 2 X T
    s_stack_trans = torch.transpose(s_stack, 1, 2) # m X T X 2

    if  mode == 'disp':
        return s_stack_trans
    elif mode == 'f_disp':
        return torch.squeeze(torch.split(s_stack_trans, 1, dim=1)[-1], 1)



def displacement_error(pred_traj, pred_traj_gt, mode='average'):
    """
    Input:  
    - pred_traj: Tensor of shape (m, seq_len, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (m, seq_len, 2). Ground truth
    predictions.
    - mode: Can be one of sum, average
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
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch CNNTrajNet')
    parser.add_argument('--input_size', type=int, default=5) # input sequence length
    parser.add_argument('--output_size', type=int, default=5) # prediction sequence length
    parser.add_argument('--batch_size', type=int, default=1, 
                        help='minibatch (default: 1)')
    parser.add_argument('--epochs', type=int, default=100, 
                        help='number of epochs to train (default: 100)')

    parser.add_argument('--dev_ratio', type=int, default=0.1,      # not using dev set for now
                        help='the ratio of dev set to test set')
    parser.add_argument('--testset', type=list, default=[2],     
                        help='test_data_sets (default: [2])')
    parser.add_argument('--forcePreProcess', type=bool, default=False,     
                        help='forcePreProcess (default: False)')
    # Frequency at which the model should be saved parameter
    parser.add_argument('--save_every', type=int, default=1,      # don't forget
                        help='save frequency')
    # Dropout not implemented.
    # Dropout probability parameter
    parser.add_argument('--dropout_rate', type=float, default=0.4,       # not using dropout for now
                        help='dropout probability (default: 0.2)')
    # Dimension of the embeddings parameter
    parser.add_argument('--embedding_size', type=int, default=32,
                        help='Embedding dimension for the spatial coordinates')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                        help='learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, 
                        help='learning rate decay rate')
    # Lambda regularization parameter (L2)
    parser.add_argument('--lambda_param', type=float, default=0.0005,
                        help='L2 regularization parameter')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--verbose', type=bool, default=True,     
                        help='printing log')

    start = time.time()
    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    # Data preprocessor
    processor = CustomDataPreprocessorForCNN(input_seq_length=args.input_size, pred_seq_length=args.output_size, test_data_sets = args.testset, dev_ratio_to_test_set = args.dev_ratio, forcePreProcess=args.forcePreProcess)
    # Processed datasets. (training/dev/test)
    print("Loading data from the pickle files. This may take a while...")
    train_set = CustomDatasetForCNN(processor.processed_train_data_file)
    dev_set = CustomDatasetForCNN(processor.processed_dev_data_file)
    test_set = CustomDatasetForCNN(processor.processed_test_data_file)
    # Use DataLoader object to load data. Note batch_size=1 is necessary since each datum has different rows (i.e. number of pedestrians).
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.batch_size, shuffle=True)
    dev_loader = torch.utils.data.DataLoader(dataset=dev_set, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=True)

    print("Training set size: {}".format(len(train_loader)))
    print("Dev set size: {}".format(len(dev_loader)))
    print("Test set size: {}".format(len(test_loader)))
    
    # ---------  leave room for a resume option--------
    # add a resume option to continue training from a existing presaved model
    # ----------------------------------
    log_directory = 'log/'   # log_file format: epoch, average_train_loss, dev_loss, disp_error, final_disp_error
    log_file = open(os.path.join(log_directory, 'train_errors_per_epoch_excluding_testset_'+str(args.testset)+'.txt'), 'w')
    # log_detailed_file format: epoch, batch/example_index, train error for that batch/example at that epoch
    log_detailed_file = open(os.path.join(log_directory, 'train_errors_for_every_'+str(args.log_interval)+'th_batch_excluding_testset_'+str(args.testset)+'.txt'), 'w')


    device = torch.device("cuda" if use_cuda else "cpu")
    model = CNNTrajNet(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay=args.lambda_param)
    # optimizer = torch.optim.SGD(model.parameters(), lr= args.learning_rate, momentum=0.9, weight_decay=args.lambda_param)

    for epoch in range(1, args.epochs + 1):
        # adjust_learning_rate(optimizer, epoch, args.lr_decay_rate, args.learning_rate)
        train_losses = train(args, model, device, train_loader, optimizer, epoch, log_detailed_file)   #--------- use losses to graph? ---------
        log_file.write(str(epoch)+','+str(train_losses)+',')

        dev_losses = vali(args, model, device, dev_loader)     
        log_file.write(str(dev_losses[0])+','+str(dev_losses[1])+','+str(dev_losses[2])+'\n')
        print('finish epoch {}; time elapsed: {}'.format(epoch,  time_elapsed(time.time() - start)))                   
    log_file.close()
    log_detailed_file.close()



if __name__ == '__main__':
    main()
