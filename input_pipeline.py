import os
import torch
import torch.utils.data
import pickle
import numpy as np
import random

# -------------- Question 11/20 ---------------- #
# 1. test set hasn't been set up
# 2. certain pedestrain doesn't exit in the future. why not just omit
# 3. how confident about the way of this data preparing..

# 4. for batch_idx, (data, target) in enumerate(train_loader)

# 5. DataLoader object? shuffle
# 6. so confused about padding embedding layer. The size depends on the # of pedestrains???
# 7. conv2d size

# *ways to deal with variable input length:
# 1. feed one example at a time to CNN
# 2. remove all linear layers and build a pure CNN network
# 3. Pad zeros to keep the length

# *ways to handle outpout
# 1. average pooling
# 2. smart way to pad to make a m X L size of output



# -------------- Finish Questions ---------------- #

class CustomDataPreprocessorForCNN():
    def __init__(self, input_seq_length=5, pred_seq_length=5, datasets=[i for i in range(37)], test_data_sets = [2], dev_ratio_to_test_set = 0.1, forcePreProcess=False):
        '''
        Initializer function for the CustomDataSetForCNN class
        params:
        input_seq_length : input sequence length to be considered
        output_seq_length : output sequence length to be predicted
        datasets : The indices of the datasets to use
        test_data_sets : The indices of the test sets from datasets
        dev_ratio_to_test_set : ratio of the validation set size to the test set size
        forcePreProcess : Flag to forcefully preprocess the data again from csv files
        '''
        # List of data directories where raw data resides
        self.data_dirs = ['./data/train/processed/biwi/biwi_hotel', './data/train/processed/crowds/arxiepiskopi1',
                          './data/train/processed/crowds/crowds_zara02', './data/train/processed/crowds/crowds_zara03',
                          './data/train/processed/crowds/students001', './data/train/processed/crowds/students003', 
                          './data/train/processed/stanford/bookstore_0',
                          './data/train/processed/stanford/bookstore_1', './data/train/processed/stanford/bookstore_2',
                          './data/train/processed/stanford/bookstore_3', './data/train/processed/stanford/coupa_3',
                          './data/train/processed/stanford/deathCircle_0', './data/train/processed/stanford/deathCircle_1',
                          './data/train/processed/stanford/deathCircle_2', './data/train/processed/stanford/deathCircle_3',
                          './data/train/processed/stanford/deathCircle_4', './data/train/processed/stanford/gates_0',
                          './data/train/processed/stanford/gates_1', './data/train/processed/stanford/gates_3',
                          './data/train/processed/stanford/gates_4', './data/train/processed/stanford/gates_5',
                          './data/train/processed/stanford/gates_6', './data/train/processed/stanford/gates_7',
                          './data/train/processed/stanford/gates_8', './data/train/processed/stanford/hyang_4',
                          './data/train/processed/stanford/hyang_5', './data/train/processed/stanford/hyang_6',
                          './data/train/processed/stanford/hyang_7', './data/train/processed/stanford/hyang_9',
                          './data/train/processed/stanford/nexus_0', './data/train/processed/stanford/nexus_1',
                          './data/train/processed/stanford/nexus_2', './data/train/processed/stanford/nexus_3',
                          './data/train/processed/stanford/nexus_4', './data/train/processed/stanford/nexus_7',
                          './data/train/processed/stanford/nexus_8', './data/train/processed/stanford/nexus_9']
        train_datasets = datasets
        for dataset in test_data_sets:
            train_datasets.remove(dataset)
        self.train_data_dirs = [self.data_dirs[x] for x in train_datasets]
        self.test_data_dirs = [self.data_dirs[x] for x in test_data_sets]
        
        # Number of datasets
        self.numDatasets = len(self.data_dirs)
        
        # Data directory where the pre-processed pickle file resides
        self.data_dir = './data/train/processed'
        
        # Store the arguments
        self.input_seq_length = input_seq_length
        self.pred_seq_length = pred_seq_length
        
        # Validation arguments
        self.dev_fraction = dev_ratio_to_test_set
        
        # Define the path in which the process data would be stored
        self.processed_train_data_file = os.path.join(self.data_dir, "trajectories_cnn_train.cpkl")
        self.processed_dev_data_file = os.path.join(self.data_dir, "trajectories_cnn_dev.cpkl")
        self.processed_test_data_file = os.path.join(self.data_dir, "trajectories_cnn_test.cpkl")
        
        # If the file doesn't exist or forcePreProcess is true
        if not(os.path.exists(self.processed_train_data_file)) or forcePreProcess:
            print("------ Creating pre-processed training data for CNN ------")
            self.preprocess(self.train_data_dirs, self.processed_train_data_file)
        if not(os.path.exists(self.processed_dev_data_file)) or not(os.path.exists(self.processed_test_data_file)) or forcePreProcess:
            print("------ Creating pre-processed dev & test data for CNN ------")
            self.preprocess(self.test_data_dirs, self.processed_test_data_file, self.dev_fraction, self.processed_dev_data_file)
        
    def preprocess(self, data_dirs, data_file, dev_fraction = 0., data_file_2 = None):
        #frameList_data = []
        #pedsInFrameList_data = []
        #pedsPosInFrameList_data = []
        processed_input_output_pairs = []
        
        for directory in data_dirs:
            print('------ Processing dataset ' + str(directory) + ' ------')
            # define path of the csv file of the current dataset
            file_path = os.path.join(directory, 'world_pos_normalized.csv')
            
            # Load the data from the csv file
            data = np.genfromtxt(file_path, delimiter=',')
            
            # Frame IDs of the frames in the current dataset
            frameList = np.unique(data[0, :]).tolist()
            numFrames = len(frameList)
            
            # Add the list of frameIDs to the frameList_data
            #frameList_data.append(frameList)
            
            # For this dataset check which pedestrians exist in each frame.
            pedsInFrameList = []
            pedsPosInFrameList = []
            for ind, frame in enumerate(frameList):
                # For this frame check the pedestrian IDs.
                pedsInFrame = data[:, data[0, :] == frame]
                pedsList = pedsInFrame[1, :].tolist()
                pedsInFrameList.append(pedsList)
                # Position information for each pedestrian.
                pedsPos = []
                for ped in pedsList:
                    # Extract x and y positions
                    current_x = pedsInFrame[2, pedsInFrame[1, :] == ped][0]
                    current_y = pedsInFrame[3, pedsInFrame[1, :] == ped][0]
                    pedsPos.extend([current_x, current_y])
                pedsPosInFrameList.append(pedsPos)
            
            # Go over the frames in this data again to extract data.
            ind = 0  # frame index
            while ind < len(frameList) - (self.input_seq_length + self.pred_seq_length - 1):
                # List of pedestrians in this frame.
                pedsList = pedsInFrameList[ind]
                # Check if same pedestrians exist in the next (input_seq_length + pred_seq_length - 1) frames.
                peds_contained = True
                for ii in range(self.input_seq_length + self.pred_seq_length):
                    if pedsInFrameList[ind + ii] != pedsList:
                        peds_contained = False
                if peds_contained:
                    print(str(int(self.input_seq_length + self.pred_seq_length)) + ' frames starting from Frame ' + str(int(frameList[ind])) +  ' contain pedestrians ' + str(pedsList))
                    # Initialize numpy arrays for input-output pair
                    data_input = np.zeros((2*len(pedsList), self.input_seq_length))
                    data_output = np.zeros((2*len(pedsList), self.pred_seq_length))
                    for ii in range(self.input_seq_length):
                        data_input[:, ii] = np.array(pedsPosInFrameList[ind + ii])
                    for jj in range(self.pred_seq_length):
                        data_output[:, jj] = np.array(pedsPosInFrameList[ind + (self.input_seq_length - 1) + jj])
                    processed_pair = (torch.from_numpy(data_input), torch.from_numpy(data_output))
                    processed_input_output_pairs.append(processed_pair)
                    
                    # Perform data augmentation. Rotate (x,y)-coordinates from 5 deg to 355 deg with 5 deg space, and flip y. The amount of data is x144 the original data.
                    # # First, flip the original data.
                    data_input_yflipped = np.zeros_like(data_input)
                    data_output_yflipped = np.zeros_like(data_output)
                    for kk in range(len(pedsList)):
                        data_input_yflipped[2*kk, :] = data_input[2*kk, :]
                        data_input_yflipped[2*kk+1, :] = -1*data_input[2*kk+1, :]
                        data_output_yflipped[2*kk, :] = data_output[2*kk, :]
                        data_output_yflipped[2*kk+1, :] = -1*data_output[2*kk+1, :]
                    processed_pair_yflipped = (torch.from_numpy(data_input_yflipped), torch.from_numpy(data_output_yflipped))
                    processed_input_output_pairs.append(processed_pair_yflipped)
                    # # Then rotate by 5 deg sequentially and also flip for each rotated data
                    for deg in range(5, 360, 5):
                        data_input_rotated = np.zeros_like(data_input)
                        data_input_rotated_yflipped = np.zeros_like(data_input)
                        data_output_rotated = np.zeros_like(data_output)
                        data_output_rotated_yflipped = np.zeros_like(data_output)
                        rad = np.radians(deg)
                        c, s = np.cos(rad), np.sin(rad)
                        Rot = np.array(((c,-s), (s, c)))
                        for ii in range(len(pedsList)):
                            for jj in range(self.input_seq_length):
                                coordinates_for_this_ped = data_input[2*ii:2*(ii+1), jj]
                                new_coordinates_for_this_ped = np.dot(Rot, coordinates_for_this_ped)
                                data_input_rotated[2*ii:2*(ii+1), jj] = new_coordinates_for_this_ped
                            data_input_rotated_yflipped[2*ii, :] = data_input_rotated[2*ii, :]    
                            data_input_rotated_yflipped[2*ii+1, :] = -1*data_input_rotated[2*ii+1, :]
                            for jj in range(self.pred_seq_length):
                                coordinates_for_this_ped = data_output[2*ii:2*(ii+1), jj]
                                new_coordinates_for_this_ped = np.dot(Rot, coordinates_for_this_ped)
                                data_output_rotated[2*ii:2*(ii+1), jj] = new_coordinates_for_this_ped
                            data_output_rotated_yflipped[2*ii, :] = data_output_rotated[2*ii, :]
                            data_output_rotated_yflipped[2*ii+1, :] = -1*data_output_rotated[2*ii+1, :]
                        processed_pair_rotated = (torch.from_numpy(data_input_rotated), torch.from_numpy(data_output_rotated))
                        processed_input_output_pairs.append(processed_pair_rotated)   
                        processed_pair_rotated_yflipped = (torch.from_numpy(data_input_rotated_yflipped), torch.from_numpy(data_output_rotated_yflipped))
                        processed_input_output_pairs.append(processed_pair_rotated_yflipped)
                        
                    ind += self.input_seq_length +  self.pred_seq_length - 1
                else:
                    ind += 1
                
            #pedsInFrameList_data.append(pedsInFrameList)
            #pedsPosInFrameList_data.append(pedsPosInFrameList)
            
        # Shuffle data, possibly divide into train and dev sets.
        random.seed(1)
        random.shuffle(processed_input_output_pairs)
        if dev_fraction != 0.:
            assert(data_file_2 != None)
            dev_size = int(len(processed_input_output_pairs)*dev_fraction)
            processed_dev_set = processed_input_output_pairs[:dev_size]
            processed_test_set = processed_input_output_pairs[dev_size:]
            # Save processed data.
            f = open(data_file, 'wb')
            pickle.dump(processed_test_set, f, protocol=2)
            f.close()
            f2 = open(data_file_2, 'wb')
            pickle.dump(processed_dev_set, f2, protocol=2)
            f2.close()
        else:
            # Save processed data.
            f = open(data_file, 'wb')
            pickle.dump(processed_input_output_pairs, f, protocol=2)
            f.close()

class CustomDatasetForCNN(torch.utils.data.Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = open(self.file_path, 'rb')
        self.data = pickle.load(self.file)
        self.file.close()
    
    def __getitem__(self, index):
        item = self.data[index]
        return item
    
    def __len__(self):
        return len(self.data)    
