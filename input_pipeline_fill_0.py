import os
import torch
import torch.utils.data
import pickle
import numpy as np
import random
from tqdm import tqdm

class CustomDataPreprocessorForCNN():
    def __init__(self, input_seq_length=5, pred_seq_length=5, datasets=[i for i in range(37)], test_data_sets = [2], dev_ratio_to_test_set = 0.5, forcePreProcess=False, augmentation=False):
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
        self.data_paths = ['./data/train/raw/biwi/biwi_hotel.txt', './data/train/raw/crowds/arxiepiskopi1.txt',
                          './data/train/raw/crowds/crowds_zara02.txt', './data/train/raw/crowds/crowds_zara03.txt',
                          './data/train/raw/crowds/students001.txt', './data/train/raw/crowds/students003.txt', 
                          './data/train/raw/stanford/bookstore_0.txt',
                          './data/train/raw/stanford/bookstore_1.txt', './data/train/raw/stanford/bookstore_2.txt',
                          './data/train/raw/stanford/bookstore_3.txt', './data/train/raw/stanford/coupa_3.txt',
                          './data/train/raw/stanford/deathCircle_0.txt', './data/train/raw/stanford/deathCircle_1.txt',
                          './data/train/raw/stanford/deathCircle_2.txt', './data/train/raw/stanford/deathCircle_3.txt',
                          './data/train/raw/stanford/deathCircle_4.txt', './data/train/raw/stanford/gates_0.txt',
                          './data/train/raw/stanford/gates_1.txt', './data/train/raw/stanford/gates_3.txt',
                          './data/train/raw/stanford/gates_4.txt', './data/train/raw/stanford/gates_5.txt',
                          './data/train/raw/stanford/gates_6.txt', './data/train/raw/stanford/gates_7.txt',
                          './data/train/raw/stanford/gates_8.txt', './data/train/raw/stanford/hyang_4.txt',
                          './data/train/raw/stanford/hyang_5.txt', './data/train/raw/stanford/hyang_6.txt',
                          './data/train/raw/stanford/hyang_7.txt', './data/train/raw/stanford/hyang_9.txt',
                          './data/train/raw/stanford/nexus_0.txt', './data/train/raw/stanford/nexus_1.txt',
                          './data/train/raw/stanford/nexus_2.txt', './data/train/raw/stanford/nexus_3.txt',
                          './data/train/raw/stanford/nexus_4.txt', './data/train/raw/stanford/nexus_7.txt',
                          './data/train/raw/stanford/nexus_8.txt', './data/train/raw/stanford/nexus_9.txt']
        train_datasets = datasets
        for dataset in test_data_sets:
            train_datasets.remove(dataset)
        self.train_data_paths = [self.data_paths[x] for x in train_datasets]
        self.test_data_paths = [self.data_paths[x] for x in test_data_sets]
        print("Using the following dataset(s) as test set")
        print(self.test_data_paths)
        
        # Number of datasets
        self.numDatasets = len(self.data_paths)
        
        # Data directory where the pre-processed pickle file resides
        self.data_dir = './data/train/processed'
        
        # Store the arguments
        self.input_seq_length = input_seq_length
        self.pred_seq_length = pred_seq_length
        
        # Validation arguments
        self.dev_ratio = dev_ratio_to_test_set
        
        # Buffer for storing raw data.
        self.raw_data_train = []
        self.raw_data_test = []
        # Buffer for storing processed data.
        self.processed_input_output_pairs_train = []
        self.processed_input_output_pairs_test = []
        
        # Scale Factor for x and y (computed in self.process())
        self.scale_factor_x = None
        self.scale_factor_y = None
        
        # Data augmentation flag
        self.augmentation = augmentation
        # Rotation increment (deg) for data augmentation (only valid if augmentation is True)
        self.rot_deg_increment = 120
        # How many pedestrian permutations to consider (only valid if augmentation is True)
        self.permutations = 4
        
        # Define the path in which the process data would be stored
        self.processed_train_data_file = os.path.join(self.data_dir, "trajectories_cnn_train.cpkl")
        self.processed_dev_data_file = os.path.join(self.data_dir, "trajectories_cnn_dev.cpkl")
        self.processed_test_data_file = os.path.join(self.data_dir, "trajectories_cnn_test.cpkl")
        
        # If the file doesn't exist or forcePreProcess is true
        if not(os.path.exists(self.processed_train_data_file)) or not(os.path.exists(self.processed_dev_data_file)) or not(os.path.exists(self.processed_test_data_file)) or forcePreProcess:
            print("============ Normalizing raw data (after rotation data augmentation) ============")
            print("--> Finding max coordinate values for train data")
            x_max_train, x_min_train, y_max_train, y_min_train = self.find_max_coordinates(self.train_data_paths, self.raw_data_train)
            print("--> Finding max coordinate values for test data")
            x_max_test, x_min_test, y_max_test, y_min_test = self.find_max_coordinates(self.test_data_paths, self.raw_data_test)
            x_max_global, y_max_global = max([x_max_train, x_max_test]), max([y_max_train, y_max_test])
            x_min_global, y_min_global = min([x_min_train, x_min_test]), min([y_min_train, y_min_test])
            self.scale_factor_x = (x_max_global - x_min_global)/(1 + 1)
            self.scale_factor_y = (y_max_global - y_min_global)/(1 + 1)
            print("--> Normalizing train data")
            self.normalize(self.raw_data_train, x_max_global, x_min_global, y_max_global, y_min_global)
            print("--> Normalizing test data")
            self.normalize(self.raw_data_test, x_max_global, x_min_global, y_max_global, y_min_global)
            print("============ Creating pre-processed training data for CNN ============")
            self.preprocess(self.raw_data_train, self.processed_input_output_pairs_train, self.processed_train_data_file)
            print("============ Creating pre-processed dev & test data for CNN ============")
            self.preprocess(self.raw_data_test, self.processed_input_output_pairs_test, self.processed_test_data_file, self.dev_ratio, self.processed_dev_data_file)
            
    def find_max_coordinates(self, data_paths, raw_data_buffer):
        if self.augmentation:
            print('--> Data Augmentation: Rotation (by ' + str(self.rot_deg_increment) + ' deg incrementally up to 360 deg)')
        for path in data_paths:
            # Load data from txt file.
            txtfile = open(path, 'r')
            lines = txtfile.read().splitlines()
            data = [line.split() for line in lines]
            data = np.transpose(sorted(data, key=lambda line: int(line[0]))).astype(float)
            raw_data_buffer.append(data)            
            if self.augmentation:
                # Rotate data by deg_increment deg sequentially for data augmentation (only rotation is considered here)
                deg_increment_int = int(self.rot_deg_increment)
                for deg in range(deg_increment_int, 360, deg_increment_int):
                    data_rotated = np.zeros_like(data)
                    rad = np.radians(deg)
                    c, s = np.cos(rad), np.sin(rad)
                    Rot = np.array(((c,-s), (s, c)))
                    for ii in range(data.shape[1]):
                        data_rotated[0:2, ii] = data[0:2, ii]
                        data_rotated[2:, ii] = np.dot(Rot, data[2:, ii])
                    raw_data_buffer.append(data_rotated)
        # Find x_max, x_min, y_max, y_min across all the data in data_paths.
        x_max_global, x_min_global, y_max_global, y_min_global = -1000, 1000, -1000, 1000
        for data in raw_data_buffer:
            x = data[2,:]
            x_min, x_max = min(x), max(x)
            if x_min < x_min_global:
                x_min_global = x_min
            if x_max > x_max_global:
                x_max_global = x_max
            y = data[3,:]
            y_min, y_max = min(y), max(y)
            if y_min < y_min_global:
                y_min_global = y_min
            if y_max > y_max_global:
                y_max_global = y_max
        return x_max_global, x_min_global, y_max_global, y_min_global
        
    def normalize(self, raw_data_buffer, x_max_global, x_min_global, y_max_global, y_min_global):
        # Normalize all the data in this buffer to range from -1 to 1.
        for data in raw_data_buffer:
            x = data[2,:]
            x = (1 + 1)*(x - x_min_global)/(x_max_global - x_min_global)
            x = x - 1.0
            for jj in range(len(x)):
                if abs(x[jj]) < 0.0001:
                    data[2,jj] = 0.0
                else:
                    data[2,jj] = x[jj] 
            y = data[3,:]
            y = (1 + 1)*(y - y_min_global)/(y_max_global - y_min_global)
            y = y - 1.0
            for jj in range(len(y)):
                if abs(y[jj]) < 0.0001:
                    data[3,jj] = 0.0
                else:
                    data[3,jj] = y[jj]
        '''# Sanity check.
        # Find x_max, x_min, y_max, y_min in this raw_data_buffer
        x_max_buffer, x_min_buffer, y_max_buffer, y_min_buffer = -1000, 1000, -1000, 1000
        for data in raw_data_buffer:
            x = data[2,:]
            x_min, x_max = min(x), max(x)
            if x_min < x_min_buffer:
                x_min_buffer = x_min
            if x_max > x_max_buffer:
                x_max_buffer = x_max
            y = data[3,:]
            y_min, y_max = min(y), max(y)
            if y_min < y_min_buffer:
                y_min_buffer = y_min
            if y_max > y_max_buffer:
                y_max_buffer = y_max
        print(x_min_buffer, x_max_buffer)
        print(y_min_buffer, y_max_buffer)
        '''
    def preprocess(self, raw_data_buffer, processed_input_output_pairs, processed_data_file, dev_ratio=0., processed_data_file_2=None):
        random.seed(1) # Random seed for pedestrian permutation and data shuffling
        for data in raw_data_buffer:
            # Frame IDs of the frames in the current dataset
            frameList = np.unique(data[0, :].astype(int)).tolist()
            #print(frameList)
            numFrames = len(frameList)
            
            # Frame ID increment for this dataset.
            frame_increment = np.min(np.array(frameList[1:-1]) - np.array(frameList[0:-2]))
            
            # For this dataset check which pedestrians exist in each frame.
            pedsInFrameList = []
            pedsPosInFrameList = []
            for ind, frame in enumerate(frameList):
                # For this frame check the pedestrian IDs.
                pedsInFrame = data[:, data[0, :].astype(int) == frame]
                pedsList = pedsInFrame[1, :].astype(int).tolist()
                pedsInFrameList.append(pedsList)
                # Position information for each pedestrian.
                pedsPos = []
                for ped in pedsList:
                    # Extract x and y positions
                    current_x = pedsInFrame[2, pedsInFrame[1, :].astype(int) == ped][0]
                    current_y = pedsInFrame[3, pedsInFrame[1, :].astype(int) == ped][0]
                    pedsPos.extend([current_x, current_y])
                    if (current_x == 0.0 and current_y == 0.0):
                        print('[WARNING] There exists a pedestrian at coordinate [0.0, 0.0]')
                pedsPosInFrameList.append(pedsPos)
            # Go over the frames in this data again to extract data.
            ind = 0
            while ind < len(frameList) - (self.input_seq_length + self.pred_seq_length):
                # Check if this sequence contains consecutive frames. Otherwise skip this sequence.
                if not frameList[ind + self.input_seq_length + self.pred_seq_length - 1] - frameList[ind] == (self.input_seq_length + self.pred_seq_length - 1)*frame_increment:
                    ind += 1
                    continue
                # List of pedestirans in this sequence.
                pedsList = np.unique(np.concatenate(pedsInFrameList[ind : ind + self.input_seq_length + self.pred_seq_length])).tolist()
                # Print the Frame numbers and pedestrian IDs in this sequence for sanity check.
                # print(str(int(self.input_seq_length + self.pred_seq_length)) + ' frames starting from Frame ' + str(int(frameList[ind])) +  ' contain pedestrians ' + str(pedsList))
                # Initialize numpy arrays for input-output pair
                data_input = np.zeros((2*len(pedsList), self.input_seq_length))
                data_output = np.zeros((2*len(pedsList), self.pred_seq_length))
                for ii in range(self.input_seq_length):
                    for jj in range(len(pedsList)):
                        if pedsList[jj] in pedsInFrameList[ind + ii]:
                            datum_index = pedsInFrameList[ind + ii].index(pedsList[jj])
                            data_input[2*jj:2*(jj + 1), ii] = np.array(pedsPosInFrameList[ind + ii][2*datum_index:2*(datum_index + 1)])
                for ii in range(self.pred_seq_length):
                    for jj in range(len(pedsList)):
                        if pedsList[jj] in pedsInFrameList[ind + self.input_seq_length + ii]:
                            datum_index = pedsInFrameList[ind + self.input_seq_length + ii].index(pedsList[jj])
                            data_output[2*jj:2*(jj + 1), ii] = np.array(pedsPosInFrameList[ind + self.input_seq_length + ii][2*datum_index:2*(datum_index + 1)])
                processed_pair = (torch.from_numpy(data_input), torch.from_numpy(data_output))
                processed_input_output_pairs.append(processed_pair)
                ind += self.input_seq_length + self.pred_seq_length
        print('--> Data Size: ' + str(len(processed_input_output_pairs)))
        if self.augmentation:
            # Perform data augmentation
            self.augment_flip(processed_input_output_pairs)
            self.augment_permute(processed_input_output_pairs)
        else:
            print('--> Skipping data augmentation')
        # Shuffle data.
        print('--> Shuffling all data before saving')
        random.shuffle(processed_input_output_pairs)
        if dev_ratio != 0.:
            # Split data into dev and test sets.
            dev_size = int(len(processed_input_output_pairs)*dev_ratio)
            processed_dev_set = processed_input_output_pairs[:dev_size]
            processed_test_set = processed_input_output_pairs[dev_size:]
            print('--> Dumping dev data with size ' + str(len(processed_dev_set)) + ' to pickle file')
            f_dev = open(processed_data_file_2, 'wb')
            pickle.dump(processed_dev_set, f_dev, protocol=2)
            f_dev.close()
            print('--> Dumping test data with size ' + str(len(processed_test_set)) + ' to pickle file')
            f_test = open(processed_data_file, 'wb')
            pickle.dump(processed_test_set, f_test, protocol=2)
            f_test.close()
            # Clear buffer
            raw_data_buffer = []
            processed_input_output_pairs = []
        else:
            assert(processed_data_file_2 == None)
            processed_train_set = processed_input_output_pairs
            print('--> Dumping train data with size ' + str(len(processed_train_set)) + ' to pickle file')
            f_train = open(processed_data_file, 'wb')
            pickle.dump(processed_train_set, f_train, protocol=2)
            f_train.close()
            # Clear buffer
            raw_data_buffer = []
            processed_input_output_pairs = []
    
    def augment_flip(self, processed_input_output_pairs):
        print('--> Data Augmentation: Y Flip')
        augmented_input_output_pairs = []
        for processed_input_output_pair in tqdm(processed_input_output_pairs):
            data_input, data_output = processed_input_output_pair[0].numpy(), processed_input_output_pair[1].numpy()
            num_peds = int(data_input.shape[0]/2)
            # Flip y
            data_input_yflipped = np.zeros_like(data_input)
            data_output_yflipped = np.zeros_like(data_output)
            for kk in range(num_peds):
                data_input_yflipped[2*kk, :] = data_input[2*kk, :]
                data_input_yflipped[2*kk+1, :] = -1*data_input[2*kk+1, :]
                data_output_yflipped[2*kk, :] = data_output[2*kk, :]
                data_output_yflipped[2*kk+1, :] = -1*data_output[2*kk+1, :]
            processed_pair_yflipped = (torch.from_numpy(data_input_yflipped), torch.from_numpy(data_output_yflipped))
            augmented_input_output_pairs.append(processed_pair_yflipped)
        processed_input_output_pairs.extend(augmented_input_output_pairs)
        print('--> Augmented Data Size: ' + str(len(processed_input_output_pairs)))
        
    def augment_permute(self, processed_input_output_pairs):
        # Specify how many pedestrian permutations to consider per input-output pair
        print('--> Data Augmentation: Pedestrian Permutation (' + str(self.permutations) + ' random permutations per input-output pair)')
        augmented_input_output_pairs = []
        for processed_input_output_pair in tqdm(processed_input_output_pairs):
            data_input, data_output = processed_input_output_pair[0].numpy(), processed_input_output_pair[1].numpy()
            num_peds = int(data_input.shape[0]/2)
            for ii in range(self.permutations):
                perm = np.random.permutation(num_peds)
                data_input_permuted = np.zeros_like(data_input)
                data_output_permuted = np.zeros_like(data_output)
                for jj in range(len(perm)):
                    data_input_permuted[2*jj:2*(jj+1), :] = data_input[2*perm[jj]:2*(perm[jj]+1), :]
                    data_output_permuted[2*jj:2*(jj+1), :] = data_output[2*perm[jj]:2*(perm[jj]+1), :]
                processed_pair_permuted = (torch.from_numpy(data_input_permuted), torch.from_numpy(data_output_permuted))
                augmented_input_output_pairs.append(processed_pair_permuted)
        processed_input_output_pairs.extend(augmented_input_output_pairs)
        print('--> Augmented Data Size: ' + str(len(processed_input_output_pairs)))

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
