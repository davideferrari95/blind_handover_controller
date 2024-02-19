import os, dill
import numpy as np
import pandas as pd
from typing import List, Tuple
from termcolor import colored

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pl_utils import get_dataset_name

# Import Sklearn Resample
from sklearn.utils import resample

# Get Data Path
from pathlib import Path
PACKAGE_PATH = f'{str(Path(__file__).resolve().parents[2])}'

# Data Path - Hyperparameters - Balance Strategy
DATA_PATH = f'{PACKAGE_PATH}/data'
BATCH_SIZE, PATIENCE, LOAD_VELOCITIES, DISTURBANCES = 256, 50, False, True
HIDDEN_SIZE, SEQUENCE_LENGTH, STRIDE, OPEN_GRIPPER_LEN = [512, 256], 1000, 10, 100
BALANCE_STRATEGY = ['weighted_loss', 'oversampling', 'undersampling']

# Model Type (CNN, LSTM, Feedforward, MultiClassifier, BinaryClassifier)
# MODEL_TYPE = 'MultiClassifier'
MODEL_TYPE = 'LSTM'

class CustomDataset(Dataset):

    """ Create Custom Dataset Neural Network """

    """
        Input: List of DataFrames, Sequence Length, Stride
        Output: Sequences and Labels for LSTM Network

        Eg. DataFrame length = 1000, sequence_length = 50, stride = 10

            Sequence 1: samples 1-50
            Sequence 2: samples 11-60
            Sequence 3: samples 21-70
            And so on...

        Repeat for all DataFrames in the List
    """

    def __init__(self, dataframe_list:List[pd.DataFrame], sequence_length:int=100, stride:int=10, balance_strategy:List[str]=['weighted_loss'], disturbances:bool=False):

        """ Initialize Custom Dataset """

        # Compute Dataset, Model and Config Names
        dataset_name = get_dataset_name(sequence_length, stride, balance_strategy, disturbances)

        # Load Dataset if Exists
        if os.path.exists(f'{PACKAGE_PATH}/dataset/{dataset_name}.pkl'):

            # Override Self with Loaded Dataset
            print(colored(f'\nLoading Dataset...', 'green'))
            self.sequences, self.labels, self.class_weight = self.load_dataset(dataset_name)

            return

        assert len(dataframe_list) > 0, 'Empty DataFrame List'
        assert sequence_length > 0, 'Invalid Sequence Length'
        assert stride > 0, 'Invalid Stride'

        # Initialize Sequences and Labels
        self.sequences, self.labels = [], []

        # Iterating over the List of DataFrames
        for num, df in enumerate(dataframe_list):

            # Iterating over the DataFrame to Create Sequences and Labels
            for i in range(0, len(df) - sequence_length + 1, stride):

                # Get Sequence and Label (last sample's 'open_gripper' value)
                sequence = df.iloc[i:i+sequence_length, :-1].values
                label = df.iloc[i+sequence_length-1, -1]

                # Convert to Torch Tensor
                sequence = torch.tensor(sequence, dtype=torch.float32)
                label = torch.tensor(label)
                label = torch.nn.functional.one_hot(label, num_classes=2)

                # Append Sequence and Label to Lists
                self.sequences.append(sequence)
                self.labels.append(label)

                # Debug
                print(f'DataFrame {num+1} | Sequence {len(self.sequences)} | Label {label}')

        # Initialize Class Weights
        self.class_weight = torch.tensor([1.0, 1.0])

        # Apply Balance Strategy
        for strategy in balance_strategy: assert strategy in ['weighted_loss', 'oversampling', 'undersampling', 'smote'], f'Invalid Balance Strategy: {strategy}'
        if balance_strategy is not None: print(colored('\nApplying Balance Strategy\n', 'green'))
        if 'undersampling' in balance_strategy: self.apply_undersampling()
        if 'oversampling'  in balance_strategy: self.apply_oversampling()
        if 'weighted_loss' in balance_strategy: self.apply_weighted_loss()

        print(colored(f'\nDataset Created: \n', 'green'))
        print(f'    Sequences: {len(self.sequences)} | Labels: {len(self.labels)}')
        print(f'    Sequences - Classes 0: {np.count_nonzero([1 if all(t == torch.Tensor([1,0])) else 0 for t in self.labels])}')
        print(f'    Sequences - Classes 1: {np.count_nonzero([1 if all(t == torch.Tensor([0,1])) else 0 for t in self.labels])}')
        print(f'    Class Weights: {self.class_weight}')

        # Save Dataset
        print(colored(f'\nSaving Dataset...', 'green'))
        self.save_dataset(dataset_name)

    def save_dataset(self, name:str='dataset'):

        """ Save Dataset to File - Dill """

        # Create Directory if it Doesn't Exist
        os.makedirs(f'{PACKAGE_PATH}/dataset', exist_ok=True)

        # Save Dataset to File
        with open(f'{PACKAGE_PATH}/dataset/{name}.pkl', 'wb') as file:
            dill.dump((self.sequences, self.labels, self.class_weight), file)
            print('Saved')

    def load_dataset(self, name:str='dataset') -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:

        """ Load Dataset from File - Dill """

        # Load Dataset from File
        with open(f'{PACKAGE_PATH}/dataset/{name}.pkl', 'rb') as file:
            sequences, labels, class_weight = dill.load(file)
            print('Loaded')

        print(colored(f'\nDataset Loaded: \n', 'green'))
        print(f'    Sequences: {len(sequences)} | Labels: {len(labels)}')
        print(f'    Sequences - Classes 0: {np.count_nonzero([1 if all(t == torch.Tensor([1,0])) else 0 for t in labels])}')
        print(f'    Sequences - Classes 1: {np.count_nonzero([1 if all(t == torch.Tensor([0,1])) else 0 for t in labels])}')
        print(f'    Class Weights: {class_weight}')

        # Return Sequences, Labels and Class Weights
        return sequences, labels, class_weight

    def __len__(self):

        return len(self.sequences)

    def __getitem__(self, idx):

        # Get Sequence and Label
        sequence = self.sequences[idx]
        label = self.labels[idx]

        return sequence, label

    def apply_weighted_loss(self):

        """ Apply Weighted Loss to the Dataset """

        print(colored('    Applying Weighted Loss', 'yellow'))

        # Get Sequences and Labels as Numpy Arrays
        labels_concatenated = [1 if all(t == torch.Tensor([0,1])) else 0 for t in self.labels]

        # Compute Class Weights
        class_counts = torch.bincount(torch.tensor(labels_concatenated))
        total_samples = float(sum(class_counts))
        class_weights = total_samples / (2.0 * class_counts.float())

        # Compute Class Weights Tensor
        self.class_weight = torch.tensor([class_weights[0], class_weights[1]])

    def apply_oversampling(self):

        """ Apply Oversampling to the Dataset """

        print(colored('    Applying Oversampling', 'yellow'))

        # Get Sequences and Labels as Numpy Arrays
        labels_concatenated = [1 if all(t == torch.Tensor([0,1])) else 0 for t in self.labels]

        # Get Indices of Majority and Minority Classes
        minority_indices = torch.where(torch.tensor(labels_concatenated) == 1)[0]
        majority_indices = torch.where(torch.tensor(labels_concatenated) == 0)[0]

        # Resample the Minority Class
        oversampled_indices = resample(minority_indices.numpy(), n_samples=len(minority_indices)*(len(majority_indices)//len(minority_indices)//2))

        # Update Sequences and Labels
        self.sequences += [self.sequences[i] for i in oversampled_indices]
        self.labels += [self.labels[i] for i in oversampled_indices]

    def apply_undersampling(self):

        """ Apply Undersampling to the Dataset """

        print(colored('    Applying Undersampling', 'yellow'))

        # Get Sequences and Labels as Numpy Arrays
        labels_concatenated = [1 if all(t == torch.Tensor([0,1])) else 0 for t in self.labels]

        # Get Indices of Majority and Minority Classes
        minority_indices = torch.where(torch.tensor(labels_concatenated) == 1)[0]
        majority_indices = torch.where(torch.tensor(labels_concatenated) == 0)[0]

        # Resample the Majority Class
        undersampled_indices = resample(majority_indices.numpy(), n_samples=len(majority_indices)//(len(majority_indices)//len(minority_indices)//2))

        # Remove Duplicates
        undersampled_indices = list(dict.fromkeys(undersampled_indices))

        # Update Sequences and Labels
        for index in sorted(undersampled_indices, reverse=True):
            self.sequences.pop(index)
            self.labels.pop(index)

class ProcessDataset():

    """ Process Dataset for LSTM Network """

    """ Balance Strategies:

        1. Weighted Loss -> Apply Class Weights to the Loss Function
        2. Oversampling  -> Resample the Minority Class
        3. Undersampling -> Resample the Majority Class

    """

    def __init__(self, batch_size:int=32, sequence_length:int=100, stride:int=10, open_gripper_len:int=100, shuffle:bool=True, balance_strategy:List[str]=['weighted_loss'], disturbances:bool=False):

        # Read CSV Files
        dataframe_list = self.read_csv_files(DATA_PATH)

        # Add Experiment ID and Boolean Parameter (Open Gripper)
        dataframe_list = self.complete_dataset(dataframe_list, open_gripper_len=open_gripper_len)

        # Add Disturbances Data
        if disturbances: dataframe_list += self.process_disturbances_data()

        # Dataset Creation
        self.dataset = CustomDataset(dataframe_list, sequence_length, stride, balance_strategy, disturbances)
        self.sequence_shape = self.dataset[0][0].shape

        # DataLoader Creation
        self.split_dataloader(self.dataset, batch_size, train_size=0.8, test_size=0.15, validation_size=0.05, shuffle=shuffle)

        # Print
        print(colored('\nDataLoader Created\n', 'green'))

    def get_datasets(self) -> Tuple[CustomDataset, CustomDataset, CustomDataset]:

        """ Get Train, Test and Validation Datasets """

        return self.train_dataset, self.test_dataset, self.val_dataset

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:

        """ Get Train, Test and Validation DataLoaders """

        return self.train_dataloader, self.test_dataloader, self.val_dataloader

    def get_class_weights(self) -> torch.Tensor:

        """ Get Class Weights """

        return self.dataset.class_weight

    def read_csv_files(self, path:str=DATA_PATH) -> List[pd.DataFrame]:

        """ Read CSV Files """

        # DataFrames List
        dataframe_list = []

        for folder in os.listdir(path):

            # Skip if not a Directory
            if not os.path.isdir(f'{path}/{folder}'): print(f'Skipping {folder}'); continue

            # Skip if not `Test - ...` Folder
            if not folder.startswith('Test'): print(f'Skipping {folder}'); continue

            # Read CSV Files
            velocity_df, ft_sensor_df = pd.read_csv(f'{path}/{folder}/joint_states_data.csv'), pd.read_csv(f'{path}/{folder}/ft_sensor_data.csv')

            # Assert DataFrames Length Match
            assert len(velocity_df) == len(ft_sensor_df), f'DataFrames Length Mismatch | {folder} | {len(velocity_df)} != {len(ft_sensor_df)}'

            # Merge DataFrames if Load Velocities
            if LOAD_VELOCITIES: df = pd.concat([velocity_df, ft_sensor_df], axis=1)
            else: df = ft_sensor_df

            # Append DataFrame to List
            dataframe_list.append(df)

        return dataframe_list

    def complete_dataset(self, dataframe_list: List[pd.DataFrame], open_gripper_len:int=100) -> List[pd.DataFrame]:

        """ Complete Dataset - Add Boolean Parameter (Open Gripper) """

        for _, df in enumerate(dataframe_list):

            # Add Boolean Parameter (Open Gripper - 1 if last 100 samples, 0 otherwise)
            df['open_gripper'] = 0
            df.iloc[-open_gripper_len:, df.columns.get_loc('open_gripper')] = 1

        return dataframe_list

    def process_disturbances_data(self) -> List[pd.DataFrame]:

        """ Process Disturbances Data """

        # Read Disturbances CSV Files
        disturbances_dataframe_list = self.read_csv_files(f'{DATA_PATH}/Disturbances')

        # Add Boolean Parameter (Open Gripper = 0)
        for df in disturbances_dataframe_list: df['open_gripper'] = 0

        return disturbances_dataframe_list

    def split_dataloader(self, dataset:Dataset, batch_size:int=64, train_size:float=0.8, test_size:float=0.15, validation_size:float=0.05, shuffle:bool=True):

        """ Split DataLoader into Train, Test and Validation """

        # Split Dataset
        assert train_size + test_size + validation_size == 1, 'Train, Test and Validation Sizes do not add up to 1'
        self.train_dataset, self.test_dataset, self.val_dataset = random_split(dataset, [train_size, test_size, validation_size])
        assert len(self.train_dataset) + len(self.test_dataset) + len(self.val_dataset) == len(dataset), 'Train, Test and Validation Sizes do not add up to Dataset Length'

        # Create DataLoaders
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=os.cpu_count())
        self.test_dataloader  = DataLoader(self.test_dataset,  batch_size=batch_size, shuffle=False,   num_workers=os.cpu_count())
        self.val_dataloader   = DataLoader(self.val_dataset,   batch_size=batch_size, shuffle=False,   num_workers=os.cpu_count())
