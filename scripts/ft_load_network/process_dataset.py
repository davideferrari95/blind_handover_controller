import os, pandas as pd
from typing import List, Tuple
from termcolor import colored

import torch
from torch.utils.data import Dataset, DataLoader, random_split

# Get Data Path
from pathlib import Path
PACKAGE_PATH = f'{str(Path(__file__).resolve().parents[2])}'
DATA_PATH = f'{PACKAGE_PATH}/data'
# DATA_PATH = f'{PACKAGE_PATH}/data_test'

"""
proviamo diversamente:

creo il dataset e lo processo così:

    1. per ogni cartella in data -> faccio un dataframe con i dati di joint_states_data.csv e ft_sensor_data.csv
    2. aggiungo una colonna experiment_id che è l'indice della cartella, e una colonna open_gripper che è 1 se siamo negli ultimi 100 samples, 0 altrimenti
    3. aggiungo il dataframe alla lista dataframe_list

poi creo il dataset e i dataloader così:

    1. creo un dataset con la lista di dataframe
    2. ogni dataframe nella lista verrà salvato come un item del dataset
    3. quando chiedo il getitem del dataset, mi ritorna uno slice di 100 samples del dataframe idx, e l'etichetta è l'open_gripper dell'ultimo sample

alternativa:

    1. per ogni dataframe creo un numero M di sequenze di N samples, con stride S:

        Ad esempio, se hai un DataFrame di lunghezza 1000 e vuoi creare sequenze di 50 campioni con uno stride di 10, il processo potrebbe apparire così:

        Sequenza 1: campioni 1-50
        Sequenza 2: campioni 11-60
        Sequenza 3: campioni 21-70
        E così via...

    2. ogni sequenza diventa un item del dataset, e l'etichetta è l'open_gripper dell'ultimo sample
    3. quando chiedo il getitem del dataset, mi ritorna una di queste sequenze con il label correlato

creo il modello e lo addestro così:

    1. creo un modello con LSTM, FC Layer, Sigmoid + BCELoss e AdamW come ottimizzatore
    input_size = dataframe.shape[1] - 2, hidden_size = [64], output_size = 1, num_layers = sequence_length, learning_rate = 0.001
    2. addestro il modello con il train_dataloader e il val_dataloader
    3. testo il modello con il test_dataloader

"""

class CustomDataset(Dataset):

    """ Create Custom Dataset for LSTM Network """

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

    def __init__(self, dataframe_list:List[pd.DataFrame], sequence_length:int=100, stride:int=10):

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

                # Append Sequence and Label to Lists
                self.sequences.append(sequence)
                self.labels.append(label)

                # Debug
                print(f'DataFrame {num+1} | Sequence {len(self.sequences)} | Label {label}')

    def __len__(self):

        return len(self.sequences)

    def __getitem__(self, idx):

        # Get Sequence and Label
        sequence = self.sequences[idx]
        label = self.labels[idx]

        # Convert to Torch Tensor
        sequence = torch.tensor(sequence, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return sequence, label

class ProcessDataset():

    """ Process Dataset for LSTM Network """

    def __init__(self, batch_size:int=32, sequence_length:int=100, stride:int=10, open_gripper_len:int=100, shuffle:bool=True):

        # Read CSV Files
        dataframe_list = self.read_csv_files()

        # Add Experiment ID and Boolean Parameter (Open Gripper)
        dataframe_list = self.complete_dataset(dataframe_list, open_gripper_len=open_gripper_len)

        # Dataset Creation
        dataset = CustomDataset(dataframe_list, sequence_length, stride)
        self.sequence_shape = dataset[0][0].shape

        # DataLoader Creation
        self.split_dataloader(dataset, batch_size, train_size=0.8, test_size=0.15, validation_size=0.05, shuffle=shuffle)

        # Print
        print(colored('\nDataLoader Created\n', 'green'))

    def get_datasets(self) -> Tuple[CustomDataset, CustomDataset, CustomDataset]:

        """ Get Train, Test and Validation Datasets """

        return self.train_dataset, self.test_dataset, self.val_dataset

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:

        """ Get Train, Test and Validation DataLoaders """

        return self.train_dataloader, self.test_dataloader, self.val_dataloader

    def read_csv_files(self) -> List[pd.DataFrame]:

        """ Read CSV Files """

        # DataFrames List
        dataframe_list = []

        for folder in os.listdir(DATA_PATH):

            # Read CSV Files
            velocity_df, ft_sensor_df = pd.read_csv(f'{DATA_PATH}/{folder}/joint_states_data.csv'), pd.read_csv(f'{DATA_PATH}/{folder}/ft_sensor_data.csv')

            # Assert DataFrames Length Match
            assert len(velocity_df) == len(ft_sensor_df), f'DataFrames Length Mismatch | {folder} | {len(velocity_df)} != {len(ft_sensor_df)}'

            # Merge DataFrames
            # df = pd.concat([velocity_df, ft_sensor_df], axis=1)

            # Append DataFrame to List
            # dataframe_list.append(df)
            dataframe_list.append(ft_sensor_df)

        return dataframe_list

    def complete_dataset(self, dataframe_list: List[pd.DataFrame], open_gripper_len:int=100) -> List[pd.DataFrame]:

        """ Complete Dataset - Add Boolean Parameter (Open Gripper) """

        for _, df in enumerate(dataframe_list):

            # Add Boolean Parameter (Open Gripper - 1 if last 100 samples, 0 otherwise)
            df['open_gripper'] = 0
            df.iloc[-open_gripper_len:, df.columns.get_loc('open_gripper')] = 1

        return dataframe_list

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

if __name__ == '__main__':

    # ProcessDataset(32, 100, 10, 100, True)
    ProcessDataset(32, 10, 2, 2, True)
