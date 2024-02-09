import os, pandas as pd
from typing import List

import torch, pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence

# Get Data Path
from pathlib import Path
PACKAGE_PATH = f'{str(Path(__file__).resolve().parents[2])}'
DATA_PATH = f'{PACKAGE_PATH}/data/test'

class CustomDataset(Dataset):

    def __init__(self, packed_sequences, labels):

        # Dataset Initialization
        self.packed_sequences = packed_sequences
        self.labels = labels

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, idx):

        return self.packed_sequences[idx], self.labels[idx]

class ProcessDataset():

    """ Process Dataset for LSTM Network """

    def __init__(self):

        # Read CSV Files
        dataframe_list = self.read_csv_files()

        # Add Experiment ID and Boolean Parameter (Open Gripper)
        dataframe_list = self.complete_dataset(dataframe_list)

        print(dataframe_list[0])
        print(dataframe_list[1])

        # Merge DataFrames
        df = pd.concat(dataframe_list, ignore_index=True)
        # print(df)

        # Pack Sequences
        sequences, labels = self.parse_sequences(df)
        
        print(sequences, labels)

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
            df = pd.concat([velocity_df, ft_sensor_df], axis=1)

            # Append DataFrame to List
            dataframe_list.append(df)

        return dataframe_list

    def complete_dataset(self, dataframe_list: List[pd.DataFrame]) -> List[pd.DataFrame]:

        """ Complete Dataset - Add Experiment ID and Boolean Parameter (Open Gripper) """

        for i, df in enumerate(dataframe_list):

            # Add Experiment ID
            df['experiment_id'] = i

            # Add Boolean Parameter (Open Gripper - 1 if last 100 samples, 0 otherwise)
            df['open_gripper'] = 0
            df.iloc[-5:, df.columns.get_loc('open_gripper')] = 1
            # df.iloc[-100:, df.columns.get_loc('open_gripper')] = 1

        return dataframe_list

    def parse_sequences(self, df: pd.DataFrame):

        """ Parse Sequences """

        # Create Dataset
        sequences, labels = [], []

        for group_id, group_df in df.groupby('experiment_id'):

            # Sequence (Velocity and Force/Torque Sensor Data)
            sequence = torch.tensor(group_df[df.columns.values[:-2]].values, dtype=torch.float32)
            # print(sequence)

            # Label
            label = torch.tensor(group_df['open_gripper'].values, dtype=torch.long)

            # Append to Lists
            sequences.append(sequence)
            labels.append(label)

        # Pack Sequences
        packed_sequences = pack_sequence(sequences, enforce_sorted=False)

        # Concatenate Labels
        labels = torch.cat(labels)

        return packed_sequences, labels

if __name__ == '__main__':

    ProcessDataset()
