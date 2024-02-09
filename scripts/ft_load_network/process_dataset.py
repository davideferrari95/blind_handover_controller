import os, pandas as pd
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader

# Get Data Path
from pathlib import Path
PACKAGE_PATH = f'{str(Path(__file__).resolve().parents[2])}'
DATA_PATH = f'{PACKAGE_PATH}/data/test'

class CustomDataset(Dataset):

    """ Custom Dataset """

    def __init__(self, dataframe:pd.DataFrame, sequence_length:int):

        self.data = dataframe.values
        self.sequence_length = sequence_length

    def __len__(self):

        return len(self.data) - self.sequence_length + 1

    def __getitem__(self, idx):

        # Calculate Start and End Index
        end_idx = idx + self.sequence_length

        # Get Sequence Data (Exclude 'experiment_id' and 'open_gripper') and Label ('open_gripper' as label)
        seq_data = torch.tensor(self.data[idx:end_idx, :-2], dtype=torch.float32)
        label = torch.tensor(self.data[end_idx - 1, -1], dtype=torch.float32)

        return seq_data, label
class ProcessDataset():

    """ Process Dataset for LSTM Network """

    def __init__(self, batch_size:int=32, sequence_length:int=100, shuffle:bool=True):

        # Read CSV Files
        dataframe_list = self.read_csv_files()

        # Add Experiment ID and Boolean Parameter (Open Gripper)
        dataframe_list = self.complete_dataset(dataframe_list)

        # Merge DataFrames
        df = pd.concat(dataframe_list, ignore_index=True)

        # Dataset Creation
        self.dataset = CustomDataset(df, sequence_length)

        # DataLoader Creation
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)

        # Sample Data
        # print(dataframe_list[0])
        # print(df)
        # sample_sequence, sample_label = self.dataset[0]
        # print("Length of dataset:", len(self.dataset))
        # print("Sample Sequence:", sample_sequence)
        # print("Sample Label:", sample_label)

    def get_dataset(self) -> CustomDataset:

        """ Get Dataset """

        return self.dataset

    def get_dataloader(self) -> DataLoader:

        """ Get DataLoader """

        return self.dataloader

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
            df.iloc[-100:, df.columns.get_loc('open_gripper')] = 1

        return dataframe_list

if __name__ == '__main__':

    ProcessDataset(32, 100, True)
