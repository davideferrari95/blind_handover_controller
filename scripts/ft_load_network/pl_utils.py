import os, torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback

from configparser import ConfigParser
from termcolor import colored
from typing import List, Tuple, Optional

# Get Torch Device ('cuda' or 'cpu')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def save_model(path:str, file_name:str, model:LightningModule):

    """ Save Model Function """

    # Create Directory if it Doesn't Exist
    os.makedirs(path, exist_ok=True)

    with open(os.path.join(path, f'{file_name}.pth'), 'wb') as FILE:

        # Save Model Weights
        torch.save(model.state_dict(), FILE)

    print(colored('\n\nModel Saved Correctly\n\n', 'green'))

def load_model(path:str, file_name:str, model:LightningModule):

    """ Load Model Function """

    with open(os.path.join(path, f'{file_name}.pth'), 'rb') as FILE:

        # Load Model Weights
        model.load_state_dict(torch.load(FILE, map_location=DEVICE))

    print(colored('\n\nModel Loaded Correctly\n\n', 'green'))

def save_hyperparameters(save_path:str, config_name:str, model_name:str, model_type:str, input_size:int, hidden_size:List[int], output_size:int, sequence_length:int, num_layers:Optional[int]=None, learning_rate:float=0.001):

    """ Save Hyperparameters in a Config File """

    # Create a Config Parser
    config = ConfigParser()

    # Add Sections and Values
    config['Model'] = {
        'model_type': model_type,
        'input_size': str(input_size),
        'hidden_size': ', '.join(map(str, hidden_size)),
        'output_size': str(output_size),
        'sequence_length': str(sequence_length),
        'learning_rate': str(learning_rate),
        'num_layers': str(num_layers)
    }

    config['Names'] = {
        'model_name': model_name,
        'config_name': config_name
    }

    # Create Directory if it Doesn't Exist
    os.makedirs(save_path, exist_ok=True)

    # Write Config File
    with open(os.path.join(save_path, f'{config_name}.ini'), 'w') as FILE:
        config.write(FILE)

def load_hyperparameters(path:str, name:str) -> Tuple[str, str, str, int, List[int], int, int, float]:

    """ Load Hyperparameters from a Config File:

        model_name, config_name, model_type, input_size, hidden_size, output_size, sequence_length, num_layers, learning_rate 

    """

    # Create and Read a Config Parser
    config = ConfigParser()
    config.read(os.path.join(path, f'{name}.ini'))

    # Read Values from Section 'Model'
    model_section, names = config['Model'], config['Names']

    # Model Values
    model_type = model_section['model_type']
    input_size = int(model_section['input_size'])
    hidden_size = [int(size) for size in model_section['hidden_size'].split(', ')]
    output_size = int(model_section['output_size'])
    sequence_length = int(model_section['sequence_length'])
    learning_rate = float(model_section['learning_rate'])

    # Optional Model Values
    num_layers = int(model_section['num_layers']) if model_section['num_layers'] is not None else None

    # Names of the Model and Config
    model_name, config_name = names['model_name'], names['config_name']

    return model_name, config_name, model_type, input_size, hidden_size, output_size, sequence_length, num_layers, learning_rate

def get_dataset_name(sequence_length:int, stride:int, balance_strategy:List[str]) -> str:

    """ Compute Dataset Name """

    # Compute Dataset Name
    return f'dataset_{sequence_length}_{stride}_{"".join([s[0] for s in balance_strategy])}'

def get_config_name(model_type:str, sequence_length:int, stride:int, balance_strategy:List[str]) -> str:

    """ Compute Config Name """

    # Compute Config Name
    return f'{model_type}_config_{sequence_length}_{stride}_{"".join([s[0] for s in balance_strategy])}'

def get_model_name(model_type:str, sequence_length:int, stride:int, balance_strategy:List[str]) -> str:

    """ Compute Model Name """

    # Compute Model Name
    return f'{model_type}_model_{sequence_length}_{stride}_{"".join([s[0] for s in balance_strategy])}'

class StartTrainingCallback(Callback):

    """ Print Start Training Info Callback """

    # On Start Training
    def on_train_start(self, trainer, pl_module): print(colored('\n\nStart Training Process\n\n','yellow'))

    # On End Training
    def on_train_end(self, trainer, pl_module): print(colored('\n\nTraining Done\n\n','yellow'))

class StartTestingCallback(Callback):

    """ Print Start Testing Info Callback """

    # On Start Testing
    def on_test_start(self, trainer, pl_module): print(colored('\n\nStart Testing Process\n\n','yellow'))

    # On End Testing
    def on_test_end(self, trainer, pl_module): print(colored('\n\n\nTesting Done\n\n','yellow'))

class StartValidationCallback(Callback):

    """ Print Start Testing Info Callback """

    # On Start Testing
    def on_validation_start(self, trainer, pl_module): print(colored('\n\nStart Validation Process\n\n','yellow'))

    # On End Testing
    def on_validation_end(self, trainer, pl_module): print(colored('\n\n\nValidation Done\n\n','yellow'))
