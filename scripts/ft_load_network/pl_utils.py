import os, torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback

from configparser import ConfigParser
from termcolor import colored
from typing import List, Tuple

# Get Torch Device ('cuda' or 'cpu')
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def save_model(path:str, file_name:str, model:LightningModule):

    """ Save File Function """

    # Create Directory if it Doesn't Exist
    os.makedirs(path, exist_ok=True)

    # Need input of the model
    input_shape = model.input_size
    output_shape = model.output_size

    checkpoint = {
        'input_shape': input_shape,
        'output_shape': output_shape,
        'model_state_dict': model.state_dict(),
    }

    with open(os.path.join(path, file_name), 'wb') as FILE:
        torch.save(checkpoint, FILE)
    print(colored('\n\nModel Saved Correctly\n\n', 'green'))

def save_hyperparameters(save_path:str, input_size:int, hidden_size:List[int], output_size:int, num_layers:int, learning_rate:float=0.001):

    """ Save Hyperparameters in a Config File """

    # Create a Config Parser
    config = ConfigParser()

    # Add Sections and Values
    config['Model'] = {
        'input_size': str(input_size),
        'hidden_size': ', '.join(map(str, hidden_size)),
        'output_size': str(output_size),
        'num_layers': str(num_layers),
        'learning_rate': str(learning_rate)
    }

    # Create Directory if it Doesn't Exist
    os.makedirs(save_path, exist_ok=True)

    # Write Config File
    with open(os.path.join(save_path, 'config.ini'), 'w') as FILE:
        config.write(FILE)

def load_hyperparameters(path: str) -> Tuple[int, List[int], int, int, float]:

    """ Load Hyperparameters from a Config File """

    # Create and Read a Config Parser
    config = ConfigParser()
    config.read(os.path.join(path, 'config.ini'))

    # Read Values from Section 'Model'
    model_section = config['Model']

    # Convert Values
    input_size = int(model_section['input_size'])
    hidden_size = [int(size) for size in model_section['hidden_size'].split(', ')]
    output_size = int(model_section['output_size'])
    num_layers = int(model_section['num_layers'])
    learning_rate = float(model_section['learning_rate'])

    return input_size, hidden_size, output_size, num_layers, learning_rate

# Print Start Training Info Callback
class StartTrainingCallback(Callback):

    # On Start Training
    def on_train_start(self, trainer, pl_module): print(colored('\n\nStart Training Process\n\n','yellow'))

    # On End Training
    def on_train_end(self, trainer, pl_module): print(colored('\n\nTraining Done\n\n','yellow'))

# Print Start Testing Info Callback
class StartTestingCallback(Callback):

    # On Start Testing
    def on_test_start(self, trainer, pl_module): print(colored('\n\nStart Testing Process\n\n','yellow'))

    # On End Testing
    def on_test_end(self, trainer, pl_module): print(colored('\n\n\nTesting Done\n\n','yellow'))

# Print Start Testing Info Callback
class StartValidationCallback(Callback):

    # On Start Testing
    def on_validation_start(self, trainer, pl_module): print(colored('\n\nStart Validation Process\n\n','yellow'))

    # On End Testing
    def on_validation_end(self, trainer, pl_module): print(colored('\n\n\nValidation Done\n\n','yellow'))
