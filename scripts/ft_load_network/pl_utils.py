import os, torch
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import Callback
from termcolor import colored

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
