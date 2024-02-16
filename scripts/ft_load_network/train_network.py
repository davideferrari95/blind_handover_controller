from termcolor import colored

# Import PyTorch Lightning Functions
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import Trainer, loggers as pl_loggers

# Import PyTorch Lightning NN Models
from networks import LSTMModel, CNNModel, FeedforwardModel, MultiClassifierModel, BinaryClassifierModel

# Import Processed Dataset and DataLoader
from process_dataset import ProcessDataset, PACKAGE_PATH, MODEL_TYPE, BATCH_SIZE, PATIENCE
from process_dataset import HIDDEN_SIZE, SEQUENCE_LENGTH, STRIDE, OPEN_GRIPPER_LEN, BALANCE_STRATEGY

# Import Callbacks and Utilities
from pl_utils import save_model, save_hyperparameters, DEVICE, get_model_name, get_config_name
from pl_utils import StartTestingCallback, StartTrainingCallback, StartValidationCallback

class TrainingNetwork():

    """ Train LSTM Network """

    def __init__(self, batch_size:int=32, model_type:str='MultiClassifier', sequence_length:int=100, stride:int=10, open_gripper_len:int=100, shuffle:bool=True):

        # Process Dataset
        process_dataset = ProcessDataset(batch_size, sequence_length, stride, open_gripper_len, shuffle, BALANCE_STRATEGY)
        class_weights = process_dataset.get_class_weights()

        # Model and Config Names
        self.model_name, config_name = get_model_name(model_type, sequence_length, stride, BALANCE_STRATEGY), get_config_name(model_type, sequence_length, stride, BALANCE_STRATEGY)

        # Get DataLoaders
        self.train_dataloader, self.test_dataloader, self.val_dataloader = process_dataset.get_dataloaders()

        # Model Hyperparameters (Input Size, Hidden Size, Output Size, Number of Layers)
        input_size, hidden_size, output_size, num_layers = process_dataset.sequence_shape[1], HIDDEN_SIZE, 2, 1
        learning_rate = 1e-3

        # Not Tested Model Types
        if model_type in ['CNN', 'Feedforward', 'BinaryClassifier']: print(colored(f'\n{model_type} Model Type Not Tested Yet!\n', 'red'))

        # Create NN Model
        if   model_type == 'CNN':              self.model = CNNModel(input_size, hidden_size, output_size, sequence_length, learning_rate).to(DEVICE)
        elif model_type == 'LSTM':             self.model = LSTMModel(input_size, hidden_size, output_size, num_layers, learning_rate).to(DEVICE)
        elif model_type == 'Feedforward':      self.model = FeedforwardModel(input_size * sequence_length, hidden_size, output_size, learning_rate).to(DEVICE)
        elif model_type == 'MultiClassifier':  self.model = MultiClassifierModel(input_size * sequence_length, hidden_size, output_size, learning_rate, class_weights).to(DEVICE)
        elif model_type == 'BinaryClassifier': self.model = BinaryClassifierModel(input_size * sequence_length, hidden_size, output_size, learning_rate).to(DEVICE)
        else: raise ValueError(f'Invalid Model Type: {model_type}')

        # Save Hyperparameters in Config File
        save_hyperparameters(f'{PACKAGE_PATH}/model', config_name, self.model_name, model_type, input_size, hidden_size, output_size, sequence_length, num_layers, learning_rate)

    def train_network(self):

        """ Train LSTM Network """

        # PyTorch Lightning Trainer
        trainer = Trainer(

            # Devices
            devices= 'auto',

            # Hyperparameters
            min_epochs = 50,
            max_epochs = 500,
            log_every_n_steps = 1,

            # Instantiate Early Stopping Callback
            callbacks = [StartTrainingCallback(), StartTestingCallback(),
                         EarlyStopping(monitor='train_loss', mode='min', min_delta=0, patience=PATIENCE, verbose=True)],

            # Custom TensorBoard Logger
            logger = pl_loggers.TensorBoardLogger(save_dir=f'{PACKAGE_PATH}/model/data/logs/'),

            # Developer Test Mode
            fast_dev_run = False

        )

        # Train Model
        trainer.fit(self.model, train_dataloaders=self.train_dataloader, val_dataloaders=self.val_dataloader)

        # Test Model
        trainer.test(self.model, dataloaders=self.test_dataloader)

        # Validate Model
        trainer.validate(self.model, dataloaders=self.val_dataloader)

        # Save Model
        save_model(f'{PACKAGE_PATH}/model', self.model_name, self.model)

if __name__ == '__main__':

    # Train LSTM Network
    training_network = TrainingNetwork(BATCH_SIZE, MODEL_TYPE, SEQUENCE_LENGTH, STRIDE, OPEN_GRIPPER_LEN, shuffle=True) 
    training_network.train_network()
