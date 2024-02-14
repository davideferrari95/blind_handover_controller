from typing import List, Tuple
from termcolor import colored

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import Trainer, loggers as pl_loggers
from torchmetrics.classification import Accuracy

# Import Processed Dataset and DataLoader
from process_dataset import ProcessDataset, PACKAGE_PATH
from process_dataset import BATCH_SIZE, SEQUENCE_LENGTH, OPEN_GRIPPER_LEN, STRIDE, BALANCE_STRATEGY

# Import Callbacks and Utilities
from pl_utils import save_model, save_hyperparameters, DEVICE
from pl_utils import StartTestingCallback, StartTrainingCallback, StartValidationCallback

# Set Torch Matmul Precision
torch.set_float32_matmul_precision('high')

class BaseModel(pl.LightningModule):

    """ Base Neural Network PyTorch Lightning Model """

    def __init__(self):

        """ Neural Network Initialization """

        super(BaseModel, self).__init__()

        # Initialize Accuracy Metrics
        self.train_accuracy, self.test_accuracy, self.val_accuracy = Accuracy(task="binary"), Accuracy(task="binary"), Accuracy(task="binary")

        # Initialize Loss
        self.loss = torch.nn.BCELoss()
        self.valid_loss, self.num_val_batches  = 0, 0
        self.test_loss,  self.num_test_batches = 0, 0

    def forward(self, x:torch.Tensor):

        return x

    def training_step(self, batch:Tuple[torch.Tensor, torch.Tensor], batch_idx):

        """ Training Step """

        x, y = batch
        y_pred:torch.Tensor = self(x)

        # Calculate Loss
        loss:torch.Tensor = self.loss(y_pred, y.float())
        self.log('train_loss', loss)

        # Update Accuracy Metric
        self.train_accuracy(y_pred, y)
        self.log('train_accuracy', self.train_accuracy, prog_bar=True)

        return loss

    def validation_step(self, batch:Tuple[torch.Tensor, torch.Tensor], batch_idx):

        """ Validation Step """

        x, y = batch
        y_pred:torch.Tensor = self(x)

        # Calculate Loss for Validation
        val_loss:torch.Tensor = self.loss(y_pred, y.float())

        # Update Accuracy Metric
        self.val_accuracy(y_pred, y)

        # Update Validation Loss
        self.valid_loss += val_loss.item()
        self.num_val_batches += 1
        return val_loss

    def on_validation_epoch_end(self):

        """ Validation Epoch End """

        # Calculate Average Validation Loss
        avg_val_loss = self.valid_loss / self.num_val_batches
        self.log('val_loss', avg_val_loss)
        self.log('val_accuracy', self.val_accuracy.compute(), prog_bar=True)

    def test_step(self, batch:Tuple[torch.Tensor, torch.Tensor], batch_idx):

        """ Test Step """

        x, y = batch
        y_pred:torch.Tensor = self(x)

        # Calculate Loss for Test
        test_loss:torch.Tensor = self.loss(y_pred, y.float())

        # Update Accuracy Metric
        self.test_accuracy(y_pred, y)

        # Update Test Loss
        self.test_loss += test_loss.item()
        self.num_test_batches += 1
        return test_loss

    def on_test_epoch_end(self):

        """ Test Epoch End """

        # Calculate Average Test Loss
        avg_test_loss = self.test_loss / self.num_test_batches
        self.log('test_loss', avg_test_loss)
        self.log('test_accuracy', self.test_accuracy.compute(), prog_bar=True)

    def configure_optimizers(self):

        """ Configure Optimizer """

        # Use AdamW Optimizer
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

class FeedforwardModel(BaseModel):

    """ Feedforward Neural Network PyTorch Lightning Model """

    def __init__(self, input_size:int, hidden_size:List[int], output_size:int, learning_rate:float=0.001):

        """ Neural Network Initialization """

        super(FeedforwardModel, self).__init__()

        # Save Hyperparameters
        self.input_size, self.output_size, self.hidden_size = input_size, output_size, hidden_size
        self.learning_rate = learning_rate

        # Create Neural Network Fully Connected Layers
        self.fc1 = torch.nn.Linear(input_size, hidden_size[0])
        self.fc2 = torch.nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = torch.nn.Linear(hidden_size[-1], output_size)

        # Sigmoid Activation
        self.sigmoid = torch.nn.Sigmoid()

        # Initialize Accuracy Metrics
        self.train_accuracy, self.test_accuracy, self.val_accuracy = Accuracy(task="binary"), Accuracy(task="binary"), Accuracy(task="binary")

        # Initialize Loss
        self.loss = torch.nn.BCELoss()
        self.valid_loss, self.num_val_batches  = 0, 0
        self.test_loss,  self.num_test_batches = 0, 0

    def forward(self, x:torch.Tensor):

        # Reshape X to a Flat Vector
        x = x.view(x.size(0), -1)

        # Pass through Fully Connected Layers
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        x = self.fc3(x)

        # Sigmoid Activation
        return self.sigmoid(x)

class MultiClassifierModel(BaseModel):

    """ Multi-Classifier Neural Network PyTorch Lightning Model """

    def __init__(self, input_size:int, hidden_size:List[int], num_classes:int, learning_rate:float=0.001, class_weights:torch.Tensor=torch.tensor([1.0, 1.0])):

        """ Neural Network Initialization """

        super(MultiClassifierModel, self).__init__()

        # Save Hyperparameters
        self.input_size, self.output_size, self.hidden_size = input_size, num_classes, hidden_size
        self.learning_rate = learning_rate

        # Create Neural Network Fully Connected Layers
        self.fc1 = torch.nn.Linear(input_size, hidden_size[0])
        self.fc2 = torch.nn.Linear(hidden_size[0], hidden_size[1])
        self.fc3 = torch.nn.Linear(hidden_size[-1], num_classes)

        print(colored(f'Model Initialized:\n', 'green'), self, '\n')

        # Initialize Accuracy Metrics
        self.train_accuracy, self.test_accuracy, self.val_accuracy = Accuracy(task="binary"), Accuracy(task="binary"), Accuracy(task="binary")

        # Initialize Loss (Weighted Loss - Class Imbalance)
        self.loss = torch.nn.BCEWithLogitsLoss(weight=class_weights)
        self.valid_loss, self.num_val_batches  = 0, 0
        self.test_loss,  self.num_test_batches = 0, 0

    def forward(self, x:torch.Tensor):

        # print(x.shape)

        # Reshape X to a Flat Vector
        x = x.view(x.size(0), -1)

        # print(x.shape)

        # Pass through Fully Connected Layers
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        x = self.fc3(x)

        return x

class CNNModel(BaseModel):

    """ Convolutional Neural Network PyTorch Lightning Model """

    def __init__(self, input_channels:int, hidden_size:List[int], output_size:int, sequence_length:int=100, learning_rate=0.001):

        """ Neural Network Initialization """

        super(CNNModel, self).__init__()

        # Save Hyperparameters
        self.input_channels, self.output_size, self.hidden_size = input_channels, output_size, hidden_size
        self.learning_rate = learning_rate

        # Create Neural Network Layers
        self.conv1d = torch.nn.Conv1d(in_channels=input_channels, out_channels=hidden_size, kernel_size=3)
        self.fc1 = torch.nn.Linear(hidden_size * (sequence_length - 2), output_size)

        # Sigmoid Activation
        self.sigmoid = torch.nn.Sigmoid()

        # Initialize Accuracy Metrics
        self.train_accuracy, self.test_accuracy, self.val_accuracy = Accuracy(task="binary"), Accuracy(task="binary"), Accuracy(task="binary")

        # Initialize Loss
        self.loss = torch.nn.BCELoss()
        self.valid_loss, self.num_val_batches  = 0, 0
        self.test_loss,  self.num_test_batches = 0, 0

    def forward(self, x:torch.Tensor):

        # Pass through Conv1D Layer
        x = self.conv1d(x)
        x = torch.nn.functional.relu(x)

        # Reshape to a Flat Vector
        x = x.view(x.size(0), -1)

        # Pass through Fully Connected Layer
        x = self.fc1(x)

        # Sigmoid Activation
        return self.sigmoid(x)

class LSTMModel(BaseModel):

    """ LSTM Neural Network PyTorch Lightning Model """

    def __init__(self, input_size:int, hidden_size:List[int], output_size:int, num_layers:int, learning_rate:float=0.001):

        """ Neural Network Initialization """

        super(LSTMModel, self).__init__()

        # Save Hyperparameters
        self.input_size, self.output_size = input_size, output_size
        self.hidden_size, self.num_layers = hidden_size, num_layers
        self.learning_rate = learning_rate

        # Create Neural Network Layers
        # self.gru = torch.nn.GRU(input_size=input_size, hidden_size=hidden_size[0], num_layers=num_layers, batch_first=True)
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size[0], num_layers=num_layers, batch_first=True)

        # Hidden Fully Connected Layer
        self.fc1 = torch.nn.Linear(hidden_size[0], hidden_size[1])

        # Last Fully Connected Layer
        self.fc2 = torch.nn.Linear(hidden_size[-1], output_size)

        # Sigmoid Activation
        self.sigmoid = torch.nn.Sigmoid()

        # Initialize Accuracy Metrics
        self.train_accuracy, self.test_accuracy, self.val_accuracy = Accuracy(task="binary"), Accuracy(task="binary"), Accuracy(task="binary")

        # Initialize Loss
        self.loss = torch.nn.BCELoss()
        self.valid_loss, self.num_val_batches  = 0, 0
        self.test_loss,  self.num_test_batches = 0, 0

    def forward(self, x:torch.Tensor):

        """ Forward Pass """

        # Pass through LSTM Layer
        lstm_out, _ = self.lstm(x)

        # Only Last Time Step Output
        output = self.fc1(lstm_out[:, -1, :])
        output = self.fc2(output)

        # Sigmoid Activation
        return self.sigmoid(output)

class TrainingNetwork():

    """ Train LSTM Network """

    def __init__(self, batch_size:int=32, sequence_length:int=100, stride:int=10, open_gripper_len:int=100, shuffle:bool=True):

        # Process Dataset
        process_dataset = ProcessDataset(batch_size, sequence_length, stride, open_gripper_len, shuffle, BALANCE_STRATEGY)
        class_weights = process_dataset.get_class_weights()

        # Get DataLoaders
        self.train_dataloader, self.test_dataloader, self.val_dataloader = process_dataset.get_dataloaders()

        # Model Hyperparameters (Input Size, Hidden Size, Output Size, Number of Layers)
        input_size, hidden_size, output_size, num_layers = process_dataset.sequence_shape[1], [256, 128], 2, 1
        learning_rate = 0.001

        # Save Hyperparameters in Config File
        save_hyperparameters(f'{PACKAGE_PATH}/model', input_size * sequence_length, hidden_size, output_size, num_layers, learning_rate)

        # Create NN Model
        # self.model = FeedforwardModel(input_size * sequence_length, hidden_size, output_size, learning_rate).to(DEVICE)
        self.model = MultiClassifierModel(input_size * sequence_length, hidden_size, output_size, learning_rate, class_weights).to(DEVICE)
        # self.model = CNNModel(input_size, hidden_size, output_size, sequence_length, learning_rate).to(DEVICE)
        # self.model = LSTMModel(input_size, hidden_size, output_size, num_layers, learning_rate).to(DEVICE)

    def train_network(self):

        """ Train LSTM Network """

        # PyTorch Lightning Trainer
        trainer = Trainer(

            # Devices
            devices= 'auto',

            # Hyperparameters
            # min_epochs = 200,
            max_epochs = 500,
            log_every_n_steps = 1,

            # Instantiate Early Stopping Callback
            # callbacks = [StartTrainingCallback(), StartTestingCallback(), StartValidationCallback(),
            callbacks = [StartTrainingCallback(), StartTestingCallback(),
                        EarlyStopping(monitor='train_loss', mode='min', min_delta=0, patience=50, verbose=True)],

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
        # save_model(os.path.join(f'{PACKAGE_PATH}/model'), 'model.pth', self.model)

if __name__ == '__main__':

    # Train LSTM Network
    training_network = TrainingNetwork(batch_size=BATCH_SIZE, sequence_length=SEQUENCE_LENGTH, open_gripper_len=OPEN_GRIPPER_LEN, stride=STRIDE, shuffle=True)
    training_network.train_network()
