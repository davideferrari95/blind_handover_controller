from typing import List, Tuple, Optional
from termcolor import colored

import torch
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy

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

    # Neural Network Creation Function
    def mlp(self, input_size:int, output_size:int, hidden_size:Optional[List[int]]=[512,256], hidden_mod=torch.nn.ReLU(), output_mod=Optional[torch.nn.Module]):

        ''' Neural Network Creation Function '''

        # No Hidden Layers
        if hidden_size is None or hidden_size == []:

            # Only one Linear Layer
            net = [torch.nn.Linear(input_size, output_size)]

        else:

            # First Layer with ReLU Activation
            net = [torch.nn.Linear(input_size, hidden_size[0]), hidden_mod]

            # Add the Hidden Layers
            for i in range(len(hidden_size) - 1):
                net += [torch.nn.Linear(hidden_size[i], hidden_size[i+1]), hidden_mod]

            # Add the Output Layer
            net.append(torch.nn.Linear(hidden_size[-1], output_size))

        if output_mod is not None:
            net.append(output_mod)

        # Create a Sequential Neural Network
        return torch.nn.Sequential(*net)

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
        self.net = self.mlp(input_size, num_classes, hidden_size, torch.nn.ReLU(), torch.nn.Softmax(dim=1))

        print(colored(f'Model Initialized:\n\n', 'green'), self, '\n')

        # Initialize Accuracy Metrics
        self.train_accuracy, self.test_accuracy, self.val_accuracy = Accuracy(task="binary"), Accuracy(task="binary"), Accuracy(task="binary")

        # Initialize Loss (Weighted Loss - Class Imbalance)
        self.loss = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.valid_loss, self.num_val_batches  = 0, 0
        self.test_loss,  self.num_test_batches = 0, 0

    def forward(self, x:torch.Tensor):

        # Reshape X to a Flat Vector
        x = x.view(x.size(0), -1)

        # Pass through NN Layers
        return self.net(x)

class BinaryClassifierModel(BaseModel):

    """ Binary-Classifier Neural Network PyTorch Lightning Model """

    def __init__(self, input_size:int, hidden_size:List[int], learning_rate:float=0.001):

        """ Neural Network Initialization """

        super(BinaryClassifierModel, self).__init__()

        # Save Hyperparameters
        self.input_size, self.output_size, self.hidden_size = input_size, 1, hidden_size
        self.learning_rate = learning_rate

        # Create Neural Network Fully Connected Layers
        self.net = self.mlp(input_size, 1, hidden_size, torch.nn.ReLU(), torch.nn.Sigmoid())

        print(colored(f'Model Initialized:\n\n', 'green'), self, '\n')

        # Initialize Accuracy Metrics
        self.train_accuracy, self.test_accuracy, self.val_accuracy = Accuracy(task="binary"), Accuracy(task="binary"), Accuracy(task="binary")

        # Initialize Loss (Weighted Loss - Class Imbalance)
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.valid_loss, self.num_val_batches  = 0, 0
        self.test_loss,  self.num_test_batches = 0, 0

    def forward(self, x:torch.Tensor):

        # Reshape X to a Flat Vector
        x = x.view(x.size(0), -1)

        # Pass through NN Layers
        return self.net(x)

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
        self.input_size, self.output_size, self.hidden_size = input_size, output_size, hidden_size
        self.learning_rate = learning_rate

        # Create Neural Network Layers
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size[0], num_layers=num_layers, batch_first=True)

        # Create Fully Connected Layers
        self.net = self.mlp(hidden_size[0], output_size, hidden_size[1:], torch.nn.ReLU(), torch.nn.Sigmoid())

        print(colored(f'Model Initialized:\n\n', 'green'), self, '\n')

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
        return self.net(lstm_out[:, -1, :])
