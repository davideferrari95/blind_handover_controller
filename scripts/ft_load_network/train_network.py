import torch, os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import Trainer, loggers as pl_loggers

# Import Processed Dataset and DataLoader
from process_dataset import ProcessDataset, PACKAGE_PATH

# Import PyTorch Lightning Callbacks
from pl_utils import save_model, StartTestingCallback, StartTrainingCallback, DEVICE

# Set Torch Matmul Precision
torch.set_float32_matmul_precision('high')

class LSTMModel(pl.LightningModule):

    """ LSTM Model Network """

    def __init__(self, input_size:int, hidden_size:int, output_size:int, num_layers:int, learning_rate:float=0.001):
        super(LSTMModel, self).__init__()

        # Save Hyperparameters
        self.input_size, self.output_size = input_size, output_size
        self.hidden_size, self.num_layers = hidden_size, num_layers
        self.learning_rate = learning_rate

        # Create Neural Network Layers
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):

        """ Forward Pass """

        # Pass through LSTM Layer
        lstm_out, _ = self.lstm(x)

        # Only take the output at the last time step
        output = self.fc(lstm_out[:, -1, :])
        return self.sigmoid(output)

    def training_step(self, batch, batch_idx):

        """ Training Step """

        x, y = batch
        y_pred = self(x)

        # Calculate Loss
        loss = torch.nn.BCELoss()(y_pred.squeeze(), y)
        
        return loss

    def configure_optimizers(self):

        """ Configure Optimizer """
        
        # Use Adam Optimizer
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

class TrainingNetwork():

    """ Train LSTM Network """

    def __init__(self, batch_size:int=32, sequence_length:int=100, shuffle:bool=True):

        # Process Dataset
        process_dataset = ProcessDataset(batch_size, sequence_length, shuffle)

        # Get Dataset and  DataLoader
        dataframe, self.dataloader = process_dataset.get_dataframe(), process_dataset.get_dataloader()

        # Model Hyperparameters (Input Size, Hidden Size, Output Size, Number of Layers)
        input_size, hidden_size, output_size, num_layers = dataframe.shape[1] - 2, 64, 1, 1
        learning_rate = 0.001

        # Create LSTM Model
        self.lstm_model = LSTMModel(input_size, hidden_size, output_size, num_layers, learning_rate).to(DEVICE)

    def train_network(self):

        """ Train LSTM Network """

        # PyTorch Lightning Trainer
        trainer = Trainer(

            # Devices
            devices= 'auto',

            # Hyperparameters
            # min_epochs = 200,
            max_epochs = 2000,
            log_every_n_steps = 1,

            # Instantiate Early Stopping Callback
            callbacks = [StartTrainingCallback(), StartTestingCallback(),
                        EarlyStopping(monitor='train_loss', mode='min', min_delta=0, patience=100, verbose=True)],

            # Custom TensorBoard Logger
            logger = pl_loggers.TensorBoardLogger(save_dir=f'{PACKAGE_PATH}/model/data/logs/'),

            # Developer Test Mode
            fast_dev_run = True

        )

        # Train Model
        trainer.fit(self.lstm_model, train_dataloaders=self.dataloader)

        # Save Model
        save_model(os.path.join(f'{PACKAGE_PATH}/model'), 'model.pth', self.lstm_model)

if __name__ == '__main__':

    # Train LSTM Network
    training_network = TrainingNetwork(batch_size=64, sequence_length=100, shuffle=True)
    training_network.train_network()
