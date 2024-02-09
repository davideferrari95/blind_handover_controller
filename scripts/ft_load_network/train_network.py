import pandas as pd

import torch, pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_sequence

class CustomDataset(Dataset):

    def __init__(self, packed_sequences, labels):

        # Dataset Initialization
        self.packed_sequences = packed_sequences
        self.labels = labels
    
    def __len__(self):

        return len(self.labels)
    
    def __getitem__(self, idx):

        return self.packed_sequences[idx], self.labels[idx]

# Leggi il tuo file CSV
df = pd.read_csv('tuo_file.csv')

# Aggiungi una colonna con l'ID dell'esperimento
df['experiment_id'] = df.groupby('experiment')['experiment'].ngroup()

# Crea il tuo dataset
sequences = []
labels = []

for group_id, group_df in df.groupby('experiment_id'):
    sequence = torch.tensor(group_df[['v[1]', 'v[2]', 'v[3]', 'v[4]', 'v[5]', 'v[6]', 'fx']].values, dtype=torch.float32)
    label = torch.tensor(group_df['gripper_parameter'].values, dtype=torch.long)  # Supponendo che il parametro booleano sia in 'gripper_parameter'
    
    sequences.append(sequence)
    labels.append(label)

# Utilizza pack_sequence per gestire le sequenze con lunghezze diverse
packed_sequences = pack_sequence(sequences)
labels = torch.cat(labels)

# Usa CustomDataset nel resto del tuo codice per creare il dataloader e addestrare il modello


# # Definisci la tua rete LSTM
# class GripperLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(GripperLSTM, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
    
#     def forward(self, x):
#         out, _ = self.lstm(x)
#         out = self.fc(out[:, -1, :])  # Utilizza solo l'output dell'ultima sequenza
#         return out

# # Definisci il tuo dataset personalizzato
# class CustomDataset(Dataset):
#     def __init__(self, data, labels):
#         self.data = data
#         self.labels = labels
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])

# # Definisci il tuo modulo di Lightning
# class GripperLSTMModule(pl.LightningModule):
#     def __init__(self, input_size, hidden_size, output_size, learning_rate):
#         super(GripperLSTMModule, self).__init__()
#         self.model = GripperLSTM(input_size, hidden_size, output_size)
#         self.criterion = nn.CrossEntropyLoss()
#         self.learning_rate = learning_rate

#     def forward(self, x):
#         return self.model(x)

#     def training_step(self, batch, batch_idx):
#         inputs, labels = batch
#         outputs = self(inputs)
#         loss = self.criterion(outputs, labels)
#         return loss

#     def configure_optimizers(self):
#         return optim.Adam(self.model.parameters(), lr=self.learning_rate)

# # Configura e addestra il modello
# input_size = 6  # Dimensione dell'input (ad esempio, 6 per le letture di forza)
# hidden_size = 64  # Dimensione dello strato nascosto LSTM
# output_size = 2  # Dimensione dell'output (apri o chiudi gripper)
# learning_rate = 0.001

# model = GripperLSTMModule(input_size, hidden_size, output_size, learning_rate)
# dataset = CustomDataset(data, labels)  # Sostituisci con i tuoi dati e etichette
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# trainer = pl.Trainer(max_epochs=10)
# trainer.fit(model, dataloader)
