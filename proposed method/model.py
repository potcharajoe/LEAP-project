import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

class ParkinsonsLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, cnn_channels, kernel_size):
        super(ParkinsonsLSTM, self).__init__()
        self.conv1 = nn.Conv1d(input_size, cnn_channels, kernel_size, padding='same')
        self.lstm = nn.LSTM(cnn_channels, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, data, mask):
        lengths = torch.sum(mask, dim=1)[:,0]
        data = data.transpose(1, 2)
        data = self.conv1(data)
        data = data.transpose(1, 2)

        packed_data = pack_padded_sequence(data, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed_data)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        pooled_out = torch.mean(out, dim=1)
        out = self.dropout(pooled_out)
        out = self.fc(out)
        
        return out
    
def train(model, train_loader, optimizer, criterion, device):
    model.train()
    train_losses, total_correct, total_samples = 0, 0, 0
    for idx, (data, mask, targets) in enumerate(tqdm(train_loader, desc="Training")):
        data, mask, targets= data.to(device), mask.to(device), targets.to(device)
        outputs = model(data, mask)
        loss = criterion(outputs, targets.long())
        masked_loss = torch.sum(loss * mask[:, 0, 0]) / torch.sum(mask[:, 0, 0])

        optimizer.zero_grad()
        masked_loss.backward()
        optimizer.step()

        train_losses += masked_loss.item()
        
        predictions = torch.argmax(outputs, dim=1)

        total_correct += torch.sum(predictions == targets).item()
        total_samples += len(targets)

    loss = train_losses/len(train_loader)    
    accuracy = 100 * total_correct / total_samples
    
    return loss, accuracy

def eval(model, val_loader, criterion, device):
    model.eval()
    with torch.no_grad():
        total_losses = 0
        total_correct = 0
        total_samples = 0
        all_predictions = torch.tensor([])
        all_labels = torch.tensor([])

        for idx, (data, mask, targets) in enumerate(tqdm(val_loader, desc="Evaluation")):
            data, mask, targets = data.to(device), mask.to(device), targets.to(device)

            outputs = model(data, mask)
            
            loss = criterion(outputs, targets.long())
            total_losses += torch.sum(loss * mask[:, 0, 0]) / torch.sum(mask[:, 0, 0])

            
            predictions = torch.argmax(outputs, dim=1)
            all_predictions = torch.cat((all_predictions, predictions))
            all_labels = torch.cat((all_labels, targets))
            
            total_correct += torch.sum(predictions == targets).item()
            total_samples += len(targets)

        loss = total_losses / len(val_loader)
        accuracy = 100 * total_correct / total_samples

        report = classification_report(
            all_predictions, all_labels, target_names=["Healthy", "Parkinsons"], zero_division=0
        )
        cm = confusion_matrix(all_labels, all_predictions)
    return loss, accuracy, report, cm