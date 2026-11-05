import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------
# Configuration
# -----------------------------
SEQUENCE_LENGTH = 28
BATCH_SIZE = 256
EPOCHS = 10
HIDDEN_SIZE = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)

# -----------------------------
# Load and prepare data
# -----------------------------
df = pd.read_csv(r"C:\Users\Shib\IntelliStock\data\train (1).csv")
df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(["store", "item", "date"])

# Create a complete date index for every store-item series
dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D")

series_list = []

for (store, item), group in df.groupby(["store", "item"]):
    sales = (
        group.set_index("date")["sales"]
        .reindex(dates)
        .fillna(0)
        .to_numpy(dtype=np.float32)
    )
    series_list.append(sales)

sales_array = np.stack(series_list)

# -----------------------------
# Chronological split
# -----------------------------
cutoff_date = pd.Timestamp("2017-10-02")
cutoff_idx = np.where(dates == cutoff_date)[0][0]

# Normalize using training data only
train_values = sales_array[:, : cutoff_idx + 1]

mean_sales = train_values.mean()
std_sales = train_values.std()

sales_scaled = (sales_array - mean_sales) / (std_sales + 1e-8)


# -----------------------------
# Dataset
# -----------------------------
class SalesSequenceDataset(Dataset):
    def __init__(self, data, start_idx, end_idx, seq_len):
        self.data = data
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.seq_len = seq_len
        self.n_series = data.shape[0]

        self.samples_per_series = end_idx - start_idx

    def __len__(self):
        return self.n_series * self.samples_per_series

    def __getitem__(self, index):
        series_idx = index // self.samples_per_series
        offset = index % self.samples_per_series

        target_idx = self.start_idx + offset

        x = self.data[series_idx, target_idx - self.seq_len : target_idx]

        y = self.data[series_idx, target_idx]

        x = torch.tensor(x, dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(y, dtype=torch.float32)

        return x, y


# Training targets begin after the first 28 days
train_dataset = SalesSequenceDataset(
    sales_scaled,
    start_idx=SEQUENCE_LENGTH,
    end_idx=cutoff_idx + 1,
    seq_len=SEQUENCE_LENGTH,
)

# Validation targets: Oct 3 to Dec 31, 2017
val_dataset = SalesSequenceDataset(
    sales_scaled, start_idx=cutoff_idx + 1, end_idx=len(dates), seq_len=SEQUENCE_LENGTH
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)

val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

print("Training samples:", len(train_dataset))
print("Validation samples:", len(val_dataset))


# -----------------------------
# LSTM Model
# -----------------------------
class LSTMForecaster(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=HIDDEN_SIZE,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )

        self.fc = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        last_output = output[:, -1, :]
        prediction = self.fc(last_output)

        return prediction.squeeze(-1)


model = LSTMForecaster().to(DEVICE)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# Training
# -----------------------------
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for x, y in train_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        predictions = model(x)
        loss = criterion(predictions, y)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Validation loss
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            predictions = model(x)
            loss = criterion(predictions, y)

            val_loss += loss.item()

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss / len(train_loader):.4f} | "
        f"Val Loss: {val_loss / len(val_loader):.4f}"
    )

# -----------------------------
# Validation predictions
# -----------------------------
model.eval()

predictions_list = []
actual_list = []

with torch.no_grad():
    for x, y in val_loader:
        x = x.to(DEVICE)

        predictions = model(x).cpu().numpy()
        actual = y.numpy()

        predictions_list.extend(predictions)
        actual_list.extend(actual)

predictions = np.array(predictions_list) * std_sales + mean_sales
actual = np.array(actual_list) * std_sales + mean_sales

predictions = np.maximum(predictions, 0)

mae = mean_absolute_error(actual, predictions)
rmse = np.sqrt(mean_squared_error(actual, predictions))

wape = (np.sum(np.abs(actual - predictions)) / np.sum(np.abs(actual))) * 100

print("\nLSTM Performance")
print("----------------")
print(f"MAE:  {mae:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"WAPE: {wape:.2f}%")

# -----------------------------
# Save model
# -----------------------------
os.makedirs("models", exist_ok=True)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "mean_sales": mean_sales,
        "std_sales": std_sales,
    },
    "models/lstm_model.pth",
)

print("Model saved successfully.")
