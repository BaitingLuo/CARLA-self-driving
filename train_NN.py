import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import joblib
# Load data
data = pd.read_csv('training_data.csv', header=None)

# Define the speed threshold and the number of augmentations per data point
speed_threshold = 10  # Define what you consider as 'small' speed
num_augmentations = 1000  # Number of augmented versions per data point

# Filter instances with small speed values
small_speed_data = data[data[9] < speed_threshold]  # Assuming the speed value is in column 10 (index 9)

# Augment this data
augmented_data = []
for _ in range(num_augmentations):
    # Add small random noise to the features (excluding the speed feature)
    noise = np.random.normal(0, 0.1, small_speed_data.shape)
    noise[:, 9] = 0  # Do not add noise to the speed feature
    augmented_data.append(small_speed_data + noise)

# Combine all augmented data into a single DataFrame
augmented_data = pd.concat(augmented_data)

# Combine augmented data with the original dataset
enhanced_dataset = pd.concat([data, augmented_data])

# Optionally, shuffle the combined dataset
enhanced_dataset = enhanced_dataset.sample(frac=1).reset_index(drop=True)


# Assuming the first 10 columns are inputs and the rest are outputs
X = data.iloc[:, :10].values
y = data.iloc[:, 10:].values

# Initialize separate scalers for X and y
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# Normalize input features X
X_scaled = scaler_X.fit_transform(X)

# Normalize target outputs y
y_scaled = scaler_y.fit_transform(y)

# Save the scalers to files
joblib.dump(scaler_X, 'scaler_X.pkl')
joblib.dump(scaler_y, 'scaler_y.pkl')

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

# Create dataloaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(10, 50)  # Adjust the sizes
        self.fc2 = nn.Linear(50, 20)  # Adjust the sizes
        self.fc3 = nn.Linear(20, y.shape[1])  # Output size

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = NeuralNetwork()

# Loss and optimizer
criterion = nn.MSELoss()  # For regression, change if different task
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Number of epochs
num_epochs = 1000  # Adjust as needed

# Training loop
best_val_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation step
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            #if epoch == 9:
            #    print("truth:",targets)
            #    print("predictions:", outputs)
            #    print("##########################")
            val_loss += loss.item()
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # Save the model
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"Model saved at Epoch {epoch+1} with Validation Loss: {val_loss:.4f}")
    val_loss /= len(val_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')

# Save the model
#torch.save(model.state_dict(), 'model.pth')