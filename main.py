import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

EPOCH = 600
LR = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

x_all = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32).to(device)
y_all = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32).to(device)

dataset = TensorDataset(x_all, y_all)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

class LinearModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.layer1 = nn.Linear(3, 5)
		self.layer2 = nn.Linear(5, 1)

	def forward(self, x):
		tmp = torch.sigmoid(self.layer1(x))
		return self.layer2(tmp)
	
model = LinearModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_history = []

for epoch in range(EPOCH):
	for x_batch, y_batch in dataloader:
		x_batch = torch.cat((x_batch, torch.ones(x_batch.size(0), 1).to(device)), dim=1)
		y_pred = model(x_batch)
		loss = criterion(y_pred, y_batch)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	loss_history.append(loss.item())

	if (epoch + 1) % 100 == 0:
		print(f"Epoch [{epoch + 1}/{EPOCH}], Loss: {loss.item():.4f}")

print("The training has finished.")
with torch.no_grad():
	x_all = torch.cat((x_all, torch.ones(x_all.size(0), 1).to(device)), dim=1)
	y_pred = model(x_all)
	print("Predicted outputs:")
	print(y_pred.cpu().numpy())

# Visualization: training loss curve and output comparison
y_true = y_all[:, 0].detach().cpu().numpy()
y_pred_np = y_pred[:, 0].detach().cpu().numpy()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(range(1, EPOCH + 1), loss_history, color="tab:blue")
axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("MSE Loss")
axes[0].grid(alpha=0.3)

indices = list(range(4))
width = 0.35
axes[1].bar([i - width / 2 for i in indices], y_true, width=width, label="True", color="tab:green")
axes[1].bar([i + width / 2 for i in indices], y_pred_np, width=width, label="Pred", color="tab:orange")
axes[1].set_xticks(indices)
axes[1].set_xticklabels(["[0,0]", "[0,1]", "[1,0]", "[1,1]"])
axes[1].set_title("XOR Outputs: True vs Predicted")
axes[1].set_ylabel("Output Value")
axes[1].legend()
axes[1].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.show()

