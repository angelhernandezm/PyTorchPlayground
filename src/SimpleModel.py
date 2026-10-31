import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

# Fake data: 1000 samples, 20 features, 3 classes
X = torch.randn(1000, 20)
y = torch.randint(0, 3, (1000,))
loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 3),       # output raw logits, no softmax
        )

    def forward(self, x):
        return self.net(x)

model = MLP().to(device)
loss_fn = nn.CrossEntropyLoss()     # applies softmax internally
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        logits = model(xb)              # 1. forward pass
        loss = loss_fn(logits, yb)      # 2. compute loss
        optimizer.zero_grad()           # 3. clear old gradients
        loss.backward()                 # 4. backprop
        optimizer.step()                # 5. update weights

    print(f"epoch {epoch}: loss {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():                   # turn off gradient tracking
    preds = model(X.to(device)).argmax(dim=1)
    print("accuracy:", (preds.cpu() == y).float().mean().item())