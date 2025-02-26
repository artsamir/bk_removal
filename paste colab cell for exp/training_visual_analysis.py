import os
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from cutnet_model import CUTNET  # Change based on your model
from data_loader import SalObjDataset  # Your dataset class
from loss_functions import muti_bce_loss_fusion  # Import your loss function

# Hyperparameters
epoch_num = 50  # Number of epochs
batch_size_train = 2
learning_rate = 0.001
save_frq = 50  # Save model every N iterations

# Paths
dataset_path = "/content/bk_removal/my_dataset/"
model_save_path = "/content/drive/MyDrive/cutnet_models/saved_models/my_train_model.pth"

# Initialize Model
net = CUTNET(3, 1)  # Adjust based on CUTNET or CUTNETP
if torch.cuda.is_available():
    net.cuda()

# Optimizer
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# Dataset & Dataloader
tra_image_dir = os.path.join(dataset_path, "input_images")
tra_mask_dir = os.path.join(dataset_path, "masks_images")
salobj_dataset = SalObjDataset(tra_image_dir, tra_mask_dir)
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=2)

# Loss Tracking
loss_values = []
target_loss_values = []
iterations = []
ite_num = 0

# Training Loop
print("🚀 Training started...")
for epoch in range(epoch_num):
    net.train()
    for i, data in enumerate(salobj_dataloader):
        ite_num += 1

        # Load Data
        inputs, labels = data["image"], data["label"]
        inputs, labels = inputs.float(), labels.float()

        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs_v, labels_v = Variable(inputs), Variable(labels)

        optimizer.zero_grad()

        # Forward Pass
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Track Loss
        loss_values.append(loss.item())
        target_loss_values.append(loss2.item())
        iterations.append(ite_num)

        print(f"[Epoch {epoch+1}, Iter {ite_num}] Loss: {loss.item():.4f}, Target Loss: {loss2.item():.4f}")

        # Save Model Periodically
        if ite_num % save_frq == 0:
            torch.save(net.state_dict(), model_save_path)
            print(f"✅ Model saved at: {model_save_path}")

# Plot Loss Curve
plt.plot(iterations, loss_values, label="Total Loss")
plt.plot(iterations, target_loss_values, label="Target Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.show()

print("🎉 Training complete!")
