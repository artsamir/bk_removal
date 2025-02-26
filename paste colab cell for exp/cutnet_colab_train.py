import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import glob

# Import custom dataset and model
from data_loader import RescaleT, RandomCrop, ToTensorLab, SalObjDataset
from cutnet_model import CUTNET, CUTNETP

# ======= CHECK & SET DATASET PATHS =======
base_dir = "/content/bk_removal/my_dataset"
tra_image_dir = os.path.join(base_dir, "input_images")
tra_label_dir = os.path.join(base_dir, "masks_images")

if not os.path.exists(tra_image_dir):
    raise FileNotFoundError(f"❌ '{tra_image_dir}' folder not found!")
if not os.path.exists(tra_label_dir):
    raise FileNotFoundError(f"❌ '{tra_label_dir}' folder not found!")

# ======= SET MODEL SAVE PATH =======
drive_save_dir = "/content/drive/MyDrive/cutnet_models/saved_models"
os.makedirs(drive_save_dir, exist_ok=True)  # Ensure save directory exists

model_name = 'u2net'  # Choose 'u2net' or 'u2netp'
model_save_path = os.path.join(drive_save_dir, "my_train_model.pth")

# ======= BINARY CROSS-ENTROPY LOSS =======
bce_loss = nn.BCELoss(reduction='mean')

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    """Computes multiple loss values for different decoder outputs."""
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    total_loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    return loss0, total_loss

# ======= LOAD TRAINING DATA =======
image_ext = ('.jpg', '.jpeg', '.png')
label_ext = '.png'

# Get all training images
tra_img_name_list = [img for ext in image_ext for img in glob.glob(os.path.join(tra_image_dir, '**', '*' + ext), recursive=True)]

# Get corresponding mask images
tra_lbl_name_list = [
    os.path.join(tra_label_dir, os.path.splitext(os.path.relpath(img, tra_image_dir))[0] + label_ext)
    for img in tra_img_name_list
]

print("--- Dataset Loaded Successfully ---")
print(f"✅ Train images found: {len(tra_img_name_list)}")
print(f"✅ Train masks found: {len(tra_lbl_name_list)}")
print("----------------------------------")

# ======= DEFINE DATASET & DATALOADER =======
batch_size_train = 5  # Adjust based on GPU memory

salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms.Compose([
        RescaleT(320),
        RandomCrop(288),
        ToTensorLab(flag=0)
    ])
)

salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=2)

# ======= MODEL INITIALIZATION =======
net = CUTNET(3, 1) if model_name == 'u2net' else CUTNETP(3, 1)
if torch.cuda.is_available():
    net.cuda()
    print("🔥 Training on GPU")
else:
    print("⚠️ GPU not found, training on CPU")

# ======= OPTIMIZER =======
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# ======= TRAINING LOOP =======
epoch_num = 100000
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0
save_frq = 2000  # Save model every 2000 iterations

print("🚀 Training started...")

for epoch in range(epoch_num):
    net.train()
    for i, data in enumerate(salobj_dataloader):
        ite_num += 1
        ite_num4val += 1

        inputs, labels = data['image'], data['label']
        inputs, labels = inputs.float(), labels.float()

        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(), requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        optimizer.zero_grad()

        # Forward pass
        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Update loss tracking
        running_loss += loss.item()
        running_tar_loss += loss2.item()

        del d0, d1, d2, d3, d4, d5, d6, loss2, loss  # Free memory

        print(f"[Epoch {epoch+1}, Iter {ite_num}] Loss: {running_loss/ite_num4val:.4f}, Target Loss: {running_tar_loss/ite_num4val:.4f}")

        # Save model periodically
        if ite_num % save_frq == 0:
            torch.save(net.state_dict(), model_save_path)
            print(f"✅ Model saved: {model_save_path}")

            running_loss = 0.0
            running_tar_loss = 0.0
            ite_num4val = 0

print("🎉 Training Completed! Model saved in Google Drive.")