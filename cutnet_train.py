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

# Define Binary Cross-Entropy Loss
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
    
    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    
    print("Loss breakdown: ", [loss0.item(), loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item(), loss6.item()])
    return loss0, loss

# Set dataset directories
model_name = 'cutnet'  # Choose between 'u2net' and 'u2netp'

data_dir = os.path.join(os.getcwd(), 'my_dataset')
tra_image_dir = 'input_images'
tra_label_dir = 'masks_images'
image_ext = ('.jpg', '.jpeg')
label_ext = '.png'

# Set model save directory inside bk_removal/saved_models/
model_dir = os.path.join(os.getcwd(), 'bk_removal', 'saved_models', model_name)
os.makedirs(model_dir, exist_ok=True)  # Ensure directory exists

# Training parameters
epoch_num = 100000
batch_size_train = 2 #14
train_num = 0

# Load training images
tra_img_name_list = [img for ext in image_ext for img in glob.glob(os.path.join(data_dir, tra_image_dir, '**', '*' + ext), recursive=True)]

# Load corresponding masks
tra_lbl_name_list = [
    os.path.join(data_dir, tra_label_dir, os.path.splitext(os.path.relpath(img, os.path.join(data_dir, tra_image_dir)))[0] + label_ext)
    for img in tra_img_name_list
]

print("---")
print(f"Train images: {len(tra_img_name_list)}")
print(f"Train labels: {len(tra_lbl_name_list)}")
print("---")

train_num = len(tra_img_name_list)

# Define dataset and dataloader
salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms.Compose([
        RescaleT(320),
        RandomCrop(288),
        ToTensorLab(flag=0)]))

salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=4)

# Initialize model
net = CUTNET(3, 1) if model_name == 'u2net' else CUTNETP(3, 1)
if torch.cuda.is_available():
    net.cuda()

# Define optimizer
print("--- Defining optimizer...")
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# Training loop
print("--- Start training...")
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0
save_frq = 2000  # Save model every 2000 iterations

for epoch in range(epoch_num):
    net.train()
    for i, data in enumerate(salobj_dataloader):
        ite_num += 1
        ite_num4val += 1

        # Load data
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

        print(f"[Epoch {epoch+1}/{epoch_num}, Batch {i+1}/{train_num}, Iter {ite_num}] Train Loss: {running_loss/ite_num4val:.4f}, Target Loss: {running_tar_loss/ite_num4val:.4f}")

        # Save model periodically
        if ite_num % save_frq == 0:
            model_save_path = os.path.join(model_dir, f"{model_name}_bce_itr_{ite_num}_train_{running_loss/ite_num4val:.3f}_tar_{running_tar_loss/ite_num4val:.3f}.pth")
            torch.save(net.state_dict(), model_save_path)
            torch.save(net.state_dict(), os.path.join(model_dir, "my_train_model.pth"))  # Always save latest model
            print(f"Model saved inside '{model_dir}' as my_train_model.pth")

            running_loss = 0.0
            running_tar_loss = 0.0
            ite_num4val = 0
