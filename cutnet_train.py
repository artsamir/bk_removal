import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import glob

from data_loader import RescaleT, RandomCrop, ToTensorLab, SalObjDataset
from model import U2NET, U2NETP

# ------- 1. define loss function --------
bce_loss = nn.BCELoss(reduction='mean')  # Use 'mean' for consistency

def muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v):
    loss0 = bce_loss(d0, labels_v)
    loss1 = bce_loss(d1, labels_v)
    loss2 = bce_loss(d2, labels_v)
    loss3 = bce_loss(d3, labels_v)
    loss4 = bce_loss(d4, labels_v)
    loss5 = bce_loss(d5, labels_v)
    loss6 = bce_loss(d6, labels_v)

    loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6
    print("l0: %3f, l1: %3f, l2: %3f, l3: %3f, l4: %3f, l5: %3f, l6: %3f\n" % (
        loss0.item(), loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss5.item(), loss6.item()))

    return loss0, loss

# ------- 2. set the directory of training dataset --------
model_name = 'u2net'  # or 'u2netp'

data_dir = os.path.join(os.getcwd(), 'my_dataset')
tra_image_dir = 'input_images'
tra_label_dir = 'masks'
image_ext = ('.jpg', '.jpeg')
label_ext = '.png'

model_dir = os.path.join(os.getcwd(), 'saved_models', model_name)
os.makedirs(model_dir, exist_ok=True)  # Ensure model directory exists

epoch_num = 100000
batch_size_train = 14
train_num = 0

tra_img_name_list = []
for ext in image_ext:
    tra_img_name_list.extend(glob.glob(os.path.join(data_dir, tra_image_dir, '**', '*' + ext), recursive=True))

tra_lbl_name_list = []
for img_path in tra_img_name_list:
    relative_path = os.path.relpath(img_path, os.path.join(data_dir, tra_image_dir))
    mask_path = os.path.join(data_dir, tra_label_dir, os.path.splitext(relative_path)[0] + label_ext)
    tra_lbl_name_list.append(mask_path)

print("---")
print("train images: ", len(tra_img_name_list))
print("train labels: ", len(tra_lbl_name_list))
print("---")

train_num = len(tra_img_name_list)

salobj_dataset = SalObjDataset(
    img_name_list=tra_img_name_list,
    lbl_name_list=tra_lbl_name_list,
    transform=transforms.Compose([
        RescaleT(320),
        RandomCrop(288),
        ToTensorLab(flag=0)]))
salobj_dataloader = DataLoader(salobj_dataset, batch_size=batch_size_train, shuffle=True, num_workers=4) #increase number of workers.

# ------- 3. define model --------
net = U2NET(3, 1) if model_name == 'u2net' else U2NETP(3, 1)

if torch.cuda.is_available():
    net.cuda()

# ------- 4. define optimizer --------
print("---define optimizer...")
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# ------- 5. training process --------
print("---start training...")
ite_num = 0
running_loss = 0.0
running_tar_loss = 0.0
ite_num4val = 0
save_frq = 2000

for epoch in range(0, epoch_num):
    net.train()

    for i, data in enumerate(salobj_dataloader):
        ite_num += 1
        ite_num4val += 1

        inputs, labels = data['image'], data['label']
        inputs, labels = inputs.type(torch.FloatTensor), labels.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_v, labels_v = Variable(inputs.cuda(), requires_grad=False), Variable(labels.cuda(), requires_grad=False)
        else:
            inputs_v, labels_v = Variable(inputs, requires_grad=False), Variable(labels, requires_grad=False)

        optimizer.zero_grad()

        d0, d1, d2, d3, d4, d5, d6 = net(inputs_v)
        loss2, loss = muti_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_tar_loss += loss2.item()

        del d0, d1, d2, d3, d4, d5, d6, loss2, loss

        print("[epoch: %3d/%3d, batch: %5d/%5d, ite: %d] train loss: %3f, tar: %3f " % (
            epoch + 1, epoch_num, (i + 1) * batch_size_train, train_num, ite_num, running_loss / ite_num4val, running_tar_loss / ite_num4val))

        if ite_num % save_frq == 0:
            torch.save(net.state_dict(), os.path.join(model_dir, f"{model_name}_bce_itr_{ite_num}_train_{running_loss / ite_num4val:.3f}_tar_{running_tar_loss / ite_num4val:.3f}.pth"))
            torch.save(net.state_dict(), "my_train_model.pth")
            print("Model saved as my_train_model.pth")
            running_loss = 0.0
            running_tar_loss = 0.0
            net.train()
            ite_num4val = 0