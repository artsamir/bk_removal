import os
from skimage import io, transform
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from google.colab import drive

drive.mount('/content/drive')

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT, ToTensorLab, SalObjDataset
from cutnet_model import CUTNET, CUTNETP

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn

def save_output(image_name, pred, d_dir):
    predict = pred.squeeze().cpu().data.numpy()
    im = Image.fromarray((predict * 255).astype(np.uint8)).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)
    imo.save(os.path.join(d_dir, img_name.split(".")[0] + '.png'))

def main():
    model_name = 'U2NET'
    image_dir = os.path.join(os.getcwd(), 'test_data', 'test_ots_images')
    prediction_dir = os.path.join(os.getcwd(), 'test_data', model_name + '_results' + os.sep)
    model_dir = "/content/drive/MyDrive/cutnet_models/saved_models/"  # Corrected: Directory path
    model_path = os.path.join(model_dir, "u2net.pth") #Added model_path.

    print(f"Model directory: {model_path}") #print model_path.

    if not os.path.exists(model_path): #check model_path
        print(f"Error: {model_path} not found.")
        return

    net = CUTNET(3, 1) if model_name == 'U2NET' else CUTNETP(3, 1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path)) #load model_path
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location='cpu')) #load model_path

    net.eval()

    img_name_list = glob.glob(os.path.join(image_dir, '*'))
    print(img_name_list)

    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list, lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)]))
    test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=1)

    for i_test, data_test in enumerate(test_salobj_dataloader):
        print("Inferencing:", img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image'].type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, d2, d3, d4, d5, d6, d7 = net(inputs_test)

        pred = normPRED(d1[:, 0, :, :])

        if not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir, exist_ok=True)
        save_output(img_name_list[i_test], pred, prediction_dir)

        del d1, d2, d3, d4, d5, d6, d7

if __name__ == "__main__":
    main()