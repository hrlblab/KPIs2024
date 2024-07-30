# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import tempfile
from glob import glob

import torch
from PIL import Image

from monai import config
from monai.data import ArrayDataset, create_test_image_2d, decollate_batch, DataLoader
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.networks.nets import UNet
from monai.transforms import Activations, AsDiscrete, Compose, LoadImage, SaveImage, ScaleIntensity, Resize
from matplotlib import cm
import matplotlib.pyplot as plt


def save_validate(val_images, val_labels, val_outputs,output_dir, images, cnt):
    for i in range(val_images.shape[0]):
        folder_list = os.path.dirname(images[cnt+i]).split('/')
        save_folder = os.path.join(output_dir, folder_list[-3], folder_list[-2])

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        now_image = val_images[i].permute([2,1,0]).detach().cpu().numpy()
        now_label = val_labels[i][0].permute([1,0]).detach().cpu().numpy()
        now_pred = val_outputs[i][0].permute([1,0]).detach().cpu().numpy()
        name = os.path.basename(images[cnt+i])
        plt.imsave(os.path.join(save_folder, 'val_%s_img.png' % (name)), now_image)
        plt.imsave(os.path.join(save_folder, 'val_%s_lbl.png' % (name)), now_label, cmap = cm.gray)
        plt.imsave(os.path.join(save_folder, 'val_%s_pred.png' % (name)), now_pred, cmap = cm.gray)

    cnt += val_images.shape[0]
    return cnt


def main(tempdir, model_dir, output_dir):
    config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    image = []
    seg = []
    types = glob(os.path.join(tempdir, '*'))
    for type in types:
        cases = glob(os.path.join(type, '*'))
        for case in cases:
            now_imgs = glob(os.path.join(case, 'img', '*img.jpg'))
            image.extend(now_imgs)
            now_lbls = glob(os.path.join(case, 'mask', '*mask.jpg'))
            seg.extend(now_lbls)

    images = sorted(image)
    segs = sorted(seg)

    print('total image: %d' % (len(images)))

    # define transforms for image and segmentation
    imtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), ScaleIntensity(), Resize(spatial_size=(512, 512), mode='nearest')])
    segtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), ScaleIntensity()])
    outputrans = Compose([Resize(spatial_size=(2048, 2048), mode='nearest')])
    val_ds = ArrayDataset(images, imtrans, segs, segtrans)
    # sliding window inference for one image at every iteration
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=1, pin_memory=torch.cuda.is_available())
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
    saver = SaveImage(output_dir=os.path.join(output_dir, "./validation_output"), output_ext=".png", output_postfix="seg")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )

    model.load_state_dict(torch.load(glob(os.path.join(model_dir,'*.pth'))[0], map_location=torch.device('cpu')))
    model.to(device)
    model.eval()
    with torch.no_grad():
        cnt = 0
        for val_data in val_loader:
            val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
            # define sliding window size and batch size for windows inference
            roi_size = (512, 512)
            sw_batch_size = 4
            val_outputs = sliding_window_inference(val_images, roi_size, sw_batch_size, model)
            val_outputs = [post_trans(i) for i in decollate_batch(val_outputs)]
            val_labels = decollate_batch(val_labels)
            # compute metric for current iteration
            val_images = outputrans(val_images[0]).unsqueeze(0)
            val_outputs = outputrans(val_outputs)
            dice_metric(y_pred=val_outputs, y=val_labels)
            # for val_output in val_outputs:
            #     saver(val_output)
            cnt = save_validate(val_images, val_labels, val_outputs, output_dir, images, cnt)

        # aggregate the final mean dice result
        print("evaluation metric:", dice_metric.aggregate().item())
        # reset the status
        dice_metric.reset()


if __name__ == "__main__":
    # with tempfile.TemporaryDirectory() as tempdir:
    input_dir = '/input/'
    output_dir = '/output/'
    wd = '/myhome/wd'
    model_dir = '/model/'

    # input_dir = '/Data/KPIs/data_train'
    # model_dir = '/Data/KPIs/checkpoint'
    # output_dir = '/Data/KPIs/validation_patch'
    main(input_dir, model_dir, output_dir)
