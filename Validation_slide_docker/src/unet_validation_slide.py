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
import cv2
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
import tifffile
import scipy.ndimage as ndi
import imageio
import numpy as np
import pandas as pd

def calculate_contour_iou(contour1, contour2, image_shape):
    x1_min = contour1[:,:,0].min()
    x1_max = contour1[:,:,0].max()
    y1_min = contour1[:,:,1].min()
    y1_max = contour1[:,:,1].max()

    x2_min = contour2[:,:,0].min()
    x2_max = contour2[:,:,0].max()
    y2_min = contour2[:,:,1].min()
    y2_max = contour2[:,:,1].max()

    if x1_max < x2_min or x2_max < x1_min:
        return 0
    if y1_max < y2_min or y2_max < y1_min:
        return 0

    'crop'
    x_min = np.min([x1_min, x2_min]) - 10
    y_min = np.min([y1_min, y2_min]) - 10
    x_max = np.max([x1_max, x2_max]) + 10
    y_max = np.max([y1_max, y2_max]) + 10

    contour1[:,:,0] = contour1[:,:,0] - x_min
    contour1[:,:,1] = contour1[:,:,1] - y_min

    contour2[:,:,0] = contour2[:,:,0] - x_min
    contour2[:,:,1] = contour2[:,:,1] - y_min
    image_shape = (y_max - y_min, x_max - x_min)

    mask1 = np.zeros(image_shape, dtype=np.uint8)
    mask2 = np.zeros(image_shape, dtype=np.uint8)
    cv2.drawContours(mask1, [contour1], -1, 1, -1)
    cv2.drawContours(mask2, [contour2], -1, 1, -1)
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union != 0 else 0


def save_validate(val_images, val_labels, val_outputs, output_dir, images, cnt):
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

def calculate_f1(precision, recall):
    f1_scores = 2 * (precision * recall) / (precision + recall)
    f1_scores = np.nan_to_num(f1_scores)  # Convert NaNs to zero if precision and recall are both zero
    return f1_scores


def dice_coefficient(mask1, mask2):
    # Convert masks to boolean arrays
    mask1 = np.asarray(mask1).astype(np.int8)
    mask2 = np.asarray(mask2).astype(np.int8)

    # Calculate intersection and union
    intersection = np.logical_and(mask1, mask2)
    mask1_sum = np.sum(mask1)
    mask2_sum = np.sum(mask2)
    intersection_sum = np.sum(intersection)

    # Compute Dice coefficient
    if mask1_sum + mask2_sum == 0:  # To handle division by zero if both masks are empty
        return 1.0
    else:
        return 2 * intersection_sum / (mask1_sum + mask2_sum)


def calculate_metrics_ap50(pred_contours_list, gt_contours_list, image_shape, iou_thresholds=[0.5]):
    # Initialize lists to hold precision and recall values for each threshold
    precision_scores = []
    recall_scores = []

    for threshold in iou_thresholds:
        tp = 0
        fp = 0
        fn = 0

        # Calculate matches for predictions
        for pred_contours in pred_contours_list:
            match_found = False
            for gt_contours in gt_contours_list:
                if calculate_contour_iou(pred_contours, gt_contours, image_shape) >= threshold:
                    tp += 1
                    match_found = True
                    break
            if not match_found:
                fp += 1

        # Calculate false negatives
        for gt_contours in gt_contours_list:
            if not any(calculate_contour_iou(pred_contours, gt_contours, image_shape) >= threshold for pred_contours in pred_contours_list):
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision_scores.append(precision)
        recall_scores.append(recall)

    # Compute F1 scores
    f1_scores = [2 * p * r / (p + r) if (p + r) > 0 else 0 for p, r in zip(precision_scores, recall_scores)]
    return precision_scores, recall_scores, f1_scores


def sodelete(wsi, min_size):
    """
    Remove objects smaller than min_size from binary segmentation image.

    Args:
    img (numpy.ndarray): Binary image where objects are 255 and background is 0.
    min_size (int): Minimum size of the object to keep.

    Returns:
    numpy.ndarray: Image with small objects removed.
    """
    # Find all connected components (using 8-connectivity, as default)
    _, binary = cv2.threshold(wsi* 255, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary.astype(np.uint8), 8, cv2.CV_32S)

    # Create an output image that will store the filtered objects
    # output = np.zeros_like(wsi, dtype=np.uint8)
    output = np.zeros_like(wsi)

    # Loop through all found components
    for i in range(1, num_labels):  # start from 1 to ignore the background
        size = stats[i, cv2.CC_STAT_AREA]

        # If the size of the component is larger than the threshold, copy it to output
        if size >= min_size:
            output[labels == i] = 1.

    return output


def calculate_ap50(precisions, recalls):
    # Ensure that the arrays are sorted by recall
    sort_order = np.argsort(recalls)
    precisions = np.array(precisions)[sort_order]
    recalls = np.array(recalls)[sort_order]

    # Pad precisions array to include the precision at recall zero
    precisions = np.concatenate(([0], precisions))
    recalls = np.concatenate(([0], recalls))

    # Calculate the differences in recall to use as weights in weighted average
    recall_diff = np.diff(recalls)

    # Compute the average precision
    ap50 = np.sum(precisions[:-1] * recall_diff)
    return ap50


def main(tempdir, model_dir, output_dir, X20_dir, X20_patch_dir, get_X20_wsi, get_X20_patch_preds, get_X20_wsi_preds, df):
    config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    image = []
    seg = []
    types = glob(os.path.join(tempdir, '*'))
    for type in types:
        now_imgs = glob(os.path.join(type, 'img', '*.tiff'))
        image.extend(now_imgs)
        now_lbls = glob(os.path.join(type, 'mask', '*mask.tiff'))
        seg.extend(now_lbls)

    images = sorted(image)
    segs = sorted(seg)

    print('total image: %d' % (len(images)))

    if get_X20_wsi:

        'read wsi and get 20X and patches'
        for img in images:
            if 'NEP25' in img:
                lv = 1
            else:
                lv = 2

            now_tiff = tifffile.imread(img,key=0)
            tiff_X20 = ndi.zoom(now_tiff, (1/lv, 1/lv, 1), order=1)

            if not os.path.exists(os.path.dirname(img.replace(tempdir, X20_dir))):
                os.makedirs(os.path.dirname(img.replace(tempdir, X20_dir)))

            imageio.imwrite(img.replace(tempdir, X20_dir).replace('.tiff', '.png'), tiff_X20)

            wsi_shape = tiff_X20.shape
            patch_size = 2048
            stride = 1024
            x_slide = int((wsi_shape[0] - patch_size) / stride) + 1
            y_slide = int((wsi_shape[1] - patch_size) / stride) + 1


            cnt = 1
            for xi in range(x_slide):
                for yi in range(y_slide):
                    if xi == x_slide - 1:
                        now_x = wsi_shape[0] - patch_size
                    else:
                        now_x = xi * stride
                    if yi == y_slide - 1:
                        now_y = wsi_shape[1] - patch_size
                    else:
                        now_y = yi * stride

                    now_patch = tiff_X20[now_x:now_x + patch_size, now_y:now_y + patch_size, :]
                    assert now_patch.shape == (patch_size, patch_size, 3)

                    root_list = os.path.dirname(img).split('/')
                    now_folder = os.path.join(X20_patch_dir,  root_list[-2], os.path.basename(img).replace('_wsi.tiff',''), root_list[-1])
                    if not os.path.exists(now_folder):
                        os.makedirs(now_folder)
                    imageio.imwrite(os.path.join(now_folder, '%s_%s_%d_%d_%d_img.png' % (root_list[-2], root_list[-1], cnt, now_x, now_y)), now_patch)
                    cnt+=1

        for seg in segs:
            if 'NEP25' in seg:
                lv = 1
            else:
                lv = 2
            now_tiff = tifffile.imread(seg, key=0)
            tiff_X20 = ndi.zoom(now_tiff, (1 / lv, 1 / lv), order=1)

            if not os.path.exists(os.path.dirname(seg.replace(tempdir, X20_dir))):
                os.makedirs(os.path.dirname(seg.replace(tempdir, X20_dir)))
            imageio.imwrite(seg.replace(tempdir, X20_dir).replace('.tiff', '.png'), tiff_X20)

            wsi_shape = tiff_X20.shape
            patch_size = 2048
            stride = 1024
            x_slide = int((wsi_shape[0] - patch_size) / stride) + 1
            y_slide = int((wsi_shape[1] - patch_size) / stride) + 1

            cnt = 1
            for xi in range(x_slide):
                for yi in range(y_slide):
                    if xi == x_slide - 1:
                        now_x = wsi_shape[0] - patch_size
                    else:
                        now_x = xi * stride
                    if yi == y_slide - 1:
                        now_y = wsi_shape[1] - patch_size
                    else:
                        now_y = yi * stride

                    now_patch = tiff_X20[now_x:now_x + patch_size, now_y:now_y + patch_size]
                    assert now_patch.shape == (patch_size, patch_size)

                    root_list = os.path.dirname(seg).split('/')
                    now_folder = os.path.join(X20_patch_dir, root_list[-2], os.path.basename(seg).replace('_mask.tiff', ''),root_list[-1])
                    if not os.path.exists(now_folder):
                        os.makedirs(now_folder)
                    imageio.imwrite(
                        os.path.join(now_folder, '%s_%s_%d_%d_%d_mask.png' % (root_list[-2], root_list[-1], cnt, now_x, now_y)),
                        now_patch)
                    cnt+=1

    if get_X20_patch_preds:
        image = []
        seg = []
        types = glob(os.path.join(X20_patch_dir, '*'))
        for type in types:
            cases = glob(os.path.join(type, '*'))
            for case in cases:
                now_imgs = glob(os.path.join(case, 'img', '*img.png'))
                image.extend(now_imgs)
                now_lbls = glob(os.path.join(case, 'mask', '*mask.png'))
                seg.extend(now_lbls)

        images = sorted(image)
        segs = sorted(seg)

        print('total image: %d' % (len(images)))

        # define transforms for image and segmentation
        imtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), Resize(spatial_size=(512, 512), mode='nearest'), ScaleIntensity()])
        segtrans = Compose([LoadImage(image_only=True, ensure_channel_first=True), ScaleIntensity()])
        outputrans = Compose([Resize(spatial_size=(2048, 2048), mode='nearest')])
        val_ds = ArrayDataset(images, imtrans, segs, segtrans)
        # sliding window inference for one image at every iteration
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=0, pin_memory=torch.cuda.is_available())
        dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=1,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
        )

        model.load_state_dict(torch.load(glob(os.path.join(model_dir, '*.pth'))[0], map_location=torch.device('cpu')))
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
                cnt = save_validate(val_images, val_labels, val_outputs, output_dir, images, cnt)


    if get_X20_wsi_preds:
        image = []
        seg = []
        types = glob(os.path.join(X20_dir, '*'))
        for type in types:
            now_imgs = glob(os.path.join(type, 'img', '*.png'))
            image.extend(now_imgs)
            now_lbls = glob(os.path.join(type, 'mask', '*mask.png'))
            seg.extend(now_lbls)

        images = sorted(image)
        segs = sorted(seg)
        patch_size = 2048


        wsi_F1_50 = []
        wsi_AP50 = []
        wsi_dice = []
        for img in images:


            case_name = os.path.basename(img)
            now_img_shape = plt.imread(img).shape
            wsi_prediction = np.zeros((now_img_shape[0],now_img_shape[1]))

            folder_list = img.split('/')

            patch_folder = os.path.join(output_dir, folder_list[-3], folder_list[-1].replace('_wsi.png',''))

            patches = glob(os.path.join(patch_folder, '*_pred.png'))

            for patch in patches:
                now_name = os.path.basename(patch).split('_')
                now_x = int(now_name[-4])
                now_y = int(now_name[-3])
                now_patch = plt.imread(patch)
                wsi_prediction[now_x:now_x+patch_size, now_y:now_y+patch_size] = wsi_prediction[now_x:now_x+patch_size, now_y:now_y+patch_size] + now_patch[:,:,0]


            preds_folder = os.path.dirname(img).replace('img','pred')
            if not os.path.exists(preds_folder):
                os.makedirs(preds_folder)

            wsi_prediction[wsi_prediction <= 1] = 0
            wsi_prediction[wsi_prediction != 0] = 1

            sm = 20000
            wsi_prediction_sm = sodelete(wsi_prediction, sm)

            preds_root = os.path.join(preds_folder, folder_list[-1].replace('_wsi','_pred_final'))
            plt.imsave(preds_root, wsi_prediction_sm, cmap=cm.gray)

            mask_root = img.replace('/img/','/mask/').replace('_wsi.png','_mask.png')
            wsi_mask = plt.imread(mask_root)


            'f1'
            ret, binary = cv2.threshold(wsi_prediction_sm, 0, 255, cv2.THRESH_BINARY_INV)
            _, preds_contours, _ = cv2.findContours(binary.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

            ret, binary = cv2.threshold(wsi_mask, 0, 255, cv2.THRESH_BINARY_INV)
            _, masks_contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            precision_scores, recall_scores, f1_scores_50 = calculate_metrics_ap50(preds_contours[1:], masks_contours[1:],(now_img_shape[0], now_img_shape[1]))
            ap50 = precision_scores


            wsi_F1_50.append((f1_scores_50[0]))
            wsi_AP50.append((ap50[0]))
            wsi_dice.append((dice_coefficient(wsi_prediction_sm, wsi_mask)))

            row = len(df)
            df.loc[row] = [case_name, dice_coefficient(wsi_prediction_sm, wsi_mask), f1_scores_50[0], ap50[0]]
            df.to_csv(os.path.join(X20_dir,'testing_wsi_results_all.csv'), index=False)

        print("slide level F1 metric:", np.mean(wsi_F1_50))
        print("slide level AP(50) metric:", np.mean(wsi_AP50))
        print("slide level Dice metric:", np.mean(wsi_dice))

        return df

if __name__ == "__main__":
    # with tempfile.TemporaryDirectory() as tempdir:
    data_dir = '/input_slide/'
    output_dir = '/output_slide/'
    patch_data_dir =  '/input_patch/'
    model_dir = '/model/'
    patch_output_dir = '/output_patch/'
    wd = '/myhome/wd'

    # data_dir = '/Data/KPIs/data_val_slide'
    # output_dir = '/Data/KPIs/validation_slide_20X'
    # patch_data_dir = '/Data/KPIs/testing_data_wsi_patch_20X'
    # model_dir = '/Data/KPIs/checkpoint'
    # patch_output_dir = '/Data/KPIs/validation_slide_20X_patchoutput'

    get_X20_wsi = 1
    get_X20_patch_preds = 1
    get_X20_wsi_preds = 1

    df = pd.DataFrame(columns=['case name','wsi_dice','wsi_F1_50','wsi_AP50'])
    main(data_dir, model_dir, patch_output_dir, output_dir, patch_data_dir, get_X20_wsi, get_X20_patch_preds, get_X20_wsi_preds, df)
