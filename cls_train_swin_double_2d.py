# -*- coding: utf-8 -*-
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

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import create_test_image_3d, list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
    SpatialPadd,
    Spacingd,
    RandFlipd,
    CropForegroundd,
    CenterSpatialCropd,
    Resized,
    RandZoomd,
)
from monai.visualize import plot_2d_or_3d_image
from model.swin_vit import SWINNet
import csv

from monai.transforms import Activations, AsDiscrete, AdjustContrastd, NormalizeIntensityd
from monai.metrics import ROCAUCMetric
# import resource
import torch.nn.functional as F

from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
import random
import datetime

from model.swin_vit_double_2d import SWINNet_Double_2d

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def main(data_path, current_path, pretrain_model, save_model_name, batch_size, lr_rate, patch_size, is_pretrain):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    train_list = []
    val_list = []
    with open('train_2d.txt', 'r') as ft:
        for r in ft.readlines():
            train_list.append(os.path.join(data_path, r.strip()))
    with open('val_2d.txt', 'r') as fv:
        for r in fv.readlines():
            val_list.append(os.path.join(data_path, r.strip()))

    random.shuffle(train_list)
    random.shuffle(val_list)

    images = []
    segs = []
    labels = []
    file_names = []
    t2f_images = []

    val_images = []
    val_segs = []
    val_labels = []
    val_t2f_images = []

    label_dict = dict()
    csv_reader = csv.reader(open('label_HER2.csv', 'r'))
    for r in csv_reader:
        if r[0] == 'ID':
            continue
        label_dict[r[0] + '_3'] = int(r[1])
        for i in range(10):
            label_dict[r[0] + '_3_' + str(i)] = int(r[1])
        label_dict[r[0] + '_T2F'] = int(r[1])
        label_dict[r[0] + '_3.nii'] = int(r[1])

    for im in train_list:
        if not os.path.exists(os.path.join(data_path, im).replace('Breast_3', 'Breast_masks')):
            print('train mask lost', os.path.join(data_path, im).replace('Breast_3', 'Breast_masks'))
            continue

        if im.split('/')[-2] not in label_dict.keys():
            print('train label lost', im)
            continue

        if not os.path.exists(im.replace('Breast_3', 'Breast_T2F')):
            print('train t2f lost', im)
            continue
        images.append(im)
        segs.append(im.replace('Breast_3', 'Breast_masks'))
        labels.append(label_dict[im.split('/')[-2]])
        file_names.append(im.split('/')[-1])
        t2f_images.append(im.replace('Breast_3', 'Breast_T2F'))

    for im in val_list:

        if im.split('/')[-2] not in label_dict.keys():
            print('val label lost', im)
            continue
        if not os.path.exists(im.replace('Breast_3', 'Breast_masks')):
            print('val mask lost', im.replace('Breast_3', 'Breast_masks'))
            continue
        if not os.path.exists(im.replace('Breast_3', 'Breast_T2F')):
            print('val t2f lost', im)
            continue
        val_images.append(im)
        val_segs.append(im.replace('Breast_3', 'Breast_masks'))
        val_labels.append(label_dict[im.split('/')[-2]])
        val_t2f_images.append(im.replace('Breast_3', 'Breast_T2F'))

    train_files = [{"img": img, "seg": seg, "her2": lab1, "img_name": file_name, "t2f_img": t2f} for
                   img, seg, lab1, file_name, t2f in
                   zip(images, segs, labels, file_names, t2f_images)]
    val_files = [{"img": img, "seg": seg, "her2": lab1, "t2f_img": t2f} for img, seg, lab1, t2f in
                 zip(val_images, val_segs, val_labels, val_t2f_images)]

    # define transforms for image and segmentation
    train_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg", "t2f_img"]),
            EnsureChannelFirstd(keys=["img", "seg", "t2f_img"]),
            Resized(keys=["img", "seg", "t2f_img"], spatial_size=patch_size),
            SpatialPadd(keys=["img", "seg", "t2f_img"],
                        spatial_size=[round(i * 1.0) for i in patch_size],
                        ),
            CenterSpatialCropd(keys=["img", "seg", "t2f_img"], roi_size=patch_size
                               ),
            RandFlipd(keys=["img", "seg", "t2f_img"], prob=0.1, spatial_axis=0),
            RandFlipd(keys=["img", "seg", "t2f_img"], prob=0.1, spatial_axis=1),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg", "t2f_img"]),
            EnsureChannelFirstd(keys=["img", "seg", "t2f_img"]),
            Resized(keys=["img", "seg", "t2f_img"], spatial_size=patch_size),
            CenterSpatialCropd(keys=["img", "seg", "t2f_img"], roi_size=patch_size),
        ]
    )

    # create a training data loader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=10,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=100, num_workers=10, collate_fn=list_data_collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SWINNet_Double_2d(num_classes=3).to(device)

    post_pred = Compose([Activations(softmax=False)])
    post_label = Compose([AsDiscrete(to_onehot=3)])
    auc_metric = ROCAUCMetric('none')

    if is_pretrain:
        model.load_state_dict(torch.load(os.path.join(data_path, pretrain_model)), strict=False)

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr_rate)

    # start a typical PyTorch training
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    writer = SummaryWriter()
    for epoch in range(300):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{300}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels, lab1, img_name, t2f_img = batch_data["img"].to(device), batch_data["seg"].to(device), batch_data[
                "her2"].to(device), batch_data["img_name"], batch_data["t2f_img"].to(device)
            optimizer.zero_grad()

            outputs = model(inputs, t2f_img)
            loss = loss_function(outputs, lab1)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                for val_data in val_loader:
                    val_images, val_labels, val_her2, val_t2f_img = val_data["img"].to(device), val_data["seg"].to(device), \
                                                       val_data["her2"].to(device), val_data["t2f_img"].to(device)
                    # val_images = torch.cat([val_images, val_images, val_images], dim=1)
                    y1 = model(val_images, val_t2f_img).softmax(dim=-1)
                    y_pred = torch.cat([y_pred, y1], dim=0)
                    y = torch.cat([y, val_her2], dim=0)
                acc_value = torch.eq(y_pred.argmax(dim=1), y)
                acc_metric = acc_value.sum().item() / len(acc_value)
                measure_result = classification_report(y.cpu().numpy(),
                                                       y_pred.argmax(dim=1).cpu().numpy())

                y_onehot = [post_label(i) for i in decollate_batch(y, detach=False)]
                y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]
                auc_metric(y_pred_act, y_onehot)
                auc_result = np.mean(auc_metric.aggregate())

                print(measure_result)
                print("auc result: ", auc_metric.aggregate())
                auc_metric.reset()
                del y_pred_act, y_onehot


                torch.save(model.state_dict(),
                           os.path.join(current_path, 'weights', save_model_name + "_epoch_" + str(epoch + 1) + ".pth"))
                print(f"saved model for epoch {epoch + 1}")

                if auc_result > best_metric:
                    best_metric = auc_result
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(),
                               os.path.join(current_path, 'weights', save_model_name + "_" + str(
                                   int(best_metric * 1000)) + ".pth"))
                    print("saved new best metric model")
                print(
                    "current epoch: {} current accuracy: {:.4f} current AUC: {:.4f} best auc: {:.4f} at epoch {}".format(
                        epoch + 1, acc_metric, auc_result, best_metric, best_metric_epoch
                    )
                )

                writer.add_scalar("val_accuracy", acc_metric, auc_result, epoch + 1)

    print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()


if __name__ == "__main__":
    current_path = os.getcwd()
    data_path = os.path.join(current_path, 'data/preprocess_data_2d/')
    pretrain_model = os.path.join(current_path,
                                  'weights/best_metric_classification2d_swin_double_20240828_224_224_resample_nor_flip_epoch_20.pth')
    date = datetime.datetime.now().strftime("%Y%m%d")

    batch_size = 110
    lr_rate = 1e-5
    patch_size = [224, 224]
    save_model_name = 'best_metric_classification2d_swin_double_' + date + '_' + str(patch_size[0]) + '_' + str(
        patch_size[-1]) + '_resample_nor_flip'
    main(data_path, current_path, pretrain_model, save_model_name, batch_size, lr_rate, patch_size, is_pretrain=False)
