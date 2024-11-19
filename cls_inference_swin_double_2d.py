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
from monai.transforms import (Activations, EnsureChannelFirstd, AsDiscrete, Compose, LoadImaged, ScaleIntensityd,
                              SpatialPadd, Spacingd, RandFlipd, CropForegroundd, CenterSpatialCropd, Resized, RandZoomd,
                              )
from monai.visualize import plot_2d_or_3d_image
from swin_transformer_pytorch import SwinTransformer
import csv

from monai.transforms import Activations, AsDiscrete, AdjustContrastd, NormalizeIntensityd
from monai.metrics import ROCAUCMetric
# import resource
import torch.nn.functional as F

from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import random
from model.swin_vit_double_2d import SWINNet_Double_2d

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def main(img_dir, pretrain_model_path, patch_size, test_txt, label_csv, result_csv_writer):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)


    metrics_csv_path = os.path.join('results', 'classification_metrics.csv')
    with open(metrics_csv_path, mode='w', newline='') as metrics_csv_file:
        metrics_csv_writer = csv.writer(metrics_csv_file)

        metrics_csv_writer.writerow(
            ['class', 'accuracy', 'sensitivity', 'specificity', 'precision', 'NPV', 'PPV', 'f1_score'])

    test_list = []
    with open(test_txt, 'r') as fv:
        for r in fv.readlines():
            test_list.append(os.path.join(img_dir, r.strip()))

    val_images = []
    val_segs = []
    val_labels = []
    val_file_names = []
    val_t2f_images = []

    label_dict = dict()
    csv_reader = csv.reader(open(label_csv, 'r'))
    for r in csv_reader:
        if r[0] == 'ID':
            continue
        label_dict[r[0] + '_3'] = int(r[1])
        for i in range(10):
            label_dict[r[0] + '_3_' + str(i)] = int(r[1])
        label_dict[r[0] + '_T2F'] = int(r[1])

    for im in test_list:

        # print(f"Processing file: {im}")

        if im.split('/')[-2] not in label_dict.keys():
            print('val label lost', im)
            continue
        if not os.path.exists(im.replace('Breast_3', 'Breast_masks')):
            print('val mask lost', im.replace('Breast_3', 'Breast_masks'))
            continue



        val_images.append(im)
        val_segs.append(im.replace('Breast_3', 'Breast_masks'))
        val_labels.append(label_dict[im.split('/')[-2]])
        val_file_names.append(im)
        val_t2f_images.append(im.replace('Breast_3', 'Breast_T2F'))


    val_files = [{"img": img, "seg": seg, "her2": lab1, "img_name": file_name, "t2f_img": t2f} for
                 img, seg, lab1, file_name, t2f in
                 zip(val_images, val_segs, val_labels, val_file_names, val_t2f_images)]

    # 如果需要，也可以在for循环外面打印出最终的val_files长度
    print(f"Total number of files processed: {len(val_files)}")

    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg", "t2f_img"]),
            EnsureChannelFirstd(keys=["img", "seg", "t2f_img"]),
            Resized(keys=["img", "seg", "t2f_img"], spatial_size=patch_size),
            CenterSpatialCropd(keys=["img", "seg", "t2f_img"], roi_size=patch_size),
        ]
    )

    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    print(f"Number of items in dataset: {len(val_ds)}")  # 检查数据集长度

    val_loader = DataLoader(val_ds, batch_size=10, num_workers=10, collate_fn=list_data_collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SWINNet_Double_2d(num_classes=3).to(device)

    post_pred = Compose([Activations(softmax=False)])
    post_label = Compose([AsDiscrete(to_onehot=3)])
    auc_metric = ROCAUCMetric('none')

    model.load_state_dict(torch.load(pretrain_model_path), strict=False)

    model.eval()
    with torch.no_grad():
        y_pred = torch.tensor([], dtype=torch.float32, device=device)
        y = torch.tensor([], dtype=torch.long, device=device)
        for val_data in val_loader:
            val_images, val_labels, val_her2, img_name, val_t2f_img = val_data["img"].to(device), val_data["seg"].to(
                device), val_data["her2"].to(device), val_data["img_name"], val_data["t2f_img"].to(device)

            print(f"Processing file: {img_name[0]}")  # 打印当前处理的文件

            y1 = model(val_images, val_t2f_img).softmax(dim=-1)
            y_pred = torch.cat([y_pred, y1], dim=0)
            y = torch.cat([y, val_her2], dim=0)
            y_pred_label = y1.argmax(dim=1).cpu().numpy()[0]

            result_csv_writer.writerow(
                [img_name[0].split('/')[-1].replace('.nii.gz', ''), y1[0][0].cpu().numpy(), y1[0][1].cpu().numpy(),
                 y1[0][2].cpu().numpy(), y_pred_label, val_her2[0].cpu().numpy()])


        acc_value = torch.eq(y_pred.argmax(dim=1), y)
        acc_metric = acc_value.sum().item() / len(acc_value)
        measure_result = classification_report(y.cpu().numpy(), y_pred.argmax(dim=1).cpu().numpy())
        y_onehot = [post_label(i) for i in decollate_batch(y, detach=False)]
        y_pred_act = [post_pred(i) for i in decollate_batch(y_pred)]
        auc_metric(y_pred_act, y_onehot)
        auc_result = np.mean(auc_metric.aggregate())
        print(measure_result)
        print('auc 0:', auc_metric.aggregate()[0])
        print('auc 1:', auc_metric.aggregate()[1])
        print('auc 2:', auc_metric.aggregate()[2])
        print('auc average:', auc_result)
        print('accuracy:', acc_metric)


        y_true = y.cpu().numpy()
        y_pred_labels = y_pred.argmax(dim=1).cpu().numpy()

        conf_matrix = confusion_matrix(y_true, y_pred_labels)
        metrics = {'class_0': {}, 'class_1': {}, 'class_2': {}}

        for i in range(3):
            TP = conf_matrix[i, i]
            FP = conf_matrix[:, i].sum() - TP
            FN = conf_matrix[i, :].sum() - TP
            TN = conf_matrix.sum() - (FP + FN + TP)

            accuracy = (TP + TN) / (TP + TN + FP + FN)
            sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            npv = TN / (TN + FN) if (TN + FN) > 0 else 0
            ppv = precision
            f1 = f1_score(y_true, y_pred_labels, labels=[i], average='macro')

            metrics[f'class_{i}'] = {
                'accuracy': accuracy,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'precision': precision,
                'NPV': npv,
                'PPV': ppv,
                'f1_score': f1,
            }

            # 将指标写入CSV文件
            with open(metrics_csv_path, mode='a', newline='') as metrics_csv_file:
                metrics_csv_writer = csv.writer(metrics_csv_file)
                metrics_csv_writer.writerow([f'class_{i}', accuracy, sensitivity, specificity, precision, npv, ppv, f1])

        # 打印每类的评估指标
        for class_label, metric_values in metrics.items():
            print(f"Metrics for {class_label}:")
            for metric_name, value in metric_values.items():
                print(f"  {metric_name}: {value:.4f}")
            print()


        auc_metric.reset()
        del y_pred_act, y_onehot



if __name__ == "__main__":
    current_path = os.getcwd()
    data_path = os.path.join(current_path, 'data/preprocess_data_2d/')
    weight_path = os.path.join(current_path,
                               "weights/best_metric_classification2d_swin_double_20241009_224_224_resample_nor_flip_epoch_272.pth")
    patch_size = [224, 224]
    test_txt = os.path.join(current_path, 'val_2d.txt')
    label_csv = 'label_HER2.csv'
    result_csv_writer = csv.writer(open('results/result_' + weight_path.split('/')[-1].replace('.pth', '.csv'), 'w'))
    result_csv_writer.writerow(['file_name', 'score_0', 'score_1', 'score_2', 'pred', 'gt'])
    main(data_path, weight_path, patch_size, test_txt, label_csv, result_csv_writer)
