from config_rcnn import ORIGINAL_DATA_IMG_DIR, ORIGINAL_DATA_ANNO_DIR, ORIGINAL_DATA_SUBDIRS
from preprocess_rcnn_utils import divide_train_val, get_transform, RCnnDataset

import sys
import math

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
# from tqdm import tqdm
from tqdm.notebook import tqdm


def collate_fn(batch):
    return tuple(zip(*batch))




def train_one_epoch(model, optimizer, lr_scheduler, train_loader, val_loader, device, epoch):
    model.to(device)
    model.train()

    all_losses = []
    all_losses_dict = []

    for images, targets in tqdm(train_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_dict_append = {k: v.item() for k, v in loss_dict.items()}
        loss_value = losses.item()

        all_losses.append(loss_value)
        all_losses_dict.append(loss_dict_append)

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping trainig")
            print(loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        # validation
        model.eval()
        with torch.no_grad():
            tk = tqdm(val_loader)
            for images, targets in tk:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                val_output = model(images)
                val_output = [{k: v.to(device) for k, v in t.items()} for t in val_output]
                IOU = []
                for j in range(len(val_output)):
                    a, b = val_output[j]['boxes'].cpu().detach(), targets[j]['boxes'].cpu().detach()
                    chk = torchvision.ops.box_iou(a, b)
                    res = np.nanmean(chk.sum(axis=1) / (chk > 0).sum(axis=1))
                    IOU.append(res)
                tk.set_postfix(IoU=np.mean(IOU))
            tk.close()


    all_losses_dict = pd.DataFrame(all_losses_dict)  # for printing
    print("Epoch {}, lr: {:.6f}, loss: {:.6f}, loss_classifier: {:.6f}, loss_box: {:.6f},"
          " loss_rpn_box: {:.6f}, loss_object: {:.6f}".format(
        epoch,
        optimizer.param_groups[0]['lr'],
        np.mean(all_losses),
        all_losses_dict['loss_classifier'].mean(),
        all_losses_dict['loss_box_reg'].mean(),
        all_losses_dict['loss_rpn_box_reg'].mean(),
        all_losses_dict['loss_objectness'].mean()
    ))


def main():
    img_dir = ORIGINAL_DATA_IMG_DIR
    anno_dir = ORIGINAL_DATA_ANNO_DIR
    subdirs = ORIGINAL_DATA_SUBDIRS

    train_size = 0.8

    paths = divide_train_val(img_dir, anno_dir, subdirs, train_size)
    img_train_paths, img_val_paths, img_test_paths, anno_train_paths, anno_val_paths, anno_test_paths = paths

    train_dataset = RCnnDataset(image_paths=img_train_paths,
                                anno_paths=anno_train_paths)
    valid_dataset = RCnnDataset(img_val_paths,
                                anno_val_paths)
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=1,
                                   shuffle=False,
                                   num_workers=2,
                                   collate_fn=collate_fn)

    valid_data_loader = DataLoader(valid_dataset,
                                   batch_size=1,
                                   shuffle=False,
                                   num_workers=2,
                                   collate_fn=collate_fn)

    num_classes = 6
    num_epochs = 100
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.DEFAULT')
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.001, weight_decay=0.0005, )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, lr_scheduler, train_data_loader, valid_data_loader,  device, epoch)


if __name__ == '__main__':
    main()
