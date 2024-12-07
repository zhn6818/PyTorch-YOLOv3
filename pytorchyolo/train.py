#! /usr/bin/env python3

from __future__ import division

import os
import argparse
import tqdm

import torch
# 检查是否为寒武纪设备
try:
    import torch_mlu
    HAS_MLU = True
except ImportError:
    HAS_MLU = False
    
from torch.utils.data import DataLoader
import torch.optim as optim
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pytorchyolo.models import load_model
from pytorchyolo.utils.logger import Logger
from pytorchyolo.utils.utils import to_cpu, load_classes, print_environment_info, provide_determinism, worker_seed_set
from pytorchyolo.utils.datasets import ListDataset
from pytorchyolo.utils.augmentations import AUGMENTATION_TRANSFORMS
#from pytorchyolo.utils.transforms import DEFAULT_TRANSFORMS
from pytorchyolo.utils.parse_config import parse_data_config
from pytorchyolo.utils.loss import compute_loss
from pytorchyolo.test import _evaluate, _create_validation_data_loader

from terminaltables import AsciiTable

from torchsummary import summary


def _create_data_loader(img_path, batch_size, img_size, n_cpu, multiscale_training=False):
    """Creates a DataLoader for training.

    :param img_path: Path to file containing all paths to training images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :param multiscale_training: Scale images to different sizes randomly
    :type multiscale_training: bool
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    dataset = ListDataset(
        img_path,
        img_size=img_size,
        multiscale=multiscale_training,
        transform=AUGMENTATION_TRANSFORMS)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        worker_init_fn=worker_seed_set)
    return dataloader


def run():
    print_environment_info()
    parser = argparse.ArgumentParser(description="Trains the YOLO model.")
    parser.add_argument("-m", "--model", type=str, default="config/yolov3.cfg", help="Path to model definition file (.cfg)")
    parser.add_argument("-d", "--data", type=str, default="config/coco.data", help="Path to data config file (.data)")
    parser.add_argument("-e", "--epochs", type=int, default=300, help="Number of epochs")
    parser.add_argument("-v", "--verbose", action='store_true', help="Makes the training more verbose")
    parser.add_argument("--n_cpu", type=int, default=1, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--pretrained_weights", type=str, help="Path to checkpoint file (.weights or .pth). Starts training from checkpoint model")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="Interval of epochs between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=1, help="Interval of epochs between evaluations on validation set")
    parser.add_argument("--multiscale_training", action="store_true", help="Allow multi-scale training")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="Evaluation: IOU threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.1, help="Evaluation: Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="Evaluation: IOU threshold for non-maximum suppression")
    parser.add_argument("--logdir", type=str, default="logs", help="Directory for training log files (e.g. for TensorBoard)")
    parser.add_argument("--seed", type=int, default=-1, help="Makes results reproducable. Set -1 to disable.")
    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    if args.seed != -1:
        provide_determinism(args.seed)

    logger = Logger(args.logdir)

    # Create output directories if missing
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(args.data)
    
    # # 根据设备类型选择不同的数据集路径
    # if torch.backends.mps.is_available():
    #     train_path = "data/coco/trainvalno5k.part"
    #     valid_path = "data/coco/5k.part"
    #     print("Using original dataset paths for MPS device")
    # else:
    #     train_path = "data/coco/trainvalno5k_copy.part"
    #     valid_path = "data/coco/5k_copy.part"
    #     print("Using copy dataset paths for non-MPS device")
    
    # 使用新的路径覆盖配置文件中的路径
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    # data_config["train"] = data_config.train
    # data_config["valid"] = data_config.valid
    
    class_names = load_classes(data_config["names"])

    # 修改设备选择逻辑
    if HAS_MLU and torch.mlu.is_available():
        device = torch.device("mlu:0")
        print("使用寒武纪 MLU 设备")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("使用 Apple Silicon MPS 设备")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("使用 NVIDIA CUDA 设备")
    else:
        device = torch.device("cpu")
        print("使用 CPU 设备")

    # Create model
    model = load_model(args.model, args.pretrained_weights)
    model = model.to(device)

    # Print model
    if args.verbose:
        summary(model, input_size=(3, model.hyperparams['height'], model.hyperparams['height']))

    mini_batch_size = model.hyperparams['batch'] // model.hyperparams['subdivisions']

    # #################
    # Create Dataloader
    # #################

    # Load training dataloader
    dataloader = _create_data_loader(
        train_path,
        mini_batch_size,
        model.hyperparams['height'],
        args.n_cpu,
        args.multiscale_training)

    # Load validation dataloader
    validation_dataloader = _create_validation_data_loader(
        valid_path,
        mini_batch_size,
        model.hyperparams['height'],
        args.n_cpu)

    # ################
    # Create optimizer
    # ################

    params = [p for p in model.parameters() if p.requires_grad]

    if (model.hyperparams['optimizer'] in [None, "adam"]):
        optimizer = optim.Adam(
            params,
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay'],
        )
    elif (model.hyperparams['optimizer'] == "sgd"):
        optimizer = optim.SGD(
            params,
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay'],
            momentum=model.hyperparams['momentum'])
    else:
        print("Unknown optimizer. Please choose between (adam, sgd).")

    # skip epoch zero, because then the calculations for when to evaluate/checkpoint makes more intuitive sense
    # e.g. when you stop after 30 epochs and evaluate every 10 epochs then the evaluations happen after: 10,20,30
    # instead of: 0, 10, 20
    for epoch in range(1, args.epochs+1):

        print("\n---- Training Model ----")
        model.train()  # Set model to training mode

        # 创建进度条
        pbar = tqdm.tqdm(dataloader, desc=f"Training Epoch {epoch}")
        
        for batch_i, (_, imgs, targets) in enumerate(pbar):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = imgs.to(device, non_blocking=True).float()
            targets = targets.to(device).float()

            outputs = model(imgs)
            loss, loss_components = compute_loss(outputs, targets, model)

            # 更新进度条信息
            pbar.set_postfix({
                'iou_loss': f'{float(loss_components[0]):.4f}',
                'obj_loss': f'{float(loss_components[1]):.4f}',
                'cls_loss': f'{float(loss_components[2]):.4f}',
                'total': f'{float(loss_components[3]):.4f}'
            })

            loss.backward()

            ###############
            # Run optimizer
            ###############

            if batches_done % model.hyperparams['subdivisions'] == 0:
                # Adapt learning rate
                # Get learning rate defined in cfg
                lr = model.hyperparams['learning_rate']
                if batches_done < model.hyperparams['burn_in']:
                    # Burn in
                    lr *= (batches_done / model.hyperparams['burn_in'])
                else:
                    # Set and parse the learning rate to the steps defined in the cfg
                    for threshold, value in model.hyperparams['lr_steps']:
                        if batches_done > threshold:
                            lr *= value
                # Log the learning rate
                logger.scalar_summary("train/learning_rate", lr, batches_done)
                # Set learning rate
                for g in optimizer.param_groups:
                    g['lr'] = lr

                # Run optimizer
                optimizer.step()
                # Reset gradients
                optimizer.zero_grad()

            # ############
            # Log progress
            # ############
            if args.verbose:
                print(AsciiTable(
                    [
                        ["Type", "Value"],
                        ["IoU loss", float(loss_components[0])],
                        ["Object loss", float(loss_components[1])],
                        ["Class loss", float(loss_components[2])],
                        ["Loss", float(loss_components[3])],
                        ["Batch loss", to_cpu(loss).item()],
                    ]).table)

            # Tensorboard logging
            tensorboard_log = [
                ("train/iou_loss", float(loss_components[0])),
                ("train/obj_loss", float(loss_components[1])),
                ("train/class_loss", float(loss_components[2])),
                ("train/loss", to_cpu(loss).item())]
            logger.list_of_scalars_summary(tensorboard_log, batches_done)

            model.seen += imgs.size(0)

        # #############
        # Save progress
        # #############

        # Save model to checkpoint file
        if epoch % args.checkpoint_interval == 0:
            # 获取当前的平均loss值
            avg_iou_loss = float(loss_components[0])
            avg_obj_loss = float(loss_components[1])
            avg_cls_loss = float(loss_components[2])
            avg_total_loss = float(loss_components[3])
            
            # 创建backup目录（如果不存在）
            backup_dir = data_config.get("backup", "checkpoints")  # 如果配置文件中没有指定backup，则使用默认的checkpoints目录
            os.makedirs(backup_dir, exist_ok=True)
            
            # 使用loss值构建文件名
            checkpoint_name = (f"yolov3_ckpt_epoch_{epoch}"
                             f"_iou_{avg_iou_loss:.4f}"
                             f"_obj_{avg_obj_loss:.4f}"
                             f"_cls_{avg_cls_loss:.4f}"
                             f"_total_{avg_total_loss:.4f}.pth")
            
            checkpoint_path = os.path.join(backup_dir, checkpoint_name)
            print(f"---- Saving checkpoint to: '{checkpoint_path}' ----")
            
            # 保存模型时确保在CPU上
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.cpu().state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': {
                    'iou_loss': avg_iou_loss,
                    'obj_loss': avg_obj_loss,
                    'cls_loss': avg_cls_loss,
                    'total_loss': avg_total_loss
                }
            }
            
            torch.save(checkpoint, checkpoint_path)
            model.to(device)  # 将模型移回设备
            
            # 打印保存信息
            print(f"Checkpoint saved with losses:")
            print(f"IoU Loss: {avg_iou_loss:.4f}")
            print(f"Object Loss: {avg_obj_loss:.4f}")
            print(f"Class Loss: {avg_cls_loss:.4f}")
            print(f"Total Loss: {avg_total_loss:.4f}")

        # ########
        # Evaluate
        # ########

        if epoch % args.evaluation_interval == 0:
            print("\n---- Evaluating Model ----")
            # Evaluate the model on the validation set
            model = model.to(device)
            metrics_output = _evaluate(
                model,
                validation_dataloader,
                class_names,
                img_size=model.hyperparams['height'],
                iou_thres=args.iou_thres,
                conf_thres=args.conf_thres,
                nms_thres=args.nms_thres,
                verbose=args.verbose,
                device=device
            )

            if metrics_output is not None:
                precision, recall, AP, f1, ap_class = metrics_output
                evaluation_metrics = [
                    ("validation/precision", precision.mean()),
                    ("validation/recall", recall.mean()),
                    ("validation/mAP", AP.mean()),
                    ("validation/f1", f1.mean())]
                logger.list_of_scalars_summary(evaluation_metrics, epoch)


if __name__ == "__main__":
    run()
