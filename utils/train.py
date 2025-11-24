import gc
import time
import atexit
import random
import smtplib
import logging
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from dotenv import dotenv_values
from utils.metrics import calculate_dice_score, calculate_iou_score

def training_step(dataloader, 
                  model, 
                  model_name,
                  optimizer,
                  criterion,
                  device):
    
    total_loss, total_dice_score, total_iou_score = 0.0, 0.0, 0.0
    
    for i, batch in enumerate(dataloader):
        optimizer.zero_grad()
        images, labels = batch 
        images, labels = images.to(device), labels.to(device)
        
        logits = model(images)
        if model_name=="polyp_pvt":
            batch_loss1 = criterion(logits[0], labels)
            batch_loss2 = criterion(logits[1], labels)
            batch_loss = batch_loss1 + batch_loss2 
        else:
            batch_loss = criterion(logits, labels)
        batch_loss.backward()
        optimizer.step()
        if model_name=="polyp_pvt":
            batch_dice_score = calculate_dice_score(preds=logits[0], targets=labels, device=device, model_name=model_name)
            batch_iou_score = calculate_iou_score(preds=logits[0], targets=labels, device=device, model_name=model_name)
        else:
            batch_dice_score = calculate_dice_score(preds=logits, targets=labels, device=device, model_name=model_name)
            batch_iou_score = calculate_iou_score(preds=logits, targets=labels, device=device, model_name=model_name)
        
        total_loss += batch_loss.item()
        total_dice_score += batch_dice_score.item()
        total_iou_score += batch_iou_score.item()
    return model, optimizer, total_loss/len(dataloader), total_dice_score/len(dataloader), total_iou_score/len(dataloader)   

  
def check_stopping_conditions(epoch_loss_list, 
                              current_loss,
                              current_ma_loss,
                              ma_window, 
                              threshold,
                              min_epochs=100):
    if len(epoch_loss_list)<ma_window or len(epoch_loss_list)<min_epochs:
        return False
    if abs(current_loss) < 1e-8:  # Small constant to avoid division by zero
        return False
    
    relative_diff = abs(current_loss - current_ma_loss) / current_loss
    
    if relative_diff <= threshold:
        return True
    
    return False

def test_loop(model, 
              model_name,
              test_dataloader, 
              criterion, 
              device,
              num_repeat):
    # ? TESTING THE DATASET `test_limit` TIMES AND TAKING AN AVARAGE FOR REDUCING RANDOMNESS
    final_test_loss, final_dice_score, final_miou_score = 0.0, 0.0, 0.0
    for _ in range(num_repeat):
        model.eval()
        all_test_dice_scores = []
        total_loss, total_dice_score, total_miou_score = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in test_dataloader:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                logits = model(images)

                if model_name=="polyp_pvt":
                    batch_loss1 = criterion(logits[0], labels)
                    batch_loss2 = criterion(logits[1], labels)
                    loss = batch_loss1 + batch_loss2 
        
                    dice_score = calculate_dice_score(
                        preds=logits[0], 
                        targets=labels, 
                        device=device, 
                        model_name=model_name)
                    iou_score = calculate_iou_score(
                        preds=logits[0], 
                        targets=labels, device=device, 
                        model_name=model_name)
                else:
                    dice_score = calculate_dice_score(
                        preds=logits, 
                        targets=labels, 
                        device=device, 
                        model_name=model_name)
                    iou_score = calculate_iou_score(
                        preds=logits, 
                        targets=labels, 
                        device=device, 
                        model_name=model_name)
                    loss = criterion(logits, labels)
    
                
                all_test_dice_scores.append(dice_score)
                
                total_loss += loss.item()
                total_dice_score += dice_score.item()
                total_miou_score += iou_score.item()

        # probabilities, entropy = get_entropy(test_dice_scores=all_test_dice_scores)

        mean_loss = total_loss / len(test_dataloader)
        mean_dice_score = total_dice_score / len(test_dataloader)
        mean_miou_score = total_miou_score / len(test_dataloader)
        final_test_loss += mean_loss
        final_dice_score += mean_dice_score
        final_miou_score += mean_miou_score
    return final_test_loss/num_repeat, final_dice_score/num_repeat, final_miou_score/num_repeat






def get_lr_scheduler(optimizer, 
                     num_epochs: int, 
                     warmup_epochs: int = 5, 
                     min_lr: float = 1e-6):
    
    warmup = LinearLR(optimizer, 
                     start_factor=0.1,
                     total_iters=warmup_epochs)
    
    cosine = CosineAnnealingLR(optimizer, 
                               T_max=num_epochs - warmup_epochs, 
                               eta_min=min_lr)
    
    return SequentialLR(optimizer, 
                       schedulers=[warmup, cosine], 
                       milestones=[warmup_epochs])

def cleanup_iteration(variables, device, logger, wait_time: int = 5):
    try:
        if 'model' in variables:
            variables['model'].cpu()
        
        keys_to_delete = list(variables.keys())
        
        for var in keys_to_delete:
            del variables[var]
            
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
        time.sleep(wait_time)
    
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")



def training_loop(dataloader, 
                  model, 
                  model_name,
                  train_data_size,
                  optimizer,
                  scheduler,
                  criterion,
                  device,
                  threshold, 
                  ma_window,
                  max_epochs, 
                  min_epochs, 
                  best_model_save_path, 
                  logger,
                  use_scheduler=False, 
                  save_model=True,
                  val_dataloader=None,         # <-- NEW: optional validation loader
                  val_num_repeat=1):           # <-- NEW: how many times to average val
    model.to(device)
    model.train()

    # train metrics (same as before)
    epoch_loss_list = []
    ma_loss_list = []
    epoch_dice_score_list = []
    epoch_miou_score_list = []

    # validation metrics (only used internally for convergence / best model)
    val_loss_list = []

    min_loss = float('inf')   # best loss (train or val, depending on what's used)
    best_model, best_optimizer = None, None
    best_model_epoch = 0
    min_loss_dice_score, min_loss_miou_score = 0.0, 0.0

    epoch = 0
    
    logger.info(f"Choosen train subset has: {train_data_size} data")
    logger.info(f"Training started...")
    while True:
        epoch += 1
        model.train()

        ## curriculum epoch set
        if hasattr(dataloader.dataset, "diff_aug") and dataloader.dataset.diff_aug is not None:
            dataloader.dataset.diff_aug.set_epoch(epoch)

        # -------- TRAIN STEP --------
        model, optimizer, mean_epoch_loss, mean_epoch_dice_score, mean_epoch_miou_score = training_step(
            dataloader=dataloader,
            model=model, 
            model_name=model_name, 
            optimizer=optimizer, 
            criterion=criterion, 
            device=device
        )
        
        epoch_loss_list.append(mean_epoch_loss)
        epoch_dice_score_list.append(mean_epoch_dice_score)
        epoch_miou_score_list.append(mean_epoch_miou_score)

        # -------- VALIDATION via test_loop (if provided) --------
        if val_dataloader is not None:
            val_loss, val_dice, val_iou = test_loop(
                model=model,
                model_name=model_name,
                test_dataloader=val_dataloader,
                criterion=criterion,
                device=device,
                num_repeat=val_num_repeat,
            )
            val_loss_list.append(val_loss)

            # For convergence & best model, use VALIDATION loss
            loss_for_early_stop = val_loss
            history_for_ma = val_loss_list

            logger.info(
                f"Epoch [{epoch:04}] "
                f"Train - Loss: {mean_epoch_loss:.4f}, Dice: {mean_epoch_dice_score:.4f}, mIoU: {mean_epoch_miou_score:.4f} | "
                f"Val - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, mIoU: {val_iou:.4f}"
            )
        else:
            # No validation loader: keep original behaviour (train-only)
            loss_for_early_stop = mean_epoch_loss
            history_for_ma = epoch_loss_list

            logger.info(
                f"Epoch [{epoch:04}], Loss: {mean_epoch_loss:.4f}, "
                f"Dice Score: {mean_epoch_dice_score:.4f}, mIoU Score: {mean_epoch_miou_score:.4f}"
            )

        # -------- Moving-average loss (on whichever loss we're using) --------
        ma_loss = np.mean(history_for_ma[-ma_window:])
        ma_loss_list.append(ma_loss)

        # -------- Track best model based on loss_for_early_stop --------
        if loss_for_early_stop < min_loss:
            min_loss = loss_for_early_stop
            best_model_epoch = epoch

            if val_dataloader is not None:
                min_loss_dice_score = val_dice
                min_loss_miou_score = val_iou
            else:
                min_loss_dice_score = mean_epoch_dice_score
                min_loss_miou_score = mean_epoch_miou_score

            best_model = copy.deepcopy(model)
            best_optimizer = copy.deepcopy(optimizer)

        # -------- Check convergence / early stopping --------
        stopping = check_stopping_conditions(
            epoch_loss_list=history_for_ma,
            current_loss=loss_for_early_stop,
            current_ma_loss=ma_loss,
            ma_window=ma_window, 
            threshold=threshold, 
            min_epochs=min_epochs
        )
        
        if stopping:
            logger.warning("Metrics converged and loss stabilized. Training stopped!")
            break
        if epoch >= max_epochs:
            logger.warning("Reached maximum number of epochs. Training stopped")
            break
         
        if use_scheduler:
            scheduler.step()
    
    if save_model and best_model is not None:
        torch.save({
            'model_state_dict': best_model.state_dict(),
            'optimizer_state_dict': best_optimizer.state_dict()
        }, f'{best_model_save_path}.traindata_{train_data_size}.pt')
        logger.info(
            f"Best model found at Epoch: {best_model_epoch:04}. "
            f"Loss: {min_loss:.4f}, Dice score: {min_loss_dice_score:.4f}, "
            f"mIoU score: {min_loss_miou_score:.4f}"
        )
    
    return best_model, epoch_loss_list, ma_loss_list, epoch_dice_score_list, epoch_miou_score_list
