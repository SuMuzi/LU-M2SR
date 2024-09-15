import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs
from utils import calculate_psnr,calculate_ssim,tensor2np
import sys
def train_one_epoch(train_loader,
                    model,
                    criterion, 
                    optimizer, 
                    scheduler,
                    epoch, 
                    logger, 
                    config, 
                    scaler=None):
    '''
    train model for one epoch
    '''
    # switch to train mode
    train_total_step = len(train_loader)-1
    model.train() 
 
    loss_list = []
    tq = tqdm(train_total_step, desc=f'Training ', mininterval=0.3)
    for _ in range(train_total_step):
        optimizer.zero_grad()
        images, targets = train_loader.next()
        # images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()
        # images, targets = images.cuda(non_blocking=True).float(), targets.cuda(non_blocking=True).float()
        images = images.to('cuda')
        targets = targets.to('cuda')
        if config.amp:
            with autocast():
                out = model(images)
                loss = criterion(out, targets)      
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(images)
            # print(out.shape)
            loss = criterion(out, targets)
            # print(loss)
            loss.backward()
            optimizer.step()
        
        loss_list.append(loss.item())

        now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        tq.update(1)
    log_info = f'[Train]  epoch: {epoch}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'

    logger.info(log_info)
    print(log_info)
    scheduler.step()
    tq.close()
    return model

def val_one_epoch(val_loader,
                    model,
                    epoch, 
                    logger):
    # switch to evaluate mode

    val_total_step = len(val_loader) - 1
    model.eval()
    psnr_scores = []
    ssim_scores = []
    tq = tqdm(val_total_step, desc=f'Validation ', mininterval=0.3)
    for _ in range(val_total_step):
        with torch.no_grad():

            lr, hr = val_loader.next()
            lr = lr.to('cuda')
            hr = hr.to('cuda')
            # lr, hr = lr.cuda(non_blocking=True).float(), hr.cuda(non_blocking=True).float()

            sr = model(lr)

            psnr_score = 0.0
            ssim_score = 0.0

            for j in range(hr.shape[0]):

                sr_img = tensor2np(sr[j])
                gt_img = tensor2np(hr[j])

                psnr_score += calculate_psnr(sr_img, gt_img)
                ssim_score += calculate_ssim(sr_img, gt_img)

            psnr_scores.append(psnr_score / hr.shape[0])
            ssim_scores.append(ssim_score / hr.shape[0])

            tq.update(1)

    mean_psnr_scores = np.mean(psnr_scores)
    mean_ssim_scores = np.mean(ssim_scores)


    log_info = f'[Validation] epoch: {epoch}, PSNR: {np.mean(mean_psnr_scores):.4f}, SSIM: {np.mean(mean_ssim_scores):.4f}'
    print(log_info)
    logger.info(log_info)
    tq.close()
    
    return np.mean(mean_psnr_scores)


def test_one_epoch(test_loader,
                    model,
                    criterion,
                    logger,
                    config,
                    test_data_name=None):
    # switch to evaluate mode
    model.eval()
    preds = []
    gts = []
    loss_list = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            out = model(img)
            loss = criterion(out, msk)
            loss_list.append(loss.item())
            msk = msk.squeeze(1).cpu().detach().numpy()
            gts.append(msk)
            if type(out) is tuple:
                out = out[0]
            out = out.squeeze(1).cpu().detach().numpy()
            preds.append(out) 
            save_imgs(img, msk, out, i, config.work_dir + 'outputs/', config.datasets, config.threshold, test_data_name=test_data_name)

        preds = np.array(preds).reshape(-1)
        gts = np.array(gts).reshape(-1)

        y_pre = np.where(preds>=config.threshold, 1, 0)
        y_true = np.where(gts>=0.5, 1, 0)

        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        miou = float(TP) / float(TP + FP + FN) if float(TP + FP + FN) != 0 else 0

        if test_data_name is not None:
            log_info = f'test_datasets_name: {test_data_name}'
            print(log_info)
            logger.info(log_info)
        log_info = f'test of best model, loss: {np.mean(loss_list):.4f},miou: {miou}, f1_or_dsc: {f1_or_dsc}, accuracy: {accuracy}, \
                specificity: {specificity}, sensitivity: {sensitivity}, confusion_matrix: {confusion}'
        print(log_info)
        logger.info(log_info)

    return np.mean(loss_list)
