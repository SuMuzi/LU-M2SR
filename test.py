import os
import cv2
import time
import torch
import torch.nn as nn
import random
import argparse
import numpy as np
# from train import Trainer
from datetime import datetime
# from data import DIV2K_train, Set5_val
from torch.utils.data import DataLoader
from dataloader import GetLoader2
# from model import cfat
# from model import hat
# from model import swin2sr
from tqdm import tqdm
import logging
import logging.handlers
os.environ['CUDA_VISIBLE_DEVICES']='0'
from thop import profile,clever_format
from PIL import Image
from configs.config_setting import setting_config

from models.LUM2SR import LU_M2SR

def get_config():
    config = setting_config
    parser = argparse.ArgumentParser(description="LU-M2SR")

    parser.add_argument("--model", type=str, default='LU-M2SR')
    parser.add_argument("--output_dir", type=str, default="/public/001/suqingguo/w2_common/test_out/")
    parser.add_argument("--log_dir", type=str, default="")
    parser.add_argument("--test_data_root_path", type=str, default="/public/001/suqingguo/w2_common/data/test_data/",
                        help='')
    parser.add_argument('--in_channel', type=int, default=3, help='')
    parser.add_argument('--latent_dim', type=int, default=64, help='')

    parser.add_argument("--ck_x2_path", type=str,
                        default="/public/001/suqingguo/w2_common/project_out/Swin2SR/DF2K/x2/2024-08-19-01:35:26/train/checkpoint/best_checkpoint_100.pth",
                        help='')
    parser.add_argument("--ck_x3_path", type=str,
                        default="/public/001/suqingguo/w2_common/project_out/Swin2SR/DF2K/x3/2024-08-19-10:04:19/train/checkpoint/best_checkpoint_70.pth",
                        help='')
    parser.add_argument("--ck_x4_path", type=str,
                        default="/public/001/suqingguo/w2_common/project_out/LU-M2SR/DF2K/x4/Saturday_07_September_2024_02h_24m_10s/checkpoints/checkpoint200.pth",
                        help='')
#4100#"/public/001/suqingguo/w2_common/project_out/LU-M2SR/DF2K/x4/Saturday_07_September_2024_16h_24m_03s/checkpoints/checkpoint_best_65.pth"
#4111#"/public/001/suqingguo/w2_common/project_out/LU-M2SR/DF2K/x4/Saturday_07_September_2024_19h_47m_01s/checkpoints/checkpoint_best_165.pth"
#4811#"/public/001/suqingguo/w2_common/project_out/LU-M2SR/DF2K/x4/Saturday_07_September_2024_15h_59m_33s/checkpoints/checkpoint160.pth"
#4811#"/public/001/suqingguo/w2_common/project_out/LU-M2SR/DF2K/x4/Saturday_07_September_2024_02h_24m_10s/checkpoints/checkpoint200.pth"
#4800#"/public/001/suqingguo/w2_common/project_out/LU-M2SR/DF2K/x4/Saturday_07_September_2024_03h_14m_18s/checkpoints/checkpoint150.pth"
    par = parser.parse_args()

    return config,par

def save_img(path,data,name):
    img = Image.fromarray(data,'RGB')
    img_dir = os.path.join(path,f'{name}.png')
    img.save(img_dir,dpi=(300,300))

def reorder_image(img, input_order='HWC'):

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' "'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img

def cal_params_flops(model, scale, size, logger):
    input = torch.randn(1, 3, size, size).cuda()
    flops, params = profile(model, inputs=(input,))
    f = flops/1e9
    p = params/1e6

    print('{0} Flops {1} G'.format(str(scale),str(f)))			## 打印计算量
    print('{0} Params {1} M'.format(str(scale),str(p)))			## 打印参数量

    total = sum(p.numel() for p in model.parameters())

    t = total/1e6
    print("Total params: {} M".format(str(t)))
    logger.info(f'Flops: {f} G, Params: {p} M, Total params: {t} M')

def get_loader(path,item,sacle):
    data_path = os.path.join(path,item)
    data_path = os.path.join(data_path, 'hdf5')
    data_path = os.path.join(data_path, f'test_{sacle}.hdf5')
    train_data_loader = GetLoader2(data_path, batch_size=1, num_workers=4,
                                   shuffle_is_true=False)
    return train_data_loader

def set_gpu(model):
    if torch.cuda.is_available():
        device = 'cuda'
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model).to(device)
        else:
            model = model.to(device)

    else:
        print('Use cpu training')
        device = 'cpu'
        model = model.to(device)
    TensorType = torch.FloatTensor
    return model,TensorType,device



def load_model_state_dict(model,path):
    if not os.path.exists(path):
        ValueError("Not find {}".format(path))
    ck = torch.load(path)
    model = model.load_state_dict({k.replace('module.',''):v for k,v in ck.items()},strict=False)
    # model = model.load_state_dict(ck)
    print("Successfully load the trained model from {}".format(path))

    return model

def get_logger(name, log_dir):
    '''
    Args:
        name(str): name of logger
        log_dir(str): path of log
    '''

    if not os.path.exists(log_dir):
        os.makedirs(log_dir,exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    info_name = os.path.join(log_dir, '{}.info.log'.format(name))
    info_handler = logging.handlers.TimedRotatingFileHandler(info_name,
                                                             when='D',
                                                             encoding='utf-8')
    info_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    info_handler.setFormatter(formatter)

    logger.addHandler(info_handler)

    return logger

def tensor2np(tensor, out_type=np.uint8, min_max=(0, 1)):
    tensor = tensor.cpu().clamp_(*min_max)
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0, 1]
    img_np = tensor.numpy()
    img_np = np.transpose(img_np, (1, 2, 0))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()

    return img_np.astype(out_type)
def calculate_psnr(img1, img2, crop_border=0, input_order='HWC', test_y_channel=False):

    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


def _ssim(img1, img2):

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2, crop_border=0, input_order='HWC', test_y_channel=False):

    assert img1.shape == img2.shape, (f'Image shapes are differnet: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img1 = to_y_channel(img1)
        img2 = to_y_channel(img2)

    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))
    return np.array(ssims).mean()


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    print("Main GPUs is: ", torch.cuda.current_device())

    now_tm = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    config,par = get_config()
    model_cfg = config.model_config
    print(f"~~~~~~~~~~~~~~~~~~~~~~~Current {config.network} Configuration~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(config)

    items = ['Set5','Set14','Manga109','BSD109','Urban100']
    scales = ['X4']#'X2','X3',
    input_size = 64

    for scale in scales:
        for item in items:

            if scale=='X2':
                sf = 2
            elif scale=='X3':
                sf = 3
            elif scale=='X4':
                sf = 4

            output_dir = os.path.join(par.output_dir,scale,item,par.model,'4800',now_tm)
            image_out_dir = os.path.join(output_dir,'img')
            os.makedirs(image_out_dir,exist_ok=True)

            print("~~~~~~~~~~Set logger~~~~~~~~~~~")
            log_name = f'Test_{item}_{scale}'
            logger = get_logger(log_name, output_dir)

            print("~~~~~~~~~~Load dataset~~~~~~~~~~~")
            train_data_loader = get_loader(par.test_data_root_path,item,scale)

            splict_nums = model_cfg['splict_nums']
            c_list = []
            if splict_nums == 8:
                c_list = [32, 64, 96, 128, 160, 192, 224, 256]
            elif splict_nums == 4:
                c_list = [16, 32, 48, 64, 80, 96, 112, 128]
            elif splict_nums == 1:
                c_list = [32, 64, 96, 128, 160, 192, 224, 256]
            else:
                print("Error splict_nums,it should be 1, 4 or 8!!!!")

            print("~~~~~~~~~~Load model~~~~~~~~~~~")
            model = LU_M2SR(input_channels=model_cfg['input_channels'],
                            out_channels=model_cfg['num_classes'],
                            rs_factor=config.scale,
                            c_list=c_list,
                            res=model_cfg['res'],
                            split_nums=model_cfg['splict_nums'],
                            atten_config=model_cfg['dic_atten'],
                            ssd_config=model_cfg['dic_ssd'])

            if sf==2:
                ck_path = par.ck_x2_path
            elif sf == 3:
                ck_path = par.ck_x3_path
            elif sf==4:
                ck_path = par.ck_x4_path


            print("~~~~~~~~~~Set GPU~~~~~~~~~~~")
            model,tesor_type,device = set_gpu(model)
            ck = torch.load(ck_path)
            model.load_state_dict({k.replace('module.', ''): v for k, v in ck.items()},strict=False)
            # model = model.load_state_dict(ck)
            print("Successfully load the trained model from {}".format(ck_path))
            # model = load_model_state_dict(model, ck_path)
            cal_params_flops(model, sf, input_size, logger)

            tq = tqdm(len(train_data_loader) - 1)
            psnr = []
            ssim = []
            tm = []
            for iteration in range(len(train_data_loader) - 1):

                lr_tensor, hr_tensor,name = train_data_loader.next()

                lr_tensor = lr_tensor.to(device)
                hr_tensor = hr_tensor.to(device)


                model.eval()
                with torch.no_grad():
                    t_start = time.time()
                    sr_tensor = model(lr_tensor)
                    t_end = time.time()

                sr_img = tensor2np(sr_tensor[0])
                gt_img = tensor2np(hr_tensor[0])

                p = calculate_psnr(sr_img, gt_img)
                s = calculate_ssim(sr_img, gt_img)
                t = t_end - t_start

                psnr.append(p)
                ssim.append(s)
                tm.append(t)

                save_img(image_out_dir,sr_img,name)

                tq.update(1)

                print("Validation Results {0} - Name: {1}".format(iteration,name))

                logger.info(f'Testing Results {iteration} - Name {name}: PSNR: {p:.6f},|| SSIM: {s:.6f},|| Time:{t:.9f} s')
                print("PSNR: {:.6f}".format(p))
                print("SSIM: {:.6f}".format(s))
                print("time: {:.9f} s".format(t))
            tq.close()

            print("===> Valid. psnr: {:.6f}, ssim: {:.6f}".format(np.mean(psnr),
                                                                  np.mean(ssim)))

            logger.info(f'Testing {item}_{scale} Average Results: AVG_PSNR: {np.mean(psnr):.6f},|| AVG_SSIM: {np.mean(ssim):.6f},||AVG_TIME: {np.mean(tm):.9f} s ')
            logging.shutdown()














