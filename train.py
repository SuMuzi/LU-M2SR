import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from utils import *
from models.LUM2SR import LU_M2SR
from engine import *
import os
import sys
import copy
import argparse
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # "0, 1, 2, 3"

from utils import *
from configs.config_setting import setting_config
from dataloader import GetLoader
import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_LAUNCH_BLOCKING']='1'
def main(config,batch_size):
    print("~~~~~~~~~~~~~~~~current scale is: X{} ~~~~~~~~~~~~~~~~~~~~~".format(config.scale))
    print('#----------config----------#')
    print(config.model_config)
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    os.makedirs(log_dir,exist_ok=True)

    global logger

    logger = get_logger('train', log_dir)

    log_config_info(config, logger)

    n_gpu = torch.cuda.device_count()



    print('#----------GPU init----------#')
    set_seed(config.seed)
    gpu_ids = [0] # 0, 1, 2, 3
    torch.cuda.empty_cache()
    device = 'cuda'

    train_data_path = os.path.join(config.train_data_path, f'train_X{config.scale}.hdf5')
    val_data_path = os.path.join(config.val_data_path, f'valid_X{config.scale}.hdf5')
    test_data_path = os.path.join(config.test_data_path, f'valid_X{config.scale}.hdf5')


    print('#----------Preparing dataset----------#')

    train_loader = GetLoader(train_data_path,
                                batch_size=batch_size,
                                num_workers=config.num_workers,
                                shuffle_is_true=True)
    val_loader = GetLoader(val_data_path,
                                batch_size=n_gpu,
                                num_workers=config.num_workers,
                                shuffle_is_true=False)
    test_loader = GetLoader(val_data_path,
                                batch_size=n_gpu,
                                num_workers=config.num_workers,
                                shuffle_is_true = False)


    print('#----------Prepareing Models----------#')
    model_cfg = config.model_config

    splict_nums = model_cfg['splict_nums']
    c_list = []
    if splict_nums==8:
        c_list= [32, 64, 96, 128, 160, 192, 224, 256]
    elif splict_nums==4:
        c_list = [16, 32, 48, 64, 80, 96, 112, 128]
    elif splict_nums == 1:
        c_list = [32, 64, 96, 128, 160, 192, 224, 256]
    else:
        print("Error splict_nums,it should be 1, 4 or 8!!!!")

    model = LU_M2SR(input_channels=model_cfg['input_channels'],
                    out_channels=model_cfg['num_classes'],
                    rs_factor = config.scale,
                    c_list=c_list,
                    res = model_cfg['res'],
                    split_nums = model_cfg['splict_nums'],
                    atten_config=model_cfg['dic_atten'],
                    ssd_config = model_cfg['dic_ssd'])
    if n_gpu>1:
        # model = torch.nn.DataParallel(model).to('cuda')
        model = torch.nn.DataParallel(model.to(device),output_device=gpu_ids[0])
    else:
        model.to('cuda')

    print('#----------test params and flops----------#')

    cal_params_flops(model,config.input_size_w,logger)


    print('#----------Prepareing loss, opt, sch and amp----------#')

    if config.criterion_type =='L1PSNRSSIMLoss':
        criterion = L1PSNRSSIMLoss()
    elif config.criterion_type =='L1SSIMLoss':
        criterion = L1SSIMLoss()
    elif config.criterion_type =='L1Loss':
        criterion = nn.L1Loss(reduction='mean')

    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = GradScaler()

    print(f'~~~~~~~~~~~~~~~~~[Current config] loss type: {config.criterion_type},  optimizer type: {config.opt}, scheduler type: {config.sch}~~~~~~~~~~~~~~~~~~~~')

    logger.info(f'c_list: {c_list}')


    print('#----------Set other params----------#')
    max_psnr = 0
    start_epoch = 0
    min_epoch = 1
    best_model = None
    best_epoch = 0


    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)


    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs):

        torch.cuda.empty_cache()

        val_model = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            logger,
            config,
            scaler=scaler
        )
        # torch.save(
        #     {
        #         'epoch': epoch,
        #         'min_psnr': min_psnr,
        #         'min_epoch': min_epoch,
        #         'psnr': psnr,
        #         'model_state_dict': model.module.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'scheduler_state_dict': scheduler.state_dict(),
        #     }, os.path.join(checkpoint_dir, 'checkpoint_lasted.pth'.format(epoch)))
        if epoch % config.val_interval ==0:
            psnr = val_one_epoch(
                    val_loader,
                    val_model,
                    epoch,
                    logger
                )

            if max_psnr < psnr:
                best_model = copy.deepcopy(val_model)
                max_psnr = psnr
                best_epoch = epoch
        if epoch % config.save_interval == 0:
            torch.save(val_model.state_dict(), os.path.join(checkpoint_dir, 'checkpoint{}.pth'.format(epoch)))
    print("Best checkpoint is: ",best_epoch)
    torch.save(best_model.state_dict(), os.path.join(checkpoint_dir, 'checkpoint_best_{}.pth'.format(best_epoch)))

    # if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
    #     print('#----------Testing----------#')
    #     best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
    #     model.module.load_state_dict(best_weight)
    #     loss = test_one_epoch(
    #             test_loader,
    #             model,
    #             criterion,
    #             logger,
    #             config,
    #         )
    #     os.rename(
    #         os.path.join(checkpoint_dir, 'best.pth'),
    #         os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
    #     )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UMambaSR')
    # Hardware specifications
    parser.add_argument('--num_workers', type=int, default=8,
                        help='number of threads for data loading')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='input batch size for training')
    par = parser.parse_args()

    config = setting_config
    config.num_workers = par.num_workers
    config.batch_size = par.batch_size


    main(config,par.batch_size)

    print(config)