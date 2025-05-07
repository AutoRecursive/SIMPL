import os
import sys
import time
import argparse
import faulthandler
from datetime import datetime
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from loader import Loader
from utils.utils import AverageMeterForDict, set_seed, save_ckpt
from utils.logger import Logger


def parse_arguments():
    """Arguments for running the two-stage training (pretrain + RL)."""
    parser = argparse.ArgumentParser()
    # General settings
    parser.add_argument("--features_dir", required=True, type=str, help="Path to the dataset")
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA for acceleration")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--logger_writer", action="store_true", help="Enable tensorboard logging")
    parser.add_argument("--no_pbar", action="store_true", help="Disable progress bar")

    # Pretrain stage settings
    parser.add_argument("--pretrain_cfg_path", type=str, default="config.simpl_av2_cfg",
                        help="Config path for pretrain stage")
    parser.add_argument("--pretrain_epochs", type=int, default=20,
                        help="Number of epochs for pretrain stage")
    parser.add_argument("--pretrain_batch_size", type=int, default=4,
                        help="Batch size for pretrain stage")
    parser.add_argument("--pretrain_val_batch_size", type=int, default=4,
                        help="Validation batch size for pretrain stage")
    parser.add_argument("--pretrain_val_interval", type=int, default=2,
                        help="Validation interval for pretrain stage")
    parser.add_argument("--resume_pretrain", action="store_true",
                        help="Resume pretrain from checkpoint")
    parser.add_argument("--pretrain_model_path", type=str, default="",
                        help="Path to pretrain model checkpoint")

    # RL stage settings
    parser.add_argument("--rl_cfg_path", type=str, default="config.simpl_rl_cfg",
                        help="Config path for RL stage")
    parser.add_argument("--rl_epochs", type=int, default=10,
                        help="Number of epochs for RL stage")
    parser.add_argument("--rl_batch_size", type=int, default=4,
                        help="Batch size for RL stage")
    parser.add_argument("--rl_val_batch_size", type=int, default=4,
                        help="Validation batch size for RL stage")
    parser.add_argument("--rl_val_interval", type=int, default=1,
                        help="Validation interval for RL stage")
    parser.add_argument("--skip_pretrain", action="store_true",
                        help="Skip pretrain stage and directly go to RL stage")

    # Data augmentation
    parser.add_argument("--data_aug", action="store_true", help="Enable data augmentation")

    return parser.parse_args()


def train_one_epoch(net, loss_fn, optimizer, train_loader, evaluator, logger, epoch, args, stage="pretrain"):
    """Train for one epoch"""
    epoch_start = time.time()
    train_loss_meter = AverageMeterForDict()
    train_eval_meter = AverageMeterForDict()
    net.train()

    niter = 0
    for i, data in enumerate(tqdm(train_loader, disable=args.no_pbar, ncols=80)):
        data_in = net.pre_process(data)
        out = net(data_in)
        loss_out = loss_fn(out, data)

        post_out = net.post_process(out)
        eval_out = evaluator.evaluate(post_out, data)

        optimizer.zero_grad()
        loss_out['loss'].backward()
        lr = optimizer.step()

        train_loss_meter.update(loss_out)
        train_eval_meter.update(eval_out)
        niter += args.train_batch_size  # Use the current batch size
        logger.add_dict(loss_out, niter, prefix=f'{stage}/train/')

    optimizer.step_scheduler()
    max_memory = torch.cuda.max_memory_allocated(device=device) // 2 ** 20

    loss_avg = train_loss_meter.metrics['loss'].avg
    logger.print(f'[{stage.capitalize()} Training] Epoch {epoch}, Avg. loss: {loss_avg:.6f}, '
                 f'time cost: {(time.time() - epoch_start) / 60.0:.3f} mins, '
                 f'lr: {lr:.6f}, peak mem: {max_memory} MB')
    logger.print('-- ' + train_eval_meter.get_info())

    logger.add_scalar(f'{stage}/train/lr', lr, it=epoch)
    logger.add_scalar(f'{stage}/train/max_mem', max_memory, it=epoch)
    for key, elem in train_eval_meter.metrics.items():
        logger.add_scalar(title=f'{stage}/train/{key}', value=elem.avg, it=epoch)

    return train_loss_meter, train_eval_meter, lr


def validate(net, loss_fn, val_loader, evaluator, logger, epoch, args, stage="pretrain"):
    """Validate the model"""
    with torch.no_grad():
        val_start = time.time()
        val_loss_meter = AverageMeterForDict()
        val_eval_meter = AverageMeterForDict()
        net.eval()

        for i, data in enumerate(tqdm(val_loader, disable=args.no_pbar, ncols=80)):
            data_in = net.pre_process(data)
            out = net(data_in)
            loss_out = loss_fn(out, data)

            post_out = net.post_process(out)
            eval_out = evaluator.evaluate(post_out, data)

            val_loss_meter.update(loss_out)
            val_eval_meter.update(eval_out)

        logger.print(f'[{stage.capitalize()} Validation] Epoch {epoch}, Avg. loss: {val_loss_meter.metrics["loss"].avg:.6f}, '
                     f'time cost: {(time.time() - val_start) / 60.0:.3f} mins')
        logger.print('-- ' + val_eval_meter.get_info())

        for key, elem in val_loss_meter.metrics.items():
            logger.add_scalar(title=f'{stage}/val/{key}', value=elem.avg, it=epoch)
        for key, elem in val_eval_meter.metrics.items():
            logger.add_scalar(title=f'{stage}/val/{key}', value=elem.avg, it=epoch)

    return val_loss_meter, val_eval_meter


def main():
    args = parse_arguments()

    faulthandler.enable()
    start_time = time.time()
    set_seed(args.seed)

    global device
    if args.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda", 0)
    else:
        device = torch.device('cpu')

    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "log/" + date_str
    logger = Logger(date_str=date_str, log_dir=log_dir, enable_flags={'writer': args.logger_writer})
    # log basic info
    logger.print(f"Starting two-stage training: Pretrain + RL")
    logger.print(f"Date: {date_str}")
    logger.print(f"Args: {args}")

    # ===== Stage 1: Pretrain =====
    if not args.skip_pretrain:
        logger.print("\n" + "="*50)
        logger.print("Stage 1: Pretraining with supervised learning")
        logger.print("="*50)

        # Set config for pretrain
        args.train_batch_size = args.pretrain_batch_size
        args.val_batch_size = args.pretrain_val_batch_size
        args.adv_cfg_path = args.pretrain_cfg_path
        args.mode = 'train'  # Set mode for dataset loading

        # Load pretrain model, dataset, and optimizer
        pretrain_loader = Loader(args, device, is_ddp=False)
        if args.resume_pretrain and args.pretrain_model_path:
            logger.print(f'[Resume] Loading pretrain state_dict from {args.pretrain_model_path}')
            pretrain_loader.set_resmue(args.pretrain_model_path)  # Note: This is a typo in loader.py, but we keep it for compatibility

        (train_set, val_set), pretrain_net, pretrain_loss_fn, pretrain_optimizer, pretrain_evaluator = pretrain_loader.load()

        dl_pretrain = DataLoader(train_set,
                              batch_size=args.pretrain_batch_size,
                              shuffle=True,
                              num_workers=8,
                              collate_fn=train_set.collate_fn,
                              drop_last=True,
                              pin_memory=True)
        dl_pretrain_val = DataLoader(val_set,
                                  batch_size=args.pretrain_val_batch_size,
                                  shuffle=False,
                                  num_workers=8,
                                  collate_fn=val_set.collate_fn,
                                  drop_last=True,
                                  pin_memory=True)

        best_pretrain_metric = 1e6
        rank_metric = "minfde_k"  # Metric to use for model selection
        net_name = pretrain_loader.network_name()

        # Pretrain loop
        for epoch in range(args.pretrain_epochs):
            logger.print(f'\nPretrain Epoch {epoch}')
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Train
            _, _, _ = train_one_epoch(
                pretrain_net, pretrain_loss_fn, pretrain_optimizer,
                dl_pretrain, pretrain_evaluator, logger, epoch, args,
                stage="pretrain"
            )

            # Validation
            if ((epoch + 1) % args.pretrain_val_interval == 0) or epoch > int(args.pretrain_epochs / 2):
                _, val_eval_meter = validate(
                    pretrain_net, pretrain_loss_fn, dl_pretrain_val,
                    pretrain_evaluator, logger, epoch, args,
                    stage="pretrain"
                )

                # Save best model
                if (epoch >= args.pretrain_epochs / 2):
                    if val_eval_meter.metrics[rank_metric].avg < best_pretrain_metric:
                        model_name = date_str + f'_pretrain_{net_name}_best.tar'
                        save_ckpt(pretrain_net, pretrain_optimizer, epoch, 'saved_models/', model_name)
                        best_pretrain_metric = val_eval_meter.metrics[rank_metric].avg
                        logger.print(f'Save the pretrain model: {model_name}, {rank_metric}: {best_pretrain_metric:.4f}, epoch: {epoch}')

                        # Save this as the starting point for RL stage
                        pretrain_final_path = f'{date_str}_pretrain_final.tar'
                        save_ckpt(pretrain_net, pretrain_optimizer, epoch, 'saved_models', pretrain_final_path)
                        logger.print(f'Saved pretrain final model to saved_models/{pretrain_final_path}')

            # Save checkpoint at regular intervals
            if int(100 * epoch / args.pretrain_epochs) in [20, 40, 60, 80]:
                model_name = date_str + f'_pretrain_{net_name}_ckpt_epoch{epoch}.tar'
                save_ckpt(pretrain_net, pretrain_optimizer, epoch, 'saved_models/', model_name)
                logger.print(f'Save the pretrain model to saved_models/{model_name}')

        # Save final pretrain model
        pretrain_final_path = f'{date_str}_pretrain_final.tar'
        save_ckpt(pretrain_net, pretrain_optimizer, args.pretrain_epochs-1, 'saved_models', pretrain_final_path)
        logger.print(f'Saved pretrain final model to saved_models/{pretrain_final_path}')
        logger.print(f"\nPretraining completed in {(time.time() - start_time) / 60.0:.2f} mins")

    # ===== Stage 2: RL Training =====
    logger.print("\n" + "="*50)
    logger.print("Stage 2: Reinforcement Learning with PPO")
    logger.print("="*50)

    # Set config for RL
    args.train_batch_size = args.rl_batch_size
    args.val_batch_size = args.rl_val_batch_size
    args.adv_cfg_path = args.rl_cfg_path
    args.mode = 'train'  # Set mode for dataset loading

    # Load RL model, dataset, and optimizer
    rl_loader = Loader(args, device, is_ddp=False)

    # If we did pretrain, use the final pretrain model as starting point
    if not args.skip_pretrain:
        pretrain_final_path = f'saved_models/{date_str}_pretrain_final.tar'
        logger.print(f'[RL] Loading pretrained model from {pretrain_final_path}')
        rl_loader.set_resmue(pretrain_final_path)  # Note: This is a typo in loader.py, but we keep it for compatibility
    # Otherwise, use the specified model path
    elif args.pretrain_model_path:
        logger.print(f'[RL] Loading pretrained model from {args.pretrain_model_path}')
        rl_loader.set_resmue(args.pretrain_model_path)  # Note: This is a typo in loader.py, but we keep it for compatibility

    (train_set, val_set), rl_net, rl_loss_fn, rl_optimizer, rl_evaluator = rl_loader.load()

    dl_rl = DataLoader(train_set,
                      batch_size=args.rl_batch_size,
                      shuffle=True,
                      num_workers=8,
                      collate_fn=train_set.collate_fn,
                      drop_last=True,
                      pin_memory=True)
    dl_rl_val = DataLoader(val_set,
                          batch_size=args.rl_val_batch_size,
                          shuffle=False,
                          num_workers=8,
                          collate_fn=val_set.collate_fn,
                          drop_last=True,
                          pin_memory=True)

    best_rl_metric = 1e6
    rank_metric = "minfde_k"  # Metric to use for model selection
    net_name = rl_loader.network_name()

    rl_start_time = time.time()

    # RL training loop
    for epoch in range(args.rl_epochs):
        logger.print(f'\nRL Epoch {epoch}')
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Train
        _, _, _ = train_one_epoch(
            rl_net, rl_loss_fn, rl_optimizer,
            dl_rl, rl_evaluator, logger, epoch, args,
            stage="rl"
        )

        # Validation
        if ((epoch + 1) % args.rl_val_interval == 0) or epoch > int(args.rl_epochs / 2):
            _, val_eval_meter = validate(
                rl_net, rl_loss_fn, dl_rl_val,
                rl_evaluator, logger, epoch, args,
                stage="rl"
            )

            # Save best model
            if val_eval_meter.metrics[rank_metric].avg < best_rl_metric:
                model_name = date_str + f'_rl_{net_name}_best.tar'
                save_ckpt(rl_net, rl_optimizer, epoch, 'saved_models/', model_name)
                best_rl_metric = val_eval_meter.metrics[rank_metric].avg
                logger.print(f'Save the RL model: {model_name}, {rank_metric}: {best_rl_metric:.4f}, epoch: {epoch}')

        # Save checkpoint at regular intervals
        if int(100 * epoch / args.rl_epochs) in [20, 40, 60, 80]:
            model_name = date_str + f'_rl_{net_name}_ckpt_epoch{epoch}.tar'
            save_ckpt(rl_net, rl_optimizer, epoch, 'saved_models/', model_name)
            logger.print(f'Save the RL model to saved_models/{model_name}')

    # Save final RL model
    model_name = date_str + f'_rl_{net_name}_final.tar'
    save_ckpt(rl_net, rl_optimizer, args.rl_epochs-1, 'saved_models/', model_name)
    logger.print(f'Save the final RL model to saved_models/{model_name}')

    logger.print(f"\nRL training completed in {(time.time() - rl_start_time) / 60.0:.2f} mins")
    logger.print(f"\nTotal training completed in {(time.time() - start_time) / 60.0:.2f} mins")
    logger.print('\nExit...\n')


if __name__ == "__main__":
    main()
