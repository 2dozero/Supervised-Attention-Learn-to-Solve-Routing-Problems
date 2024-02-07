import os
import time
from tqdm import tqdm
import torch
import math
import json
import wandb

from torch.utils.data import DataLoader
from torch.nn import DataParallel

from nets.attention_model import set_decode_type
from utils.log_utils import log_values
from utils import move_to
from problems.tsp.tsp_gurobi import *


def get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, dataset, opts):
    # Validate
    print('Validating...')
    cost = rollout(model, dataset, opts)
    avg_cost = cost.mean()
    print('Validation overall avg_cost: {} +- {}'.format(
        avg_cost, torch.std(cost) / math.sqrt(len(cost))))

    return avg_cost


def rollout(model, dataset, opts):
    # Put in greedy evaluation mode!
    set_decode_type(model, "greedy")
    model.eval()

    def eval_model_bat(bat, epoch, batch_id):
        with torch.no_grad():
            cost, _ = model(move_to(bat, opts.device), epoch, batch_id, val_mode = True)
        return cost.data.cpu()

    return torch.cat([
        eval_model_bat(bat, epoch = 0, batch_id = 0) # doyoung # check
        for bat
        in tqdm(DataLoader(dataset, batch_size=opts.eval_batch_size), disable=opts.no_progress_bar)
    ], 0)


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, val_dataset, problem, tb_logger, opts):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    step = epoch * (opts.epoch_size // opts.batch_size)
    start_time = time.time()

    # if not opts.no_tensorboard:
    #     tb_logger.log_value('learnrate_pg0', optimizer.param_groups[0]['lr'], step)

    # Generate new training data for each epoch
    training_dataset = baseline.wrap_dataset(problem.make_dataset(
        size=opts.graph_size, num_samples=opts.epoch_size, distribution=opts.data_distribution)) # len(training_dataset) = 1280000, next(iter(training_dataset)).shape = [20, 2]
    training_dataloader = DataLoader(training_dataset, batch_size=opts.batch_size, num_workers=1) # len(training_dataloader) = 2500, next(iter(training_dataloader)).shape = [512, 20, 2]

    # Put model in train mode!
    model.train()
    set_decode_type(model, "sampling")

    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)): # batch.shape : [512, 20, 2]

        train_batch(
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            step,
            batch,
            tb_logger,
            opts
        )

        step += 1

    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('Saving model and state...')
        torch.save(
            {
                'model': get_inner_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
                'baseline': baseline.state_dict()
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )

    avg_reward = validate(model, val_dataset, opts)

    if opts.wandb != 'disabled':
        wandb_log_dict = {'val_avg_cost': avg_reward,
                        # 'cum_samples': cum_samples,
                        'step': step}

        wandb.log(wandb_log_dict)

    # if not opts.no_tensorboard:
    #     tb_logger.log_value('val_avg_reward', avg_reward, step)

    baseline.epoch_callback(model, epoch)

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()

def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        step,
        batch,
        tb_logger,
        opts
):
    x, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None

    # expert_results = solve_all_gurobi(x.cpu().numpy())
    
    # # 지정된 절대 경로에 에폭 번호를 포함한 폴더 생성
    # expert_data_epoch_dir = os.path.join('/home/doyoung/바탕화면/new_super_attention/attention-learn-to-route/expert_data', f'epoch_{epoch}')
    
    # # 에폭 번호를 포함한 폴더가 존재하지 않는 경우 생성
    # if not os.path.exists(expert_data_epoch_dir):
    #     os.makedirs(expert_data_epoch_dir)

    # # 에폭 번호를 포함한 폴더에 결과를 JSON 파일로 저장
    # expert_filename = os.path.join(expert_data_epoch_dir, f'expert_results_batch_{batch_id}.json')
    # with open(expert_filename, 'w') as f:
    #     json.dump(expert_results, f, indent=4)

    cost, log_likelihood = model(x, epoch, batch_id)

    # Evaluate baseline, get baseline loss if any (only for critic)
    # bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)

    # # Calculate loss
    # reinforce_loss = ((cost - bl_val) * log_likelihood).mean()
    # loss = reinforce_loss + bl_loss

    loss = -log_likelihood.mean() # Doyoung #

    # Perform backward pass and optimization step
    optimizer.zero_grad()
    loss.backward()
    # Clip gradient norms and get (clipped) gradient norms for logging
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()

    wandb_log_dict = {'avg_cost': cost.mean().item(),
                    # 'cum_samples': cum_samples,
                    # 'rl_loss': reinforce_loss.item(),
                    # 'il_loss': distill_loss, 
                    'nll': -log_likelihood.mean().item(),
                    'epoch': epoch,
                    'lr': optimizer.param_groups[0]['lr'],
                    'step': step}

    # Logging
    if opts.wandb != 'disabled':
        wandb.log(wandb_log_dict)


    # # Logging
    # if step % int(opts.log_step) == 0:
    #     log_values(cost, grad_norms, epoch, batch_id, step,
    #                log_likelihood, reinforce_loss, bl_loss, tb_logger, opts)