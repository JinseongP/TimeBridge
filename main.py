import os
import torch
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

from engine.logger import Logger
from engine.solver import Trainer
from Data.build_dataloader import build_dataloader, build_dataloader_cond, build_dataloader_train_cond
from Models.bridge_diffusion.model_utils import unnormalize_to_zero_to_one
from Utils.io_utils import load_yaml_config, seed_everything, merge_opts_to_config, instantiate_from_config
from Utils.context_fid import Context_FID
from Utils.metric_utils import display_scores
from Utils.cross_correlation import CrossCorrelLoss, random_choice
from Utils.discriminative_metric import discriminative_score_metrics
from Utils.predictive_metric import predictive_score_metrics
from Utils.conditional_metric import perc_error_distance, satisfaction_rate

import warnings
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Training Script')
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--data', type=str, default=None)

    # parser.add_argument('--config_file', type=str, default=None, 
    #                     help='path of config file')
    parser.add_argument('--output', type=str, default='OUTPUT', 
                        help='directory to save the results')
    parser.add_argument('--tensorboard', action='store_true', 
                        help='use tensorboard for logging')

    # args for random

    parser.add_argument('--cudnn_deterministic', action='store_true', default=False,
                        help='set cudnn.deterministic True')
    parser.add_argument('--seed', type=int, default=12345, 
                        help='seed for initializing training.')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU id to use. If given, only the specific gpu will be'
                        ' used, and ddp will be disabled')
    
    # args for training
    parser.add_argument('--train', action='store_true', default=False, help='Train or Test.')
    parser.add_argument('--sample', type=int, default=0, 
                        choices=[0, 1], help='Condition or Uncondition.')
    parser.add_argument('--mode', type=str, default='infill',
                        help='Infilling or Forecasting.')
    parser.add_argument('--milestone', type=int, default=10)

    parser.add_argument('--missing_ratio', type=float, default=0., help='Ratio of Missing Values.')
    parser.add_argument('--pred_len', type=int, default=0, help='Length of Predictions.')

    # args for metric evaluation
    parser.add_argument('--eval', action='store_true', default=False, help='Metric evaluation or not.')
    parser.add_argument('--iteration', type=int, default=5, help='Iteration.')
    
    # args for modify config
    parser.add_argument('opts', help='Modify config options using the command-line',
                        default=None, nargs=argparse.REMAINDER)  
    # args for bridge
    parser.add_argument('--bridge_karras', action='store_true', default=False, help='Bridge karras or not.') ##TODO: delete
    parser.add_argument('--pred_mode', type=str, default='vp',
                        help='vp or ve.') 
    parser.add_argument('--sampler', type=str, default='sde',
                        help='sde or ode.')
    parser.add_argument('--prior', type=str, default='normal',
                        help='normal / uniform / trend.')
    parser.add_argument('--pretrain', action='store_true', default=False, 
                        help='Pretrain for bridge or not.')
    parser.add_argument('--pretrain_sampling', type=str, default=None,
                        help='Sampling from pretrained or not.')
    parser.add_argument('--config_suffix', type=str, default="",
                        help='Any suffix for config')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--result_csv', type=str, default="./result.csv")

    # args for DDBM
    parser.add_argument('--sigma_min', type=float, default=0.0001)
    parser.add_argument('--sigma_max', type=float, default=1.0)
    parser.add_argument('--sigma_data', type=float, default=0.5)

    parser.add_argument('--beta_min', type=float, default=0.1)
    parser.add_argument('--beta_d', type=float, default=2.0)

    parser.add_argument('--model_matching', type=str, default="D_matching")
    parser.add_argument('--proportion', type=float, default=1.0)
    parser.add_argument('--cond_embedding', action='store_true', default=False, help='Conditional info on embedding')

    # args for prior
    parser.add_argument('--var', type=float, default=1)
    parser.add_argument('--bw', type=str, default='GAUSSIAN')
    parser.add_argument('--kernel_type', type=str, default='rbf')
    
    # transform
    parser.add_argument('--use_img_transform', default= False, 
                    help='Use image transformation (Delay Embedding or STFT)')
    parser.add_argument('--img_transform_type', type=str, default='delay', 
                    choices=['delay', 'stft'],
                    help='Type of image transformation: delay or stft')

    # Delay Embedding parameters
    parser.add_argument('--delay_embed_dim', type=int, default=32,
                        help='Embedding dimension for Delay Embedding')
    parser.add_argument('--delay_time_delay', type=int, default=1,
                        help='Time delay for Delay Embedding')
    
    # STFT parameters
    parser.add_argument('--stft_n_fft', type=int, default=256,
                        help='FFT size for STFT')
    parser.add_argument('--stft_hop_length', type=int, default=64,
                        help='Hop length for STFT')
                        
    args = parser.parse_args()
    args.save_dir = os.path.join(args.output, f'{args.name}')

    return args

def main():
    args = parse_args()

    if args.seed is not None:
        seed_everything(args.seed)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
    path = f'./Config/{args.data}.yaml' #    path = f'./Config/{args.data}{args.config_suffix}.yaml'
    config = load_yaml_config(path)
    config = merge_opts_to_config(config, args.opts)

    ## TODO
    config['model']['target'] = 'Models.bridge_diffusion.bridge_karras_diffusion.Diffusion_TS'
    config['model']['params']['pred_mode'] = args.pred_mode
    config['model']['params']['sampler'] = args.sampler
    config['model']['params']['prior'] = args.prior
    config['model']['params']['beta_min'] = args.beta_min
    config['model']['params']['beta_d'] = args.beta_d
    config['model']['params']['sigma_min'] = args.sigma_min
    config['model']['params']['sigma_max'] = args.sigma_max
    config['model']['params']['sigma_data'] = args.sigma_data
    config['model']['params']['var'] = args.var
    config['model']['params']['bw'] = args.bw
    config['model']['params']['model_matching'] = args.model_matching
    config['model']['params']['cond_embedding'] = args.cond_embedding
    if 'proportion' in config['dataloader']['train_dataset']['params'].keys():
        config['dataloader']['train_dataset']['params']['proportion'] = args.proportion
    if hasattr(args, 'use_img_transform'):
        config['model']['params']['use_img_transform'] = args.use_img_transform
    if hasattr(args, 'img_transform_type'):
        config['model']['params']['img_transform_type'] = args.img_transform_type
    if hasattr(args, 'delay_embed_dim'):
        config['model']['params']['delay_embed_dim'] = args.delay_embed_dim
    if hasattr(args, 'delay_time_delay'):
        config['model']['params']['delay_time_delay'] = args.delay_time_delay
    if hasattr(args, 'stft_n_fft'):
        config['model']['params']['stft_n_fft'] = args.stft_n_fft
    if hasattr(args, 'stft_hop_length'):
        config['model']['params']['stft_hop_length'] = args.stft_hop_length


    base_lr = config['solver']['base_lr']
    min_lr = config['solver']['scheduler']['params']['min_lr']
    warmup_lr = config['solver']['scheduler']['params']['warmup_lr']

    config['solver']['base_lr'] = args.lr
    config['solver']['scheduler']['params']['min_lr'] = min_lr * (args.lr / base_lr)
    config['solver']['scheduler']['params']['warmup_lr'] = warmup_lr * (args.lr / base_lr)

    print(f'lr with {args.lr}')
    
    logger = Logger(args)
    logger.save_config(config)

    model = instantiate_from_config(config['model']).cuda()
    if args.sample == 1 and args.mode in ['infill', 'predict']:
        config['dataloader']['train_dataset']['params']['period'] = "train-cond"
        test_dataloader_info = build_dataloader_cond(config, args)
        dataloader_info = build_dataloader_train_cond(config, args)
    else:
        dataloader_info = build_dataloader(config, args)
    trainer = Trainer(config=config, args=args, model=model, dataloader=dataloader_info, logger=logger)
    

    if args.train:
        trainer.train()
    elif args.sample == 1 and args.mode in ['infill', 'predict']:
        trainer.load(args.milestone)
        dataloader, dataset = test_dataloader_info['dataloader'], test_dataloader_info['dataset']
        coef = config['dataloader']['test_dataset']['coefficient']
        stepsize = config['dataloader']['test_dataset']['step_size']
        sampling_steps = config['dataloader']['test_dataset']['sampling_steps']


        mse_01_score = []
        mse_score = []
        mae_score = []
        mre_score = []

        for i in range(args.iteration):
            samples, reals, masks = trainer.restore(dataloader, [dataset.window, dataset.var_num], coef, stepsize, sampling_steps)
            if dataset.auto_norm:
                samples = unnormalize_to_zero_to_one(samples)
                reals  = unnormalize_to_zero_to_one(reals)
            np.save(os.path.join(args.save_dir, f'ddpm_{args.mode}_{args.name}_{i}.npy'), samples)
            if i==0:
                np.save(os.path.join(args.save_dir, f'real_{args.mode}_{args.name}.npy'), reals)
                np.save(os.path.join(args.save_dir, f'mask_{args.mode}_{args.name}.npy'), masks)

            if args.data == 'sines':
                rescaled_reals = reals
                rescaled_samples = samples
            else:
                print(reals.reshape(-1, reals.shape[-1]).shape)
                rescaled_reals = dataloader_info['dataset'].scaler.inverse_transform(reals.reshape(-1, reals.shape[-1])).reshape(reals.shape)
                rescaled_samples = dataloader_info['dataset'].scaler.inverse_transform(samples.reshape(-1, samples.shape[-1])).reshape(samples.shape)

            masks = masks.astype(bool)
            mse_01 = mean_squared_error(samples[~masks], reals[~masks])
            mse_01_score.append(mse_01)

            mse = mean_squared_error(rescaled_samples[~masks], rescaled_reals[~masks])
            mse_score.append(mse)
            mae = mean_absolute_error(rescaled_samples[~masks], rescaled_reals[~masks])
            mae_score.append(mae)
            mre = mean_absolute_percentage_error(rescaled_samples[~masks], rescaled_reals[~masks])
            mre_score.append(mre)

        print('mean_squared_error_01_scale', end='')
        mean1, std1 = display_scores(mse_01_score)
        print('mean_squared_error', end='')
        mean2, std2 = display_scores(mse_score)
        print('mean_absolute_error', end='')
        mean3, std3 = display_scores(mae_score)
        print('mean_absolute_percentage_error', end='')
        mean4, std4 = display_scores(mre_score)
                            
    
        df = pd.DataFrame({'name': [args.name], 'data' : [args.data], 'prior' : [args.prior], \
                           'mean_squared_error_01_scale': [np.round(np.mean(mean1), 3)], \
                           'mean_squared_error': [np.round(np.mean(mean2), 3)], \
                           'mean_absolute_error': [np.round(np.mean(mean3), 3)], \
                           'mean_absolute_percentage_error': [np.round(np.mean(mean4), 3)], \
                           'mean_squared_error_01_scale-std': [np.round(std1, 3)], \
                           'mean_squared_error-std': [np.round(std2, 3)], \
                           'mean_absolute_error-std': [np.round(std3, 3)], \
                           'mean_absolute_percentage_error-std': [np.round(std4, 3)],
                          })
        
             
    else:
        trainer.load(args.milestone)
        dataset = dataloader_info['dataset']
        # for i in range(args.iteration): ## For visualization
        samples, priors = trainer.sample(num=len(dataset), size_every=2001, shape=[dataset.window, dataset.var_num])
        if dataset.auto_norm:
            samples = unnormalize_to_zero_to_one(samples)
            priors =  unnormalize_to_zero_to_one(priors)
        np.save(os.path.join(args.save_dir, f'ddpm_fake_{args.name}.npy'), samples)
        np.save(os.path.join(args.save_dir, f'prior_fake_{args.name}.npy'), priors)

    if args.eval:
        context_fid_score = []
        correlational_score = []
        discriminative_score = []
        predictive_score = []

        if args.data == 'sines':
            ori_data = np.load(os.path.join(args.save_dir, f"samples/sine_ground_truth_{dataset.window}_train.npy"))
        elif args.data =='fmri':
            ori_data = np.load(os.path.join(args.save_dir, f"samples/fMRI_norm_truth_{dataset.window}_train.npy")) 
        else:
            ori_data = np.load(os.path.join(args.save_dir, f"samples/{args.data}_norm_truth_{dataset.window}_train.npy"))  
        #ori_data = np.load(os.path.join(f'toy_exp/samples/{args.data}_ground_truth_24_train.npy'))
        x_real = torch.from_numpy(ori_data)
        x_fake = torch.from_numpy(samples)
        size = int(x_real.shape[0] / args.iteration)

        
        for i in range(args.iteration):
            print(f'Iter {i}', '\n')
            #Context_FID_Score
            context_fid = Context_FID(ori_data[:], samples[:ori_data.shape[0]])
            context_fid_score.append(context_fid)
            print('context-fid =', context_fid, '\n')
            
            #Correlation
            real_idx = random_choice(x_real.shape[0], size)
            fake_idx = random_choice(x_fake.shape[0], size)
            corr = CrossCorrelLoss(x_real[real_idx, :, :], name='CrossCorrelLoss')
            loss = corr.compute(x_fake[fake_idx, :, :])
            correlational_score.append(loss.item())
            print('cross-correlation =', loss.item(), '\n')
                
            if "BEST" in args.name or "BEST" in args.config_suffix:
                #Discriminative_score
                temp_disc, fake_acc, real_acc = discriminative_score_metrics(ori_data[:], samples[:ori_data.shape[0]])
                discriminative_score.append(temp_disc)
                print('discriminative_score =', temp_disc, 'with fake/real', fake_acc, real_acc,'\n')

                #Predictive_score
                predictive = predictive_score_metrics(ori_data[:], samples[:ori_data.shape[0]])
                predictive_score.append(predictive)
                print('predictive_score =', predictive, '\n')             

            else:
                temp_disc = 0
                discriminative_score.append(temp_disc)
                predictive = 0
                predictive_score.append(predictive)

        print('Context-Fid ', end='')
        mean1, std1 = display_scores(context_fid_score)
        print('Cross-Correlation ', end='')
        mean2, std2 = display_scores(correlational_score)
        print('Discriminative ', end='')
        mean3, std3 = display_scores(discriminative_score)
        print('Predictive ', end='')
        mean4, std4 = display_scores(predictive_score)
                            
    
        df = pd.DataFrame({'name': [args.name], 'data' : [args.data], 'prior' : [args.prior], \
                           'Context-Fid': [np.round(np.mean(context_fid_score), 3)], \
                           'Cross-Correlation': [np.round(np.mean(correlational_score), 3)], \
                           'Discriminative': [np.round(np.mean(discriminative_score), 3)], \
                           'Predictive': [np.round(np.mean(predictive_score), 3)], \
                           'Context-Fid-std': [np.round(std1, 3)], \
                           'Cross-Correlation-std': [np.round(std2, 3)], \
                           'Discriminative-std': [np.round(std3, 3)], \
                           'Predictive-std': [np.round(std4, 3)],
                          })
        
        if not os.path.exists(args.result_csv):
            df.to_csv(args.result_csv, index=False, mode='w', encoding='utf-8-sig')
        else:
            df.to_csv(args.result_csv, index=False, mode='a', encoding='utf-8-sig', header=False)
            
       

if __name__ == '__main__':
    main()