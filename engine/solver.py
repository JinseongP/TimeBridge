import os
import sys
import time
import torch
import numpy as np
import statsmodels.api as sm

from pathlib import Path
from tqdm.auto import tqdm
from ema_pytorch import EMA
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from Utils.io_utils import instantiate_from_config, get_model_parameters_info
from Models.bridge_diffusion.model_utils import normalize_to_neg_one_to_one, butter_lowpass_filter, poly_approx, gp
from scipy.signal import butter, filtfilt

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

def cycle(dl):
    while True:
        for data in dl:
            yield data

class Trainer(object):
    def __init__(self, config, args, model, dataloader, logger=None):
        super().__init__()
        self.model = model
        self.device = next(self.model.parameters()).device
        self.train_num_steps = config['solver']['max_epochs']
        self.gradient_accumulate_every = config['solver']['gradient_accumulate_every']
        self.save_cycle = config['solver']['save_cycle']
        self.dl = cycle(dataloader['dataloader'])
        self.step = 0
        self.milestone = 0
        self.args = args
        self.logger = logger
        self.results_folder = Path(config['solver']['results_folder'] + f'_{args.name}')
        os.makedirs(self.results_folder, exist_ok=True)

        start_lr = config['solver'].get('base_lr', 1.0e-4)
        ema_decay = config['solver']['ema']['decay']
        ema_update_every = config['solver']['ema']['update_interval']

        self.opt = Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=start_lr, betas=[0.9, 0.96])
        self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every).to(self.device)

        sc_cfg = config['solver']['scheduler']
        sc_cfg['params']['optimizer'] = self.opt
        self.sch = instantiate_from_config(sc_cfg)


        
        if hasattr(self.model, 'img_embedder') and \
           self.model.img_embedder is not None and \
           hasattr(self.model, 'img_transform_type') and \
           self.model.img_transform_type == 'stft':
            print("Initializing STFT min/max parameters...")
            train_data = torch.tensor(dataloader['dataset'].samples).float()
            self.model.img_embedder.cache_min_max_params(train_data)
            print("STFT parameters initialized!")
        
        if self.logger is not None:
            self.logger.log_info(str(get_model_parameters_info(self.model)))
        self.log_frequency = 100

        if args.train and args.sample == 1 and args.mode in ['infill', 'predict']:
            # Imputation의 경우
            original_samples = torch.tensor(dataloader['dataset'].samples).float()
            original_masks = torch.tensor(dataloader['dataset'].masking)
            
            # Image transform 사용 시 변환
            if hasattr(self.model, 'use_img_transform') and self.model.use_img_transform:
                print("Computing imputation prior in TRANSFORMED space...")
                with torch.no_grad():
                    transformed_samples = self.model._apply_img_transform(
                        original_samples.to(self.device)
                    ).cpu()
                    transformed_masks = self.model._apply_img_transform(
                        original_masks.to(self.device)
                    ).cpu()
                
                self.saved_prior = generate_imputation_prior(
                    transformed_samples, 
                    transformed_masks,
                    prior_mode=args.prior, 
                    device=self.device
                )
            else:
                self.saved_prior = generate_imputation_prior(
                    original_samples, 
                    original_masks,
                    prior_mode=args.prior, 
                    device=self.device
                )
        else:
            # Generation의 경우
            original_samples = torch.tensor(dataloader['dataset'].samples).float()
            
            # ===== Image Transform 사용 시 변환된 공간에서 Prior 계산 =====
            if hasattr(self.model, 'use_img_transform') and self.model.use_img_transform:
                print(f"\n{'='*60}")
                print(f"Using Image Transform: {self.model.img_transform_type}")
                print(f"Original data shape: {original_samples.shape}")
                
                # 원본 데이터를 변환된 공간으로
                with torch.no_grad():
                    transformed_samples = self.model._apply_img_transform(
                        original_samples.to(self.device)
                    ).cpu()
                
                print(f"Transformed data shape: {transformed_samples.shape}")
                print(f"Computing prior in TRANSFORMED space...")
                
                # 변환된 공간에서 prior 계산
                self.saved_prior = self.generate_prior(
                    transformed_samples,
                    prior_mode=args.prior
                )
                print(f"Prior computed successfully!")
                print(f"{'='*60}\n")
                
            else:
                # 원본 공간에서 prior 계산
                self.saved_prior = self.generate_prior(
                    original_samples,
                    prior_mode=args.prior
                )
        
        self.path_count = 0

        # if hasattr(self.model, 'img_embedder') and \
        #    self.model.img_embedder is not None and \
        #    self.model.img_transform_type == 'stft':
        #     print("Initializing STFT min/max parameters...")
        #     train_data = torch.tensor(dataloader['dataset'].samples).float()
        #     self.model.img_embedder.cache_min_max_params(train_data)
        #     print("STFT parameters initialized!")


    def generate_prior(self, dataset, prior_mode='normal'):
        ### for uncond
        if prior_mode=='mean':
            mean = torch.tensor(dataset).float().mean(dim=0)
            std = torch.tensor(dataset).float().std(dim=0)
            saved_prior = (mean, std)
        ### for soft cond
        elif prior_mode=='trend': 
            saved_prior = butter_lowpass_filter(dataset, device=self.device)
        elif prior_mode=='trend-poly':
            degree = 3
            saved_prior = poly_approx(dataset, degree=degree, device=self.device)
        elif prior_mode=='trend-linear':
            degree = 1
            saved_prior = poly_approx(dataset, degree=degree, device=self.device)
        elif prior_mode =='gp':
            saved_prior = gp(dataset, kernel_type=self.args.kernel_type, bw=self.args.bw, var=self.args.var, device=self.device)
            
        ### for hard cond
        elif prior_mode=='cond-fixed':
            p = 0.1
            mask = torch.rand_like(dataset) > 1-p
            saved_prior = mask * dataset
        elif prior_mode=='cond-minmax':
            batch_size, seq_length, feature_size = dataset.shape

            min_values, min_indices = torch.min(dataset, dim=1)
            max_values, max_indices = torch.max(dataset, dim=1)

            # Create a mask of zeros
            mask = torch.zeros_like(dataset, dtype=torch.int64)

            # Create ranges for the batch indices and feature indices
            batch_range = torch.arange(batch_size)[:, None]
            feature_range = torch.arange(feature_size)[None, :]

            # Use the ranges and the min and max indices to set the corresponding mask values to 1
            mask[batch_range, min_indices, feature_range] = 1
            mask[batch_range, max_indices, feature_range] = 1
            saved_prior = mask * dataset
        elif prior_mode=='cond-ohlc':
            assert self.args.data=='stock' # only for stock (Open, High, Low, Close, Adj Close, Volume)
            open_t = dataset[..., 0]
            saved_prior = open_t.unsqueeze(-1).expand_as(dataset)

        else:
            saved_prior = None
        return saved_prior



    
    def save(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Save current model to {}'.format(str(self.results_folder / f'checkpoint-{milestone}.pt')))
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'opt': self.opt.state_dict(),
            'prior':self.saved_prior
        }
        torch.save(data, str(self.results_folder / f'checkpoint-{milestone}.pt'))

    def load(self, milestone, verbose=False):
        if self.logger is not None and verbose:
            self.logger.log_info('Resume from {}'.format(str(self.results_folder / f'checkpoint-{milestone}.pt')))
        device = self.device
        data = torch.load(str(self.results_folder / f'checkpoint-{milestone}.pt'), map_location=device)
        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])
        self.saved_prior = data['prior']
        if self.args.pretrain_sampling:
            print("Load saved prior samples")
            self.saved_prior = normalize_to_neg_one_to_one(torch.tensor(np.load(self.args.pretrain_sampling)).float().to(device))
        self.milestone = milestone

    def train(self):
        device = self.device
        step = 0
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('{}: start training...'.format(self.args.name), check_primary=False)

        with tqdm(initial=step, total=self.train_num_steps) as pbar:
            while step < self.train_num_steps:
                epoch_start = time.time()  # Epoch 시작
                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    if self.args.mode in ['infill', 'predict']:
                        (x, t_m, ind) = next(self.dl)
                        data, t_m, ind = x.to(device), t_m.to(device), ind.to(device)
                        cond = data*t_m
                    else:
                        data, ind = next(self.dl)
                        data, ind = data.to(device), ind.to(device)
                        cond = None

                    loss = self.model(data, target=data, cond=cond, ind=ind, prior_info=self.saved_prior)
                    loss = loss / self.gradient_accumulate_every
                    loss.backward()
                    # check_model_parameters_and_gradients(self.model)
                    total_loss += loss.item()
                
                pbar.set_description(f'loss: {total_loss:.6f}')
                
                clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                if "ReduceLROnPlateauWithWarmup" in str(self.sch.__class__):
                    self.sch.step(total_loss)
                else:
                    self.sch.step()
                self.opt.zero_grad()
                self.step += 1
                step += 1
                self.ema.update()
                # check_weights_update(self.model, initial_weights)
                with torch.no_grad():
                    if self.step != 0 and self.step % self.save_cycle == 0:
                        self.milestone += 1
                        if self.milestone == 10: ##TODO: delete
                            self.save(self.milestone)
                        # self.logger.log_info('saved in {}'.format(str(self.results_folder / f'checkpoint-{self.milestone}.pt')))
                    
                    if self.logger is not None and self.step % self.log_frequency == 0:
                        # info = '{}: train'.format(self.args.name)
                        # info = info + ': Epoch {}/{}'.format(self.step, self.train_num_steps)
                        # info += ' ||'
                        # info += '' if loss_f == 'none' else ' Fourier Loss: {:.4f}'.format(loss_f.item())
                        # info += '' if loss_r == 'none' else ' Reglarization: {:.4f}'.format(loss_r.item())
                        # info += ' | Total Loss: {:.6f}'.format(total_loss)
                        # self.logger.log_info(info)
                        self.logger.add_scalar(tag='train/loss', scalar_value=total_loss, global_step=self.step)

                pbar.update(1)
            print(f"\n{'='*60}")
            print(f"{'='*60}\n")

        print('training complete',f'loss: {total_loss:.6f}')
        if self.logger is not None:
            self.logger.log_info('Training done, time: {:.2f}'.format(time.time() - tic))

    def sample(self, num, size_every, shape=None):
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to sample...')
        samples = np.empty([0, shape[0], shape[1]])
        priors = np.empty([0, shape[0], shape[1]])
        num_cycle = int(num // size_every) + 1

        for idx in range(num_cycle):
            sample, prior, path = self.ema.ema_model.generate_mts(batch_size=size_every, prior_info=self.saved_prior)
            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            priors = np.row_stack([priors, prior.detach().cpu().numpy()])
            torch.cuda.empty_cache()
            
            # if idx==0: ##FOR VISUALIZATION
            #     path = [p.detach().cpu().numpy() for p in path]
            #     np.save(f'./visualization/path_{self.args.prior}_{self.path_count}.npy',np.array(path))
            #     self.path_count += 1

        if self.logger is not None:
            self.logger.log_info('Sampling done, time: {:.2f}, with samples {}'.format(time.time() - tic, samples.shape))
        return samples, priors

    def restore(self, raw_dataloader, shape=None, coef=1e-1, stepsize=1e-1, sampling_steps=50):
        if self.logger is not None:
            tic = time.time()
            self.logger.log_info('Begin to restore...')
        model_kwargs = {}
        model_kwargs['coef'] = coef
        model_kwargs['learning_rate'] = stepsize
        samples = np.empty([0, shape[0], shape[1]])
        reals = np.empty([0, shape[0], shape[1]])
        masks = np.empty([0, shape[0], shape[1]])

        for idx, (x, t_m, ind) in enumerate(raw_dataloader):
            x, t_m = x.to(self.device), t_m.to(self.device)
            sample, path = self.ema.ema_model.sample_infill(shape=x.shape, target=x*t_m, partial_mask=t_m, model_kwargs=model_kwargs)
            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            reals = np.row_stack([reals, x.detach().cpu().numpy()])
            masks = np.row_stack([masks, t_m.detach().cpu().numpy()])

            # if idx==0:
            #     path = [p.detach().cpu().numpy() for p in path]
            #     np.save(f'./visualization/path_restore_{self.args.prior}_{self.path_count}.npy',np.array(path))
            #     self.path_count += 1


        
        if self.logger is not None:
            self.logger.log_info('Imputation done, time: {:.2f}'.format(time.time() - tic))
        return samples, reals, masks
        # return samples

import torch
import numpy as np
from numba import jit, prange

@jit(nopython=True, parallel=True, fastmath=True)
def _interpolate_numba(data, mask, prior_mode, fill_end_mode):
    batch_size, seq_len, n_features = data.shape
    output = np.copy(data)
    
    for b in prange(batch_size):
        for f in range(n_features):
            masked_idx = np.where(mask[b, :, f] == 1)[0]
            
            if len(masked_idx) < 2:
                continue
            
            for j in range(len(masked_idx) - 1):
                start_idx = masked_idx[j]
                end_idx = masked_idx[j + 1]
                
                if end_idx - start_idx <= 1:
                    continue
                
                if prior_mode == 0:
                    start_val = data[b, start_idx, f]
                    end_val = data[b, end_idx, f]
                    
                    for k in range(start_idx + 1, end_idx):
                        alpha = (k - start_idx) / (end_idx - start_idx)
                        output[b, k, f] = start_val * (1 - alpha) + end_val * alpha
                
                elif prior_mode == 1:
                    if j + 2 < len(masked_idx):
                        x0, x1, x2 = masked_idx[j], masked_idx[j + 1], masked_idx[j + 2]
                        y0, y1, y2 = data[b, x0, f], data[b, x1, f], data[b, x2, f]
                        
                        for k in range(start_idx + 1, end_idx):
                            t = k
                            L0 = ((t - x1) * (t - x2)) / ((x0 - x1) * (x0 - x2))
                            L1 = ((t - x0) * (t - x2)) / ((x1 - x0) * (x1 - x2))
                            L2 = ((t - x0) * (t - x1)) / ((x2 - x0) * (x2 - x1))
                            output[b, k, f] = y0 * L0 + y1 * L1 + y2 * L2
                    else:
                        start_val = data[b, start_idx, f]
                        end_val = data[b, end_idx, f]
                        for k in range(start_idx + 1, end_idx):
                            alpha = (k - start_idx) / (end_idx - start_idx)
                            output[b, k, f] = start_val * (1 - alpha) + end_val * alpha
            
            first_idx = masked_idx[0]
            last_idx = masked_idx[-1]
            
            if fill_end_mode == 0:
                if first_idx > 0:
                    output[b, :first_idx, f] = data[b, first_idx, f]
                if last_idx < seq_len - 1:
                    output[b, last_idx + 1:, f] = data[b, last_idx, f]
            
            elif fill_end_mode == 1 and len(masked_idx) > 1:
                if first_idx > 0:
                    second_idx = masked_idx[1]
                    slope = (data[b, second_idx, f] - data[b, first_idx, f]) / (second_idx - first_idx)
                    for k in range(first_idx):
                        output[b, k, f] = data[b, first_idx, f] + (k - first_idx) * slope
                
                if last_idx < seq_len - 1:
                    prev_idx = masked_idx[-2]
                    slope = (data[b, last_idx, f] - data[b, prev_idx, f]) / (last_idx - prev_idx)
                    for k in range(last_idx + 1, seq_len):
                        output[b, k, f] = data[b, last_idx, f] + (k - last_idx) * slope
    
    return output


def generate_imputation_prior(dataset, mask, prior_mode='impt-linear', fill_end='nearest', device='cpu'):
    assert mask.shape == dataset.shape
    
    data_np = (mask * dataset).cpu().numpy()
    mask_np = mask.cpu().numpy()
    
    mode_map = {'impt-linear': 0, 'impt-quadratic': 1}
    if prior_mode not in mode_map:
        raise ValueError(f"Unknown prior_mode: {prior_mode}")
    
    prior_mode_int = mode_map[prior_mode]
    fill_end_mode = 0 if fill_end == 'nearest' else 1
    
    result_np = _interpolate_numba(data_np, mask_np, prior_mode_int, fill_end_mode)
    
    return torch.from_numpy(result_np).to(device).float()


# def generate_imputation_prior(dataset, mask, prior_mode='impt-linear', fill_end='nearest', device='cpu'):
#     # Handling different types of mask input
#     assert mask.shape == dataset.shape  # Ensuring mask shape

#     if prior_mode=='impt-linear':
#         dataset = torch.swapaxes(dataset,1,2)
#         mask = torch.swapaxes(mask,1,2)
#         # Calculate the prior with mask applied
#         tmp_prior = mask * dataset
#         batch_size, n, m = dataset.shape
#         saved_prior = torch.zeros_like(tmp_prior)
#         # Processing each batch and row
#         for b in range(batch_size):
#             for i in range(n):
#                 masked_indices = torch.where(mask[b, i] == 1)[0]
#                 if len(masked_indices) < 2:
#                     saved_prior[b, i] = tmp_prior[b, i]
#                     continue
    
#                 # Copy the masked values directly
#                 saved_prior[b, i, masked_indices] = tmp_prior[b, i, masked_indices]
    
#                 # Interpolate between each pair of masked indices
#                 for j in range(len(masked_indices) - 1):
#                     start_idx = masked_indices[j]
#                     end_idx = masked_indices[j + 1]
#                     start_val = tmp_prior[b, i, start_idx]
#                     end_val = tmp_prior[b, i, end_idx]
#                     num_elements = end_idx - start_idx - 1
    
#                     if num_elements > 0:
#                         interpolated_values = torch.linspace(start_val, end_val, num_elements + 2)[1:-1]
#                         saved_prior[b, i, start_idx + 1:end_idx] = interpolated_values
    
#                 # Handling 'nearest' fill
#                 if fill_end == 'nearest':
#                     if masked_indices[0] > 0:
#                         saved_prior[b, i, :masked_indices[0]] = tmp_prior[b, i, masked_indices[0]]
#                     if masked_indices[-1] < m - 1:
#                         saved_prior[b, i, masked_indices[-1] + 1:] = tmp_prior[b, i, masked_indices[-1]]
                
#                 # Handling 'linear' fill
#                 elif fill_end == 'linear':
#                     if masked_indices[0] > 0:
#                         first_val = tmp_prior[b, i, masked_indices[0]]
#                         if len(masked_indices) > 1:
#                             first_step = (tmp_prior[b, i, masked_indices[1]] - first_val) / (masked_indices[1] - masked_indices[0])
#                             saved_prior[b, i, :masked_indices[0]] = torch.arange(-masked_indices[0], 0, dtype=torch.float32) * first_step + first_val
                
#                     if masked_indices[-1] < m - 1:
#                         last_val = tmp_prior[b, i, masked_indices[-1]]
#                         last_idx = masked_indices[-1]
#                         if len(masked_indices) > 1 and masked_indices[-2] < last_idx:  # Ensure there's a previous point and it's not the same as last_idx
#                             last_step = (last_val - tmp_prior[b, i, masked_indices[-2]]) / (last_idx - masked_indices[-2])
#                         else:
#                             last_step = 0  # Fallback if no valid
#                         extension_range = m - last_idx - 1  # Corrected to ensure it matches the slicing range
#                         if extension_range > 0:  # Protect against negative or zero ranges
#                             saved_prior[b, i, last_idx + 1:] = torch.arange(1, extension_range + 1, dtype=torch.float32) * last_step + last_val

#     elif prior_mode=='impt-gp':
#         dataset = torch.swapaxes(dataset,1,2)
#         mask = torch.swapaxes(mask,1,2)
#         # Calculate the prior with mask applied
#         tmp_prior = mask * dataset
#         batch_size, n, m = dataset.shape
#         saved_prior = torch.zeros_like(tmp_prior)
#         # Processing each batch and row
#         for b in range(batch_size):
#             for i in range(n):
#                 masked_indices = torch.where(mask[b, i] == 1)[0]
#                 if len(masked_indices) < 2:
#                     saved_prior[b, i] = tmp_prior[b, i]
#                     continue
    
#                 # Copy the masked values directly
#                 saved_prior[b, i, masked_indices] = tmp_prior[b, i, masked_indices]
    
#                 # Interpolate between each pair of masked indices
#                 for j in range(len(masked_indices) - 1):
#                     start_idx = masked_indices[j]
#                     end_idx = masked_indices[j + 1]
#                     start_val = tmp_prior[b, i, start_idx]
#                     end_val = tmp_prior[b, i, end_idx]
#                     num_elements = end_idx - start_idx - 1
    
#                     if num_elements > 0:
#                         interpolated_values = torch.linspace(start_val, end_val, num_elements + 2)[1:-1]
#                         saved_prior[b, i, start_idx + 1:end_idx] = interpolated_values
    
#                 # Handling 'nearest' fill
#                 if fill_end == 'nearest':
#                     if masked_indices[0] > 0:
#                         saved_prior[b, i, :masked_indices[0]] = tmp_prior[b, i, masked_indices[0]]
#                     if masked_indices[-1] < m - 1:
#                         saved_prior[b, i, masked_indices[-1] + 1:] = tmp_prior[b, i, masked_indices[-1]]
                
#                 # Handling 'linear' fill
#                 elif fill_end == 'linear':
#                     if masked_indices[0] > 0:
#                         first_val = tmp_prior[b, i, masked_indices[0]]
#                         if len(masked_indices) > 1:
#                             first_step = (tmp_prior[b, i, masked_indices[1]] - first_val) / (masked_indices[1] - masked_indices[0])
#                             saved_prior[b, i, :masked_indices[0]] = torch.arange(-masked_indices[0], 0, dtype=torch.float32) * first_step + first_val
                
#                     if masked_indices[-1] < m - 1:
#                         last_val = tmp_prior[b, i, masked_indices[-1]]
#                         last_idx = masked_indices[-1]
#                         if len(masked_indices) > 1 and masked_indices[-2] < last_idx:  # Ensure there's a previous point and it's not the same as last_idx
#                             last_step = (last_val - tmp_prior[b, i, masked_indices[-2]]) / (last_idx - masked_indices[-2])
#                         else:
#                             last_step = 0  # Fallback if no valid
#                         extension_range = m - last_idx - 1  # Corrected to ensure it matches the slicing range
#                         if extension_range > 0:  # Protect against negative or zero ranges
#                             saved_prior[b, i, last_idx + 1:] = torch.arange(1, extension_range + 1, dtype=torch.float32) * last_step + last_val
    
#     saved_prior = torch.swapaxes(saved_prior,1,2)
#     saved_prior = saved_prior.to(device).float()
#     return saved_prior


