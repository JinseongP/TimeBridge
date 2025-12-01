import math
import torch
import torch.nn.functional as F
import numpy as np

from torch import nn
from einops import reduce
from tqdm.auto import tqdm
from functools import partial

from Models.bridge_diffusion.transformer import Transformer
from Models.bridge_diffusion.karras_diffusion import KarrasDenoiser, karras_sample, vp_logsnr, vp_logs, get_sigmas_karras, sample_heun
from Models.bridge_diffusion.resample import create_named_schedule_sampler
from Models.bridge_diffusion.model_utils import default, identity, extract, get_dev, exists, butter_lowpass_filter, poly_approx
from engine.solver import generate_imputation_prior
from .nn import mean_flat, append_dims, append_zero
from .img_transformations import DelayEmbedder, STFTEmbedder

class KarrasDenoiserTS(KarrasDenoiser):
    def __init__(
        self,
        sigma_data: float = 0.5, # -1 to 1
        sigma_max=80.0,
        sigma_min=0.002,
        beta_d=2,
        beta_min=0.1,
        cov_xy=0., # 0 for uncorrelated, sigma_data**2 / 2 for  C_skip=1/2 at sigma_max
        rho=7.0,
        seq_length=64,
        weight_schedule="bridge_karras",
        pred_mode='both',
        num_timesteps=40,
        loss_type='l1', # below for Diffusion-TS
        use_ff=True,
        ff_weight=1,
        model_matching = 'F_matching',
    ):
        self.sigma_data = sigma_data
        
        self.sigma_max = sigma_max 
        self.sigma_min = sigma_min 

        self.beta_d = beta_d
        self.beta_min = beta_min
        
        self.sigma_data_end = self.sigma_data
        self.cov_xy = cov_xy
            
        self.c = 1

        self.weight_schedule = weight_schedule
        self.pred_mode = pred_mode

        self.rho = rho
        self.seq_length = seq_length

        self.loss_type = loss_type
        self.use_ff = use_ff
        self.ff_weight = ff_weight
        self.model_matching = model_matching

        if self.pred_mode.startswith('vp'):
            assert self.sigma_max==1.0

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')


    def training_bridge_losses(self, model, x_start, sigmas, target=None, model_kwargs=None, noise=None, vae=None, x_prior=None): 
        if noise is None:
            noise = torch.randn_like(x_start) 
        sigmas = torch.minimum(sigmas, torch.ones_like(sigmas)* self.sigma_max)
        xT = x_prior

        dims = x_start.ndim
        def bridge_sample(x0, xT, t):
            t = append_dims(t, dims)
            # std_t = torch.sqrt(t)* torch.sqrt(1 - t / self.sigma_max)
            if self.pred_mode.startswith('ve'):
                std_t = t* torch.sqrt(1 - t**2 / self.sigma_max**2)
                mu_t= t**2 / self.sigma_max**2 * xT + (1 - t**2 / self.sigma_max**2) * x0
                samples = (mu_t +  std_t * noise )
            elif self.pred_mode.startswith('vp'):
                logsnr_t = vp_logsnr(t, self.beta_d, self.beta_min)
                logsnr_T = vp_logsnr(self.sigma_max, self.beta_d, self.beta_min)
                logs_t = vp_logs(t, self.beta_d, self.beta_min)
                logs_T = vp_logs(self.sigma_max, self.beta_d, self.beta_min)

                a_t = (logsnr_T - logsnr_t +logs_t -logs_T).exp()
                b_t = -torch.expm1(logsnr_T - logsnr_t) * logs_t.exp()
                std_t = (-torch.expm1(logsnr_T - logsnr_t)).sqrt() * (logs_t - logsnr_t/2).exp()
                
                samples= a_t * xT + b_t * x0 + std_t * noise
            return samples
        x_t = bridge_sample(x_start, xT, sigmas)
        model_output, denoised = self.denoise(model, x_t, sigmas, cond=xT, **model_kwargs) 
        weights = self.get_weightings(sigmas)
        model_out = denoised
        train_loss = self.loss_fn(model_out, target, reduction='none')

        fourier_loss = torch.tensor([0.])
        if self.use_ff:
            # print("use fft")
            fft1 = torch.fft.fft(model_out.transpose(1, 2), norm='forward')
            fft2 = torch.fft.fft(target.transpose(1, 2), norm='forward')
            fft1, fft2 = fft1.transpose(1, 2), fft2.transpose(1, 2)
            fourier_loss = self.loss_fn(torch.real(fft1), torch.real(fft2), reduction='none')\
                           + self.loss_fn(torch.imag(fft1), torch.imag(fft2), reduction='none')
            train_loss +=  self.ff_weight * fourier_loss

        train_loss = reduce(train_loss, 'b ... -> b (...)', 'mean')
        
        weights = append_dims((weights), train_loss.ndim)
        train_loss = train_loss * weights

        return train_loss.mean()


    def sigma_inv(self, sigma):
        sigma = torch.as_tensor(sigma)
        return ((self.beta_min ** 2 + 2 * self.beta_d * (1 + sigma ** 2).log()).sqrt() - self.beta_min) / self.beta_d


    def denoise(self, model, x_t, sigmas, cond=None, **model_kwargs): ## x predict
        self.device = get_dev()
        model = model.to(self.device)
        x_t = x_t.to(self.device)
        sigmas = sigmas.to(self.device)

        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_bridge_scalings(sigmas)
        ]

        # if self.pred_mode.startswith('ve'):
        #     c_noise = (0.5 * sigmas).log()
        # elif self.pred_mode.startswith('vp'):
        #     self.M = 1000
        #     c_noise = (self.M - 1) * self.sigma_inv(sigmas)

        c_skip, c_out, c_in = c_skip.to(self.device), c_out.to(self.device), c_in.to(self.device)
        rescaled_t = (1000 * 0.25 * torch.log(sigmas + 1e-44)).to(self.device)

        if self.model_matching == 'D_matching':
            trend, season = model(x_t, sigmas, cond=cond, **model_kwargs)
            model_output = trend + season
            denoised = model_output
            return model_output, denoised
        
        elif self.model_matching == 'F_matching':
            trend, season = model(c_in * x_t, rescaled_t, cond=cond, **model_kwargs)
            model_output = trend + season
            denoised = c_out * model_output + c_skip * x_t
            return model_output, denoised


class Diffusion_TS(nn.Module):
    def __init__(
            self,
            seq_length,
            feature_size,
            n_layer_enc=3,
            n_layer_dec=6,
            d_model=None,
            timesteps=1000,
            sampling_timesteps=None,
            loss_type='l1',
            beta_schedule='real-uniform',
            n_heads=4,
            mlp_hidden_times=4,
            eta=0.,
            attn_pd=0.,
            resid_pd=0.,
            kernel_size=None,
            padding_size=None,
            use_ff=True,
            reg_weight=None,
            sigma_data=0.5, #Below hyperparams for bridge
            sigma_min=0.002,
            sigma_max=80.0,
            beta_d=2,
            beta_min=0.1,
            cov_xy=0.,
            degree=3,
            pred_mode='vp',
            sampler='sde',
            weight_schedule="bridge_karras",
            prior='normal',
            model_matching='F_matching',
            deterministic=False,
            cond_embedding=False,
            rho=7.0,
            use_img_transform=True,
            img_transform_type='delay',
            delay_embed_dim=32,
            delay_time_delay=1,
            stft_n_fft=256,
            stft_hop_length=64,
            **kwargs
    ):
        super(Diffusion_TS, self).__init__()

        self.eta, self.use_ff = eta, use_ff
        self.seq_length = seq_length
        self.feature_size = feature_size
        self.prior = prior
        ff_weight = default(reg_weight, math.sqrt(self.seq_length) / 5)


        # Image transform configuration
        self.use_img_transform = use_img_transform
        self.img_transform_type = img_transform_type
        self.img_embedder = None

        if self.use_img_transform:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Using image transform: {img_transform_type}")

            if img_transform_type == 'delay':
                # if delay_embed_dim >= seq_length:
                #     original_delay_embed_dim = delay_embed_dim
                #     delay_embed_dim = max(4, seq_length // 3)
                #     print(f"⚠️  WARNING: delay_embed_dim ({original_delay_embed_dim}) >= seq_length ({seq_length})")
                #     print(f"   Auto-adjusted to: delay_embed_dim = {delay_embed_dim}")
                
                self.img_embedder = DelayEmbedder(
                    device=device,
                    seq_len=seq_length,
                    delay=delay_time_delay,
                    embedding=delay_embed_dim
                )
                
                num_cols = (seq_length - delay_embed_dim) // delay_time_delay + 1
                model_seq_length = delay_embed_dim * num_cols
                model_feature_size = feature_size
                
                print(f"  Original: ({seq_length}, {feature_size})")
                print(f"  Delay params: n={delay_embed_dim}, m={delay_time_delay}")
                print(f"  Image: ({delay_embed_dim} × {num_cols})")
                print(f"  Flattened: ({model_seq_length}, {model_feature_size})")
                
            elif img_transform_type == 'stft':
                # if stft_n_fft > seq_length:
                #     original_n_fft = stft_n_fft
                #     stft_n_fft = seq_length // 2
                #     print(f"⚠️  WARNING: n_fft ({original_n_fft}) > seq_length ({seq_length})")
                #     print(f"   Auto-adjusted to: n_fft = {stft_n_fft}")
                
                self.img_embedder = STFTEmbedder(
                    device=device,
                    seq_len=seq_length,
                    n_fft=stft_n_fft,
                    hop_length=stft_hop_length
                )
                
                freq_bins = stft_n_fft // 2 + 1
                time_bins = (seq_length - stft_n_fft) // stft_hop_length + 1
                model_seq_length = freq_bins * time_bins
                model_feature_size = feature_size * 2
                
                print(f"  Original: ({seq_length}, {feature_size})")
                print(f"  STFT params: n_fft={stft_n_fft}, hop={stft_hop_length}")
                print(f"  Spectrogram: ({freq_bins} × {time_bins})")
                print(f"  Flattened: ({model_seq_length}, {model_feature_size})")
        else:
            model_seq_length = seq_length
            model_feature_size = feature_size


        # Store transformed dimensions
        self.model_seq_length = model_seq_length
        self.model_feature_size = model_feature_size

        # Adjust conv params for image transform
        if self.use_img_transform:
            print("  Disabling conv_params for image transform")
            kernel_size = None
            padding_size = None

        # Diffusion parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.beta_min = beta_min
        self.beta_d = beta_d
        self.rho = rho
        self.pred_mode = pred_mode
        self.model_matching = model_matching
        self.degree = degree
        self.timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps
        self.deterministic = deterministic
        
        print("sigma with", self.sigma_min, self.sigma_max, self.sigma_data, "beta with", self.beta_min, self.beta_d)
        print("pred with: ", self.pred_mode)
        self.beta_schedule = 'real-uniform'

        # Transformer model
        self.model = Transformer(
            n_feat=model_feature_size,
            n_channel=model_seq_length,
            n_layer_enc=n_layer_enc,
            n_layer_dec=n_layer_dec,
            n_heads=n_heads,
            attn_pdrop=attn_pd,
            resid_pdrop=resid_pd,
            mlp_hidden_times=mlp_hidden_times,
            max_len=model_seq_length,
            n_embd=d_model,
            conv_params=[kernel_size, padding_size],
            cond=cond_embedding,
            **kwargs
        )

        # Diffusion initialization
        self.diffusion = KarrasDenoiserTS(
            sigma_data=sigma_data,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            beta_d=beta_d,
            beta_min=beta_min,
            cov_xy=cov_xy,
            seq_length=model_seq_length,
            weight_schedule=weight_schedule,
            pred_mode=pred_mode,
            loss_type=loss_type,
            use_ff=use_ff,
            ff_weight=ff_weight,
            model_matching=model_matching
        )
        self.sampler = sampler
        
        self.churn_step_ratio = 0
        if self.sampler == 'sde':
            self.churn_step_ratio = 0.33
        print("churn ratio", self.churn_step_ratio)

        self.schedule_sampler = create_named_schedule_sampler(self.beta_schedule, self.diffusion)
    
    
    def _apply_img_transform(self, x):
        """
        Time series to image transformation (STFT / Delay embedding)
        """
        if not self.use_img_transform:
            return x
    
        B, L, C = x.shape
    
        if self.img_transform_type == 'delay':
            if L != self.seq_length:
                print(f"WARNING: Input length {L} != config length {self.seq_length}")
    
            original_seq_len = self.img_embedder.seq_len
            self.img_embedder.seq_len = L
    
            x_img = self.img_embedder.ts_to_img(x, pad=False)
            self.img_embedder.seq_len = original_seq_len
    
            B, C, H, W = x_img.shape
            x_transformed = x_img.permute(0, 2, 3, 1).reshape(B, H * W, C)
    
            expected_length = self.model_seq_length
            actual_length = x_transformed.shape[1]
    
            if actual_length != expected_length:
                if actual_length < expected_length:
                    padding = torch.zeros(B, expected_length - actual_length, C,
                                          device=x.device, dtype=x.dtype)
                    x_transformed = torch.cat([x_transformed, padding], dim=1)
                else:
                    x_transformed = x_transformed[:, :expected_length, :]
    
        elif self.img_transform_type == 'stft':
            import torchaudio.transforms as T
            from .img_transformations import MinMaxArgs
    
            original_seq_len = self.img_embedder.seq_len
            self.img_embedder.seq_len = L
    
            # Flatten and fix dtype/device
            x_flat = x.permute(0, 2, 1).reshape(B * C, L)
            if x_flat.dtype in (torch.float16, torch.bfloat16):
                x_flat = x_flat.to(torch.float32)
            x_flat = x_flat.to(x.device).contiguous()
    
            # Generate STFT
            stft_flat = torch.stft(
                x_flat,
                n_fft=self.img_embedder.n_fft,
                hop_length=self.img_embedder.hop_length,
                center=True,
                return_complex=True
            )
    
            # Process large batches in chunks
            bc = x_flat.shape[0]
            chunk = 4096
            outs = []
            for s in range(0, bc, chunk):
                e = min(s + chunk, bc)
                outs.append(spec(x_flat[s:e]))
            stft_flat = torch.cat(outs, dim=0)  # (B*C, F, T) complex
    
            Fbins, Tbins = stft_flat.shape[1], stft_flat.shape[2]
    
            # Separate real and imaginary parts
            real = stft_flat.real.reshape(B, C, Fbins, Tbins)
            imag = stft_flat.imag.reshape(B, C, Fbins, Tbins)
    
            # Normalize
            if getattr(self.img_embedder, "min_real", None) is not None:
                real = (MinMaxArgs(real,
                                   self.img_embedder.min_real.to(x.device),
                                   self.img_embedder.max_real.to(x.device)) - 0.5) * 2
                imag = (MinMaxArgs(imag,
                                   self.img_embedder.min_imag.to(x.device),
                                   self.img_embedder.max_imag.to(x.device)) - 0.5) * 2
    
            # (6) Real/Imag interleave → (B, 2*C, F, T)
            x_img = torch.stack([real, imag], dim=2).reshape(B, 2 * C, Fbins, Tbins)
    
            self.img_embedder.seq_len = original_seq_len
    
            # (7) Flatten: (B, 2*C, F, T) → (B, F*T, 2*C)
            B, C2, H, W = x_img.shape
            x_transformed = x_img.permute(0, 2, 3, 1).reshape(B, H * W, C2)
    
            # Adjust length
            expected_length = self.model_seq_length
            actual_length = x_transformed.shape[1]
            if actual_length != expected_length:
                if actual_length < expected_length:
                    padding = torch.zeros(
                        x_transformed.shape[0],
                        expected_length - actual_length,
                        x_transformed.shape[2],
                        device=x_transformed.device,
                        dtype=x_transformed.dtype
                    )
                    x_transformed = torch.cat([x_transformed, padding], dim=1)
                else:
                    x_transformed = x_transformed[:, :expected_length, :]
    
        return x_transformed
    
    
    def _inverse_img_transform(self, x):
        """
        Image to time series inverse transform
        """
        if not self.use_img_transform:
            return x
        
        B, HW, C_or_C2 = x.shape
        
        if self.img_transform_type == 'delay':
            # Restore original image size
            H = self.img_embedder.embedding
            W = (self.seq_length - H) // self.img_embedder.delay + 1
            
            actual_HW = H * W
            x_actual = x[:, :actual_HW, :]
            
            C = C_or_C2
            x_img = x_actual.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
            
            # Set img_shape explicitly
            self.img_embedder.img_shape = (B, C, H, W)
            
            x_ts = self.img_embedder.img_to_ts(x_img)
            
            # Adjust to original length
            if x_ts.shape[1] != self.seq_length:
                if x_ts.shape[1] < self.seq_length:
                    padding = torch.zeros(B, self.seq_length - x_ts.shape[1], C,
                                         device=x.device, dtype=x.dtype)
                    x_ts = torch.cat([x_ts, padding], dim=1)
                else:
                    x_ts = x_ts[:, :self.seq_length, :]
                
    def _inverse_img_transform(self, x):
        """
        Image to time series inverse transform
        """
        if not self.use_img_transform:
            return x
    
        B, HW, C_or_C2 = x.shape
    
        if self.img_transform_type == 'delay':
            # ---- Robust H/W inference ----
            L = self.seq_length
            d = int(self.img_embedder.delay)
    
            # First try using config embedding
            H_guess = int(getattr(self.img_embedder, "embedding", 0)) or 0
    
            def valid_HW_pair(H):
                if H <= 0 or H > L:
                    return None
                W = (L - H) // d + 1 if (L - H) >= 0 else 0
                if W <= 0:
                    return None
                return (H, W) if H * W == HW else None
    
            # Try config-based approach
            pair = valid_HW_pair(H_guess) if H_guess else None
    
            # If failed, search for alternative H values
            if pair is None:
                candidates = []
                for H_try in range(1, L + 1):
                    p = valid_HW_pair(H_try)
                    if p is not None:
                        candidates.append(p)
                if not candidates:
                    # Last resort: approximate using HW divisors
                    # Select closest match to condition
                    best = None
                    best_err = 1e9
                    for H_try in range(1, L + 1):
                        if HW % H_try == 0:
                            W_try = HW // H_try
                            W_formula = (L - H_try) // d + 1 if (L - H_try) >= 0 else 0
                            err = abs(W_try - W_formula) + (0 if W_try > 0 else 1e6)
                            if err < best_err:
                                best_err = err
                                best = (H_try, W_try)
                    pair = best
                else:
                    # If multiple candidates, use closest to H_guess
                    if H_guess:
                        pair = min(candidates, key=lambda p: abs(p[0] - H_guess))
                    else:
                        pair = candidates[0]
    
            H, W = pair  # 확정된 (H, W)
    
            # Restore (B, HW, C) -> (B, C, H, W)
            C = C_or_C2
            x_img = x[:, :H * W, :].reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
    
            # Reconstruct time series using overlap-add
            device = x.device
            dtype = x.dtype
    
            recon = torch.zeros(B, L, C, device=device, dtype=dtype)
            weight = torch.zeros(B, L, 1, device=device, dtype=dtype)  # Accumulate weights for overlapping
    
            for i in range(W):
                start = i * d
                end = min(start + H, L)  # Last window should not exceed
                if end <= start:
                    break  # Ignore invalid values
    
                patch_len = end - start  # Actual length to use
                # x_img[:, :, :patch_len, i] : (B, C, patch_len)
                patch = x_img[:, :, :patch_len, i].permute(0, 2, 1)  # (B, patch_len, C)
    
                recon[:, start:end, :] += patch
                weight[:, start:end, :] += 1.0
    
            # Prevent division by zero
            weight = torch.clamp(weight, min=1.0)
            x_ts = recon / weight  # Average overlapping regions
    
            # Adjust length (safety check)
            if x_ts.shape[1] != L:
                if x_ts.shape[1] < L:
                    padding = torch.zeros(B, L - x_ts.shape[1], C, device=device, dtype=dtype)
                    x_ts = torch.cat([x_ts, padding], dim=1)
                else:
                    x_ts = x_ts[:, :L, :]
    
        elif self.img_transform_type == 'stft':
            # Keep existing STFT inverse transform code
            import torchaudio.transforms as T
    
            H = self.img_embedder.n_fft // 2 + 1
            W = (self.seq_length - self.img_embedder.n_fft) // self.img_embedder.hop_length + 1
    
            actual_HW = H * W
            x_actual = x[:, :actual_HW, :]
    
            C2 = C_or_C2
            C = C2 // 2
    
            x_img = x_actual.reshape(B, H, W, C2).permute(0, 3, 1, 2)
    
            x_img_reshaped = x_img.reshape(B, C, 2, H, W)
            real = x_img_reshaped[:, :, 0, :, :]
            imag = x_img_reshaped[:, :, 1, :, :]
    
            min_real = self.img_embedder.min_real.to(x.device)
            max_real = self.img_embedder.max_real.to(x.device)
            min_imag = self.img_embedder.min_imag.to(x.device)
            max_imag = self.img_embedder.max_imag.to(x.device)
    
            unnormalized_real = ((real / 2) + 0.5) * (max_real - min_real) + min_real
            unnormalized_imag = ((imag / 2) + 0.5) * (max_imag - min_imag) + min_imag
    
            real_flat = unnormalized_real.reshape(B * C, H, W)
            imag_flat = unnormalized_imag.reshape(B * C, H, W)
    
            unnormalized_stft = torch.complex(real_flat, imag_flat)
    
            ispec = T.InverseSpectrogram(
                n_fft=self.img_embedder.n_fft,
                hop_length=self.img_embedder.hop_length,
                center=True
            ).to(x.device)
    
            x_ts_flat = ispec(unnormalized_stft, self.seq_length)
            x_ts = x_ts_flat.reshape(B, C, self.seq_length).permute(0, 2, 1)
    
            if x_ts.shape[1] != self.seq_length:
                if x_ts.shape[1] < self.seq_length:
                    padding = torch.zeros(B, self.seq_length - x_ts.shape[1], 
                                         C, device=x_ts.device, dtype=x_ts.dtype)
                    x_ts = torch.cat([x_ts, padding], dim=1)
                else:
                    x_ts = x_ts[:, :self.seq_length, :]
    
        return x_ts


    
        # self.eta, self.use_ff = eta, use_ff
        # self.seq_length = seq_length
        # self.feature_size = feature_size
        # self.prior = prior
        # ff_weight = default(reg_weight, math.sqrt(self.seq_length) / 5)

        # self.sigma_min = sigma_min
        # self.sigma_max = sigma_max
        # self.sigma_data = sigma_data
        # self.beta_min = beta_min
        # self.beta_d = beta_d
        # self.rho = rho
        # self.pred_mode = pred_mode
        # self.model_matching = model_matching
        # self.degree= degree
        # self.timesteps = timesteps ##TODO: not use
        # self.sampling_timesteps = sampling_timesteps 
        # self.deterministic = deterministic
        # print("sigma with", self.sigma_min, self.sigma_max, self.sigma_data, "beta with", self.beta_min, self.beta_d)
        # print("pred with: ",self.pred_mode)

        # self.beta_schedule = 'real-uniform' ##TODO
        # self.model = Transformer(n_feat=feature_size, n_channel=seq_length, n_layer_enc=n_layer_enc, n_layer_dec=n_layer_dec,
        #                          n_heads=n_heads, attn_pdrop=attn_pd, resid_pdrop=resid_pd, mlp_hidden_times=mlp_hidden_times,
        #                          max_len=seq_length, n_embd=d_model, conv_params=[kernel_size, padding_size], cond=cond_embedding, **kwargs)
        # self.diffusion = KarrasDenoiserTS(
        #     sigma_data=sigma_data,
        #     sigma_max=sigma_max,
        #     sigma_min=sigma_min,
        #     beta_d=beta_d,
        #     beta_min=beta_min,
        #     cov_xy=cov_xy,
        #     seq_length=seq_length,
        #     weight_schedule=weight_schedule,
        #     pred_mode=pred_mode,
        #     loss_type = loss_type,
        #     use_ff = use_ff,
        #     ff_weight = ff_weight,
        #     model_matching = model_matching
        # )
        # self.sampler = sampler
        
        # self.churn_step_ratio = 0
        # if self.sampler == 'sde':
        #    self.churn_step_ratio = 0.33    
        # print("churn ratio",self.churn_step_ratio)

        # self.schedule_sampler = create_named_schedule_sampler(self.beta_schedule, self.diffusion)
        
    def sample_prior(self, x, prior_mode='normal', is_sampling=False, x_prior_dataset=None, ind=None):
        ## uncond gen
        if prior_mode=='normal':
            x_prior = torch.randn_like(x)
        elif prior_mode=='uniform':
            x_prior = torch.rand_like(x) * 2 - 1
        elif self.prior=='mean':
            mean, std = x_prior_dataset
            mean, std = mean.to(self.device), std.to(self.device)
            # print('Yeah~~~~~~~~')
            # torch.save(std.detach().cpu(), './std.pt')
            x_ = torch.randn_like(x)
            x_prior = x_ * std + mean

        elif self.prior == 'gp':
            mean, kernels = x_prior_dataset #feature*seq, feature*seq*seq
            mean, kernels = mean.to(self.device), kernels.to(self.device)
            # torch.save(kernels.detach().cpu(), './kernels.pt')
            # print('Yeah~~~~~~~~')
            pd = torch.distributions.multivariate_normal.MultivariateNormal(mean,kernels)
            x_prior = pd.sample(sample_shape = torch.Size([x.shape[0]]))
            x_prior = x_prior.transpose(1,2).float().to(self.device)
            
        elif 'trend' in self.prior or 'impt' in self.prior:
            if is_sampling:
                random_indices = torch.randperm(len(x_prior_dataset))[:x.shape[0]]
                x_prior = x_prior_dataset[random_indices]
            else: # for training
                # ind = ind.to(self.device)
                # x_prior_dataset = x_prior_dataset.to(self.device)
                x_prior = x_prior_dataset[ind]

        ## cond (hard) gen
        elif 'cond' in prior_mode:
            if is_sampling:
                random_indices = torch.randperm(len(x_prior_dataset))[:x.shape[0]]
                x_prior = x_prior_dataset[random_indices].float().to(self.device)
            else:
                if prior_mode=='cond-fixed':
                    p = 0.1
                    mask = torch.rand_like(x) > 1-p
                    x_prior = mask * x
                elif prior_mode=='cond-minmax':
                    batch_size, seq_length, feature_size = x.shape

                    min_values, min_indices = torch.min(x, dim=1)
                    max_values, max_indices = torch.max(x, dim=1)

                    # Create a mask of zeros
                    mask = torch.zeros_like(x, dtype=torch.int64)

                    # Create ranges for the batch indices and feature indices
                    batch_range = torch.arange(batch_size)[:, None]
                    feature_range = torch.arange(feature_size)[None, :]

                    # Use the ranges and the min and max indices to set the corresponding mask values to 1
                    mask[batch_range, min_indices, feature_range] = 1
                    mask[batch_range, max_indices, feature_range] = 1
                    x_prior = mask * x
                elif prior_mode=='cond-ohlc':
                    open_t = x[..., 0]
                    x_prior = open_t.unsqueeze(-1).expand_as(x)
                x_prior = x_prior.float().to(self.device)

        assert x.shape==x_prior.shape
        return x_prior
    

    def forward(self, x, target=None, cond=None, ind=None, prior_info=None, **kwargs):
        b, c, n, device, feature_size, = *x.shape, x.device, self.feature_size
        assert n == feature_size, f'number of variable must be {feature_size}'

        x = self._apply_img_transform(x)
        if target is not None and target.shape == x.shape:
            target = self._apply_img_transform(target)
        else:
            target = x

        
        t, _ = self.schedule_sampler.sample(x.shape[0], device)
        self.device = get_dev()
        # if cond is None:
        cond = self.sample_prior(x=x, prior_mode=self.prior, x_prior_dataset=prior_info, ind=ind)

        self.model_kwargs={}

        # np.save('./x.npy',x.detach().cpu().numpy())
        # np.save('./cond.npy',cond.detach().cpu().numpy())

        # self.model_kwargs['xT'] = cond
        
        compute_losses = partial(
                self.diffusion.training_bridge_losses,
                self.model,
                x,
                sigmas = t,
                target = x,
                x_prior = cond,
                model_kwargs = self.model_kwargs,
                **kwargs
            )
        loss = compute_losses()

        return loss 

    def output(self, x, t, padding_masks=None):
        trend, season = self.model(x, t, padding_masks=padding_masks)
        model_output = trend + season
        return model_output

    def generate_mts(self, batch_size=16, prior_info=None):
        self.device = get_dev()
        self.model.to(self.device)
        # print("0000", prior_info[0].shape)

        if self.use_img_transform:
            if self.img_transform_type == 'delay':
                num_cols = (self.seq_length - self.img_embedder.embedding) // self.img_embedder.delay + 1
                seq_length = self.img_embedder.embedding * num_cols
                feature_size = self.feature_size
            elif self.img_transform_type == 'stft':
                freq_bins = self.img_embedder.n_fft // 2 + 1
                time_bins = (self.seq_length - self.img_embedder.n_fft) // self.img_embedder.hop_length + 1
                seq_length = freq_bins * time_bins
                feature_size = self.feature_size * 2
        else:
            seq_length = self.seq_length
            feature_size = self.feature_size

        shape = (batch_size, seq_length, feature_size)
        x0 = torch.randn(shape, device=self.device)
        y0 = self.sample_prior(x=x0, prior_mode=self.prior, is_sampling=True, x_prior_dataset=prior_info)

        if exists(x0): x0 = x0.to(self.device)
        if exists(y0): y0 = y0.to(self.device)

        self.model_kwargs = {}
        print(f"Sampling time steps {self.sampling_timesteps}")

        sample, path, nfe = karras_sample(
            self.diffusion,
            self.model,
            y0,
            x0,
            steps=self.sampling_timesteps,
            model_kwargs=self.model_kwargs,
            device=self.device,
            clip_denoised=False,
            sampler="heun",
            sigma_min=self.diffusion.sigma_min,
            sigma_max=self.diffusion.sigma_max,
            churn_step_ratio=self.churn_step_ratio,
            rho=self.diffusion.rho,
            guidance=1,
        )
        print(f"sampling with NFE: {nfe}")
        print("1111", sample.shape, y0.shape)
        
        # Image transform inverse
        sample = self._inverse_img_transform(sample)
        y0 = self._inverse_img_transform(y0)
        print("2222", sample.shape, y0.shape)

        
        return sample, y0, path




    def sample_infill(
        self,
        shape, 
        target,
        partial_mask=None,
        clip_denoised=True,
        model_kwargs=None,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        """

        self.device = get_dev()
        self.model.to(self.device)
        
        # Apply transform
        target_transformed = self._apply_img_transform(target)
        if partial_mask is not None:
            partial_mask_transformed = self._apply_img_transform(partial_mask)
        else:
            partial_mask_transformed = None
        
        x0 = torch.randn(target_transformed.shape, device=self.device)
        y0 = target_transformed
        if 'impt' in self.prior:
            y0 = generate_imputation_prior(target_transformed, partial_mask_transformed, prior_mode=self.prior, device=self.device)

        if exists(x0): x0 = x0.to(self.device)
        if exists(y0): y0 = y0.to(self.device)

        self.model_kwargs = {}
        print(f"Sampling time steps {self.sampling_timesteps}")

        sample, path, nfe = karras_sample(
            self.diffusion,
            self.model,
            y0,
            x0,
            steps=self.sampling_timesteps,
            model_kwargs=self.model_kwargs,
            device=self.device,
            clip_denoised=True,
            sampler="heun",
            sigma_min=self.diffusion.sigma_min,
            sigma_max=self.diffusion.sigma_max,
            churn_step_ratio=self.churn_step_ratio,
            rho=self.diffusion.rho,
            guidance=1,
            partial_mask=partial_mask_transformed,
        )
        print(f"sampling with NFE: {nfe}")
        
        # Inverse transform
        sample = self._inverse_img_transform(sample)

        return sample, path



def karras_sample(
    diffusion,
    model,
    x_T,
    x_0,
    steps,
    clip_denoised=True,
    progress=False,
    callback=None,
    model_kwargs=None,
    device=None,
    sigma_min=0.002,
    sigma_max=80,  # higher for highres?
    rho=7.0,
    sampler="heun",
    churn_step_ratio=0.,
    guidance=1,
    partial_mask=None,
):
    assert sampler in ["heun", ], 'only heun sampler is supported currently'

    sigmas = get_sigmas_karras(steps, sigma_min, sigma_max-1e-4, rho, device=device)

    sample_fn = {
        "heun": partial(sample_heun, beta_d=diffusion.beta_d, beta_min=diffusion.beta_min),
    }[sampler]

    sampler_args = dict(
            pred_mode=diffusion.pred_mode, churn_step_ratio=churn_step_ratio, sigma_max=sigma_max
        )
    def denoiser(x_t, sigma, x_T=None):
        _, denoised = diffusion.denoise(model, x_t, sigma, cond=x_T, **model_kwargs)
        
        if clip_denoised:
            denoised = denoised.clamp(-1, 1)
                
        return denoised
    
    x_0, path, nfe = sample_fn(
        denoiser,
        x_T,
        sigmas,
        progress=progress,
        callback=callback,
        guidance=guidance,
        partial_mask=partial_mask,
        **sampler_args,
    )
    # print('nfe:', nfe)

    return x_0.clamp(-1, 1), [x.clamp(-1, 1) for x in path], nfe



if __name__ == '__main__':
    pass
