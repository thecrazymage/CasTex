import torch
import torch.nn.functional as F
from diffusers import DiffusionPipeline

class SDSLoss(torch.nn.Module):
    def __init__(
        self,
        stage=None,
        model_name=None,
        min_timestamp=200,
        max_timestamp=900,
        device='cuda:0'
    ):
        """Initializes the diffusion model pipeline, scheduler, and UNet for the specific training stage."""
        super().__init__()

        self.stage = stage
        self.device = device
        self.resolution = (64, 64) if stage == 'i' else (256, 256)

        pipe = DiffusionPipeline.from_pretrained(
            model_name,
            text_encoder=None,
            safety_checker=None,
            watermarker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            variant="fp16",
            torch_dtype=torch.float16,
        ).to(self.device)
        self.unet = pipe.unet.eval()
        self.scheduler = pipe.scheduler
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)

    @torch.amp.autocast('cuda', enabled=False)
    def forward_unet(self, latents, t, encoder_hidden_states, **kwargs):
        """Executes a forward pass of the UNet model with mixed-precision handling."""
        return self.unet(
            latents.to(torch.float16),
            t.to(torch.float16),
            encoder_hidden_states.to(torch.float16),
            **kwargs
        ).sample.to(latents.dtype)

    def prepare_latents(self, images):
        """Resizes and normalizes input rendered images to match the diffusion model's expected latent format."""
        latents = F.interpolate(
            images,
            self.resolution,
            mode="bilinear",
            align_corners=False,
            antialias=True
        )
        return 2.0 * latents - 1.0
    
    def prepare_condition(self, images, lowres_noise_level):
        """
            Processes conditioning images for the second stage 
            by performing downscaling, upscaling, and noise injection.
        """
        downscaled = F.interpolate(images, (64, 64), mode="nearest")
        upscaled = F.interpolate(downscaled, (256, 256), mode="nearest")
        upscaled = 2.0 * upscaled - 1.0
        upscaled = self.scheduler.add_noise(
            upscaled,
            torch.randn_like(upscaled),
            torch.tensor(int(self.num_train_timesteps * lowres_noise_level))
        )
        return upscaled

    def predict_noise(self, latents_noisy, t, prompt_embeddings, guidance_scale, condition, lowres_noise_level):
        """
            Predicts the noise residual for the latents 
            using the diffusion model and applies classifier-free guidance.
        """
        batch_size = latents_noisy.shape[0]

        kwargs = {}
        if self.stage == 'ii':
            condition = self.prepare_condition(condition, lowres_noise_level)
            condition = self.scheduler.scale_model_input(condition, t)
            noise_level = torch.full(
                    [2 * batch_size],
                    torch.tensor(int(self.num_train_timesteps * lowres_noise_level)),
                    device=self.device
            )
            latents_noisy = torch.cat([latents_noisy, condition], dim=1)
            kwargs = {'class_labels' : noise_level}

        # conditional/unconditional prediction
        latent_model_input = torch.cat([latents_noisy] * 2)
        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

        noise_pred = self.forward_unet(
            latent_model_input,
            t.repeat(2 * batch_size),
            torch.cat(prompt_embeddings),  # [positive, negative]
            **kwargs
        )

        # CFG: uncond + scale * (cond - uncond)
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred_text, _ = noise_pred_text.split(3, dim=1)
        noise_pred_uncond, _ = noise_pred_uncond.split(3, dim=1)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        return noise_pred

    def get_denoised_latents(self, latents_noisy, noise_pred, t, clip=True):
        """Estimates the clean, original latent image from the noisy input and predicted noise."""
        alpha_t = self.alphas[t]
        beta_t = 1 - alpha_t
        x0 = (latents_noisy - noise_pred * beta_t.sqrt()) / alpha_t.sqrt()
        return x0.clamp(-1.0, 1.0)

    def forward(
        self,
        images,
        condition_images=None,
        prompt_embeddings=None,
        min_timestamp=200,
        max_timestamp=980,
        guidance_scale=15.0,
        lowres_noise_level=0.5,  # for stage ii
    ):
        """
            Computes the Score Distillation Sampling (SDS) loss by driving the rendered image 
            towards the text prompt using the diffusion model's gradients.
        """
        batch_size = images.shape[0]

        latents = self.prepare_latents(images)
        t = torch.randint(
            min_timestamp,
            max_timestamp,
            [1],
            dtype=torch.long,
            device=self.device
        )

        with torch.no_grad():
            noise = torch.randn_like(latents, device=self.device)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)

            noise_pred = self.predict_noise(
                latents_noisy,
                t,
                prompt_embeddings,
                guidance_scale,
                condition_images,
                lowres_noise_level
            )

            latents_denoised = self.get_denoised_latents(
                latents_noisy,
                noise_pred,
                t
            )


        w = ((1 - self.alphas[t]) * self.alphas[t]).sqrt().view(-1, 1, 1, 1)
        loss_sds = 0.5 * w * F.mse_loss(latents, latents_denoised, reduction="sum") / batch_size

        return loss_sds