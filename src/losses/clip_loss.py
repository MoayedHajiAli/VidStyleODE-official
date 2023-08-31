from ast import List
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import pytorch_lightning as pl

import numpy as np
import clip

class DirectionLoss(torch.nn.Module):
    def __init__(self, loss_type='mse', mse_scale=0.5, beta=None):
        super(DirectionLoss, self).__init__()

        self.loss_type = loss_type
        self.loss_func = {
            'mse':    torch.nn.MSELoss(),
            'cosine': lambda a, b : 1 - F.cosine_similarity(a, b),
            'mae':    torch.nn.L1Loss(),
            'dot': lambda a, b : -1.0 * (a * b).sum(dim=-1),
            'cosine_mse' : lambda a, b : mse_scale * F.mse_loss(a, b) + (1.0 - mse_scale) * (1 - F.cosine_similarity(a, b))
        }[loss_type]

    def forward(self, x, y):        
        return self.loss_func(x, y)

class DirectionalCLIPLoss(pl.LightningModule):
    def __init__(self, embds_mean, lambda_direction=1.,
                     lambda_global=0., clip_model_name='ViT-B/32',
                     loss_type='cosine', norm=True, **kwargs):
        super().__init__()
        self.embed_mean = None
        if embds_mean is not None:
            self.embds_mean = torch.Tensor(np.load(embds_mean))
        
        self.loss_fn = CLIPLoss(lambda_direction, lambda_global,
                                direction_loss_type=loss_type,
                                clip_model=clip_model_name,
                                norm=norm,
                                 **kwargs)

    def forward(self, pos_xgen, neg_xgen, pos_styles, neg_styles, optimizer_idx,
                global_step, split="train", **kwargs):  
        log = {}
        if optimizer_idx == 0 and 'pos' in split:
            if self.embds_mean is not None:
                pos_styles += self.embds_mean.to(pos_styles.device)
                neg_styles += self.embds_mean.to(neg_styles.device)

            loss = self.loss_fn(pos_xgen, pos_styles, neg_xgen, neg_styles,
                                 global_step=global_step)
            log[f'{split}/clip_loss'] = loss.clone().detach().mean()
        
        else:
            loss = torch.zeros(1)[0]

        return loss, log
            
    def configure_optimizers(self, optimizer_idx):
        params = []
        
        return params


class CLIPLoss(pl.LightningModule):
    def __init__(self, lambda_direction=1., lambda_global=0.,  
                    direction_loss_type='cosine', clip_model='ViT-B/32', 
                    projection_direction=None, norm=True,
                    **kwargs):
        super(CLIPLoss, self).__init__()
        self.model, clip_preprocess = clip.load(clip_model, device=self.device)
        self.norm = norm
        self.eps = 1e-9
        # freeze clip model
        for param in self.model.parameters():
            param.requires_grad = False

        self.clip_preprocess = transforms.Compose([transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])] + # Un-normalize from [-1.0, 1.0] (GAN output) to [0, 1].
                                              clip_preprocess.transforms[:2] +                                      # to match CLIP input scale assumptions
                                              clip_preprocess.transforms[4:])                                       # + skip convert PIL to tensor 


        self.direction_loss = DirectionLoss(direction_loss_type, **kwargs)
        self.lambda_global    = lambda_global
        self.lambda_direction = lambda_direction

        self.proj_directions = None
        if projection_direction is not None:
            self.proj_directions = self.compute_text_directions(projection_direction['tgt'], projection_direction['src'], norm=self.norm) # N x D
            self.proj_norm = self.proj_directions.norm(2) ** 2 # N 
            self.betas = np.linspace(projection_direction['beta_st'], projection_direction['beta_en'], projection_direction['steps'])
 

    # def get_perceptual_loss(self, imgs1, imgs2):


    def get_text_features(self, texts:List, norm: bool = None) -> torch.Tensor:
        tokens = self.tokenize(texts)
        text_features = self.encode_text(tokens).detach()
        if norm is None:
            norm = self.norm
        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features
    
    def compute_text_directions(self, src_texts: str, tgt_texts: str, norm) -> torch.Tensor:
        source_features = [self.get_text_features(lst, norm=norm) for lst in src_texts]
        target_features = [self.get_text_features(lst, norm=norm) for lst in tgt_texts]

        text_direction = [torch.mean((tgt - src), dim=0) for tgt, src in zip(target_features, source_features)] 
        text_direction = torch.stack(text_direction)

        if norm:
            text_direction /= text_direction.norm(dim=-1, keepdim=True)

        return text_direction

    def tokenize(self, strings: list):
        return clip.tokenize(strings).to(self.device)

    def encode_text(self, tokens: list) -> torch.Tensor:
        return self.model.encode_text(tokens)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        images = self.clip_preprocess(images).to(images.device)
        return self.model.encode_image(images)


    def get_image_features(self, img: torch.Tensor, norm: bool = None) -> torch.Tensor:
        image_features = self.encode_images(img)
        
        if norm is None:
            norm = self.norm
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

            
    def clip_directional_loss(self, pos_img: torch.Tensor, pos_style_embed: torch.Tensor,
                             neg_img: torch.Tensor, ref_style_embed: torch.Tensor, 
                             global_step: int, norm=True, video=False, num_samples=3) -> torch.Tensor:
        if video:
            # sample frames
            num_samples = min(num_samples, pos_img.shape[1])
            indices = np.random.choice(pos_img.shape[1], size=num_samples, replace=False)
            pos_img = pos_img[:, indices]
            neg_img = neg_img[:, indices]
            
            # pos_img: B x T x C x H x W
            # pos_style_embed: B x T x D
            B,  T,  C,  H, W = pos_img.shape

            pos_img = pos_img.contiguous().reshape(B*T, C, H, W)
            neg_img = neg_img.contiguous().reshape(B*T, C, H, W)
            pos_encoding = self.get_image_features(pos_img, norm=norm)
            neg_encoding = self.get_image_features(neg_img, norm=norm)
            
            pos_encoding = pos_encoding.view(B, T, -1) # B x T x D
            neg_encoding = neg_encoding.view(B, T, -1)
            
            # # reduce
            pos_encoding = pos_encoding.mean(1)
            neg_encoding = neg_encoding.mean(1)
            pos_style_embed = pos_style_embed.mean(1) # B x D
            ref_style_embed = ref_style_embed.mean(1)

        else:
            pos_encoding = self.get_image_features(pos_img, norm=norm)
            neg_encoding = self.get_image_features(neg_img, norm=norm)

        # normalize
        if norm:
            pos_style_embed = pos_style_embed / pos_style_embed.norm(dim=-1, keepdim=True)
            ref_style_embed = ref_style_embed / ref_style_embed.norm(dim=-1, keepdim=True)

        pos_direction = (pos_style_embed.float().contiguous()  - ref_style_embed.float().contiguous())
        neg_direction = (pos_encoding.float().contiguous()  - neg_encoding.float().contiguous())

        # TODO: fix
        if torch.all(pos_direction == 0):
            return neg_direction.norm()

        # pos_direction += self.eps
        # neg_direction += self.eps

        if norm:
            pos_direction = pos_direction / ((pos_direction.norm(dim=-1, keepdim=True)) + self.eps)
            neg_direction = neg_direction / ((neg_direction.norm(dim=-1, keepdim=True)) + self.eps)

        # project to given directions
        if self.proj_directions is not None:
            # move to correct device 
            self.proj_directions = self.proj_directions.to(pos_direction.device)
            self.proj_norm = self.proj_norm.to(pos_direction.device)

            global_step = min(global_step, len(self.betas)-1)
            pos_betas = (self.proj_directions.to(pos_direction.device).unsqueeze(0) * pos_direction.unsqueeze(1)).sum(-1) # B x N
            pos_betas =  pos_betas / self.proj_norm.to(pos_direction.device)
            # print("pos_beta", pos_betas)

            neg_betas = (self.proj_directions.to(neg_direction.device).unsqueeze(0) * neg_direction.unsqueeze(1)).sum(-1) # B x N
            # print("neg_beta", neg_betas)
            neg_betas =  neg_betas / self.proj_norm.to(neg_direction.device)

            mask = (torch.abs(pos_betas) <= self.betas[global_step]) & (torch.abs(neg_betas) <= self.betas[global_step])

            pos_betas[mask] = 0
            neg_betas[mask] = 0
            pos_direction = pos_betas.unsqueeze(-1) * self.proj_directions            
            neg_direction = neg_betas.unsqueeze(-1) * self.proj_directions


        return self.direction_loss(pos_direction, neg_direction).mean()

    
    def consistency_loss(self, generated_images, num_samples=3):
        # generated_images B x T x C x H x W
        num_samples = min(num_samples, generated_images.shape[1])
        indices = np.random.choice(generated_images.shape[1], size=num_samples, replace=False)
        generated_images = generated_images[:, indices]

        bs, T, C, H, W = generated_images.shape
        generated_images = generated_images.reshape(bs * T, C, H, W) 
        embeds = self.encode_images(generated_images) # B  * num_samples x D
        embeds = embeds.view(bs, T, -1)

        clip_loss = 0
        for i in range(1, num_samples):
            clip_loss += (1 - F.cosine_similarity(embeds, torch.roll(embeds, i, dims=1), dim=-1))
        return clip_loss.mean().mean()
    
    
    def global_clip_loss(self, img: torch.Tensor, embed) -> torch.Tensor:
        # TODO: fix         
        raise "not implemented"



    def directional_loss(self, pos_img: torch.Tensor, pos_embed: torch.Tensor, 
                                neg_img: torch.Tensor, neg_embed: torch.Tensor,
                                global_step, video=False):
        clip_loss = 0.0
        if self.lambda_direction:
            clip_loss = self.lambda_direction * self.clip_directional_loss(pos_img, pos_embed, neg_img, neg_embed, global_step=global_step, norm=self.norm, video=video)

        return clip_loss
    
    def forward(self, *args, **kwargs):
        return self.directional_loss(*args, **kwargs)