import pytorch_lightning as pl
import numpy as np
import torch
import torch.nn as nn
import os
import wandb
import imageio
import torchvision
import uuid

from src.utils import *
from src.losses.clip_loss import CLIPLoss
from src.losses.loss_lib import HybridOptim
from src.modules.schedulers import ConstScheduler


class VidStyleODE(pl.LightningModule):
    def __init__(self,
                    video_ecnoder_config,
                    style_mapper_config,
                    stylegan_gen_config,
                    modulation_network_config,
                    lambda_vgg,
                    rec_loss_lambda,
                    l2_latent_lambda,
                    clip_loss_lambda,
                    consistency_lambda,
                    delta_inversion_weight,
                    l2_latent_eps,
                    n_sampled_frames = 3,
                    perceptual_loss_config = None,
                    discriminator_config = None,
                    scheduler_config = None,
                    tgt_text = None,
                    n_critic = 1,
                    ckpt_path=None,
                    content_mode = 'mean_inversion',
                    ignore_keys=[],
                    video_length = 100,
                    frame_log_size=(1024, 512),
                    sampling_type="static",
                    manipulation_strength=2,
                    ):
        super().__init__()

        self.n_sampled_frames = n_sampled_frames
        self.sampling_type = sampling_type
        self.content_mode = content_mode
        self.scheduler_config = scheduler_config
        self.n_critic = n_critic
        self.frame_log_size = tuple(frame_log_size)
        self.video_length = video_length
        self.manipulation_strength = manipulation_strength
        self.downsample_video_size = tuple(video_ecnoder_config.params.img_size)

        self.delta_inversion_weight = delta_inversion_weight
        self.l2_latent_eps = l2_latent_eps
        self.lambda_vgg = self.prepare_scheduler(lambda_vgg)
        self.l2_latent_lambda = self.prepare_scheduler(l2_latent_lambda)
        self.clip_loss_lambda = self.prepare_scheduler(clip_loss_lambda)
        self.rec_loss_lambda = self.prepare_scheduler(rec_loss_lambda)
        self.consistency_lambda = self.prepare_scheduler(consistency_lambda)

        # initialize model
        self.bVAE_enc = instantiate_from_config(video_ecnoder_config)
        self.style_mapper = instantiate_from_config(style_mapper_config)
        self.stylegan_G = instantiate_from_config(stylegan_gen_config)
        self.modulation_network = instantiate_from_config(modulation_network_config)
        self.requires_grad(self.stylegan_G, False)

        # temporal discriminator
        self.temporal_discriminator = None
        if discriminator_config is not None:
            self.temporal_discriminator = instantiate_from_config(discriminator_config)

        # loss
        if perceptual_loss_config is not None:
            self.criterionVGG = instantiate_from_config(perceptual_loss_config).eval() #  LPIPS().eval()
        self.requires_grad(self.criterionVGG, False)

        self.rec_loss = nn.MSELoss()
        self.l2_latent_loss = nn.MSELoss()
        self.clip_loss = CLIPLoss()

        self.tgt_text_embed = None
        if tgt_text is not None:
            self.tgt_text_embed = self.clip_encode_text([tgt_text]) # 1 x 512

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def prepare_scheduler(self, val):
        if isinstance(val, float):
            return ConstScheduler(val)
        else:
            return instantiate_from_config(val)

    def requires_grad(self, model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def preprocess_text_feat(self, latent_feat, mx_roll=2):
        bs = int(latent_feat.size(0)/2)
        if self.tgt_text_embed is not None:
            self.tgt_text_embed = self.tgt_text_embed.to(latent_feat.device)
            latent_feat_mismatch = self.tgt_text_embed.repeat(latent_feat.size(0), 1)

            latent_splits = torch.split(latent_feat, bs, 0)
            latent_feat_relevant = torch.cat((self.tgt_text_embed.repeat(bs, 1), latent_splits[1]), 0)
        else:
            roll_seed = np.random.randint(1, mx_roll)
            latent_feat_mismatch = torch.roll(latent_feat, roll_seed, dims=0)

            latent_splits = torch.split(latent_feat, bs, 0)
            roll_seed = np.random.randint(1, min(bs, mx_roll))
            latent_feat_relevant = torch.cat((torch.roll(latent_splits[0], roll_seed, dims=0), latent_splits[1]), 0)
        return latent_feat_mismatch, latent_feat_relevant


    def clip_encode_text(self, texts):
        texts = self.clip_loss.tokenize(texts).to(self.device)
        return self.clip_loss.encode_text(texts).float()

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def on_train_epoch_start(self,):
        # shuffle training data
        self.trainer.train_dataloader.dataset.reset()

    def video_dynamic_rep(self, vid_bf, ts, mask):
        """
        ts: B (processed ts)
        vid_bf : B x T x C x H x W
        mask: T x 1
        """
        # videos reshape
        bs, T, ch, height, width = vid_bf.size()
        vid_tf = vid_bf.permute(1,0,2,3,4) # T x B x C x H x W
        video_sample = vid_tf.contiguous().view(T * bs, ch, height, width) # T*B x C x H x W // range [0,1]

         # downsample res for vae TODO: experiment with downsampling the resolution much more/ No downsampling
        vid_rs = nn.functional.interpolate(video_sample, size=self.downsample_video_size, mode="bicubic", align_corners=False) # T*B x C x H//2 x W//2
        vid_rs_tf = vid_rs.view(T, bs, ch, vid_rs.shape[2], vid_rs.shape[3])
        vid_rs_bf = vid_rs_tf.permute(1,0,2,3,4).contiguous() # B x T x C x H//2 x W//2

        # vae encode frames
        video_dynamics = self.bVAE_enc.video_dynamics(vid_rs_bf, ts, mask) # B x C x H' x D'
        return video_dynamics

    def sample_frames_dynamics(self, video_dynamics, ts):
        return self.bVAE_enc.solve_ode(video_dynamics, ts) # T * B x D

    def sample_frames(self, mean_inversion, condition_vector):
        # frame rep (video_style, video_content, dynamics)
        # video_dynamics # T * B x D
        # mean_inv: # 1 x B x 18 x 512
        # condition_vector: [3 x (T * B x D)]

        bs = mean_inversion.size(1)
        T = condition_vector[0].size(0) // bs
        src_inversion_tf = mean_inversion.repeat(T, 1, 1, 1)
        src_inversion = src_inversion_tf.reshape(T*bs, src_inversion_tf.shape[-2], src_inversion_tf.shape[-1])
        w_latents = src_inversion + self.delta_inversion_weight * self.style_mapper(src_inversion, condition_vector)
        ret = self.stylegan_G(w_latents)
        return ret


    def forward(self, vid_bf, sampleT, mean_inversion, txt_dir, frame_feat, mask=None, rep_video_dynamics=None):
        bs = frame_feat.shape[0]

        # repeat features
        tar_T = sampleT.shape[0]
        txt_feat_tf = txt_dir.unsqueeze(0).repeat(tar_T, 1, 1)
        txt_feat_tb = txt_feat_tf.contiguous().view(tar_T*bs, -1)
        txt_dir = txt_feat_tb

        # extract frame features
        zF_tf = frame_feat.unsqueeze(0).repeat(tar_T, 1, 1)
        zF_tb = zF_tf.contiguous().view(tar_T*bs, -1)
        frame_feat = zF_tb

        # videos reshape
        ts = (sampleT) / self.video_length
        ts = ts - ts[0]


        # vae encode frames
        if rep_video_dynamics is None:
            bs, T, ch, height, width = vid_bf.size()
            if mask is None:
                mask = torch.ones(T, 1)
            rep_video_dynamics = self.video_dynamic_rep(vid_bf, ts, mask=mask)

        frame_dynamics = self.sample_frames_dynamics(rep_video_dynamics, ts) # T * B x D x H' x W'
        frame_dynamics = frame_dynamics.permute(0, 2, 3, 1).contiguous().view(tar_T*bs, -1, frame_dynamics.shape[1]) # T * B x H' * W' x D

        frame_rep = (txt_dir.unsqueeze(1) + frame_feat.unsqueeze(1), frame_dynamics) # T*B x D1+D2
        conditional_vector = self.modulation_network(*frame_rep)

        ret = self.sample_frames(mean_inversion.unsqueeze(0), conditional_vector)
        ret = ret.reshape(tar_T, bs, ret.shape[1], ret.shape[2], ret.shape[3]).permute(1, 0, 2, 3, 4)
        return ret

    def prepare_mask(self, T):
        mask = torch.zeros(T, 1)
        if self.sampling_type == "interpolate":
            inds = np.random.choice(np.arange(1, T-1), min(self.n_samples, T) - 2 - self.n_frames_interp, replace=False)
            mask[inds, :] = 1
            mask[0, :] = 1
            mask[T-1, :] = 1
        elif self.sampling_type == "extrapolate":
            inds = np.arange(0, T-self.n_frames_ext)
            mask[inds, :] = 1
        elif self.sampling_type == "static":
            mask = torch.ones(T, 1)

        return mask

    def training_step(self, batch, batch_idx):
        if 'attribute' in batch:
            input_desc = [f"a photo of a person with {att}" for att in batch['attribute']]
        else:
            input_desc = batch['raw_desc'] # B

        # videos reshape
        vid_bf = batch['real_img'] # B x T x C x H x W
        bs, T, ch, height, width = vid_bf.size()

        # sample frames
        n_frames = min(T, self.n_sampled_frames)
        frame_inds = [0] + sorted(list(np.random.choice(T-1, n_frames-1, replace=False) + 1))

        vid_tf = vid_bf.permute(1,0,2,3,4) # T x B x C x H x W
        video_sample = vid_tf[frame_inds].contiguous().view(n_frames * bs, ch, height, width) # T*B x C x H x W // range [0,1]
        video_sample_norm = video_sample * 2 - 1 # range [-1, 1] to pass to the generator and disc

        sampleT = batch['sampleT']  # B x T
        assert torch.all(sampleT[0] == sampleT[np.random.randint(sampleT.size(0)-1)+1]), f"inconsistnet frame index: {batch['index']}"
        sampleT = sampleT[0] # B  --assumption: all batch['sampleT'] are the same
        ts = (sampleT) / self.video_length
        ts = ts - ts[0]

        # inversions reshape
        inversions_bf = batch['inversion'] # B, T x n_layers x D
        bs, T, n_channels, dim = inversions_bf.shape
        inversions_tf = inversions_bf.permute(1, 0, 2, 3)

        # predict latents delta
        if self.content_mode == 'mean_inversion':
            mean_inversion = inversions_tf.mean(0, keepdims=True) # 1 x B x 18 x 512
            mismatch_inversion = mean_inversion

        else:
            ind, ind_mismatch = np.random.choice(T, 2)
            mean_inversion = inversions_tf[ind:ind+1]
            mismatch_inversion = inversions_tf[ind_mismatch:ind_mismatch+1]
            
        with torch.no_grad():
            ref_frame = self.stylegan_G(mean_inversion[0]) / 2 + 0.5 # B x C x H x W

        # encode text
        txt_feat = self.clip_encode_text(input_desc)  # B x D
        txt_feat_tf = txt_feat.unsqueeze(0).repeat(n_frames, 1, 1)
        txt_feat_tb = txt_feat_tf.contiguous().view(n_frames*bs, -1)
        txt_feat = txt_feat_tb

        # extract frame features
        frame_feat = self.clip_loss.encode_images(ref_frame * 2 - 1) # B x D
        zF_tf = frame_feat.unsqueeze(0).repeat(n_frames, 1, 1)
        zF_tb = zF_tf.contiguous().view(n_frames*bs, -1)
        frame_video_style = zF_tb

        # vae encode frames
        rep_video_dynamics = self.video_dynamic_rep(vid_bf, ts, mask=self.prepare_mask(T))

        # sample n_frames
        frame_dynamics = self.sample_frames_dynamics(rep_video_dynamics, ts[frame_inds]) # n_frames * B x D x H' x W'
        frame_dynamics = frame_dynamics.permute(0, 2, 3, 1).contiguous().view(n_frames*bs, -1, frame_dynamics.shape[1]) # n_frames * B x H' * W' x D

        total_loss = 0
        vgg_loss = 0

        # roll batch-wise
        txt_feat_mismatch, _ = self.preprocess_text_feat(txt_feat, mx_roll=2) # T*B x D2
        txt_dir = self.manipulation_strength * (txt_feat_mismatch - txt_feat)

        # frame rep (video_style, video_content, dynamics)
        frame_rep = (frame_video_style.unsqueeze(1), frame_dynamics)# T*B x D1+D2
        frame_rep_txt_mismatched = (txt_dir.unsqueeze(1) + frame_video_style.unsqueeze(1), frame_dynamics) # T*B x D1+D2

        src_inversion_tf = mean_inversion.repeat(n_frames, 1, 1, 1)
        src_inversion = src_inversion_tf.reshape(n_frames*bs, n_channels, dim)

        src_inversion_4_mismatch_tf = mismatch_inversion.repeat(n_frames, 1, 1, 1)
        src_inversion_4_mismatch = src_inversion_4_mismatch_tf.reshape(n_frames*bs, n_channels, dim)

        conditional_vector = self.modulation_network(*frame_rep)
        conditional_vector_mismatched = self.modulation_network(*frame_rep_txt_mismatched)

        w_latents = src_inversion + self.delta_inversion_weight * self.style_mapper(src_inversion, conditional_vector)
        w_latents_txt_mismatched = src_inversion_4_mismatch + self.delta_inversion_weight * self.style_mapper(src_inversion_4_mismatch, conditional_vector_mismatched)

        reconstruction = self.stylegan_G(w_latents) # T*B x 3 x H x W
        imgs_txt_mismatched = self.stylegan_G(w_latents_txt_mismatched) # T*B x 3 x H x W

        reconstruction_inp_res = nn.functional.interpolate(reconstruction, size=(height, width), mode="bicubic", align_corners=False)
        imgs_txt_mismatched_inp_res = nn.functional.interpolate(imgs_txt_mismatched, size=(height, width), mode="bicubic", align_corners=False)


        # calculate losses
        reconstruction_loss = 0
        if self.global_step == 0 or self.rec_loss_lambda(self.global_step) > 0:
            reconstruction_loss = self.rec_loss(reconstruction_inp_res, video_sample_norm)


        latent_loss = torch.maximum(self.l2_latent_loss(src_inversion, w_latents) - self.l2_latent_eps / 2, torch.zeros(1).to(src_inversion.device)[0])
        latent_loss += torch.maximum(self.l2_latent_loss(src_inversion, w_latents_txt_mismatched) - self.l2_latent_eps, torch.zeros(1).to(src_inversion.device)[0])


        # structure/style loss
        style_loss, structure_loss = 0, 0
        structure_loss += self.criterionVGG.structure_loss(reconstruction_inp_res.contiguous() / 2 + 0.5, video_sample.contiguous().detach()).mean()
        structure_loss += self.criterionVGG.structure_loss(imgs_txt_mismatched_inp_res.contiguous() / 2 + 0.5, video_sample.contiguous().detach()).mean()

        style_loss += self.criterionVGG.style_loss(reconstruction_inp_res.contiguous() / 2 + 0.5, ref_frame.repeat(bs*T, 1, 1, 1).contiguous().detach()).mean()
        vgg_loss = structure_loss + style_loss

        ### video based losses ###
        txt_feat_bf = txt_feat_tf.permute(1, 0, 2).contiguous() # B x T x D


        txt_feat_mismatch_tf = txt_feat_mismatch.contiguous().reshape(n_frames, bs, txt_feat_mismatch.shape[1])
        txt_feat_mismatch_bf = txt_feat_mismatch_tf.permute(1, 0, 2).contiguous()  # B x T x D

        vid_norm_bf = vid_bf[:, frame_inds] * 2 - 1

        reconstruction_tf = reconstruction.contiguous().reshape(n_frames, bs, reconstruction.shape[1], reconstruction.shape[2], reconstruction.shape[3])
        reconstruction_bf = reconstruction_tf.permute(1, 0, 2, 3, 4).contiguous()  # B x T x C x H x W

        imgs_txt_mismatched_tf = imgs_txt_mismatched.contiguous().reshape(n_frames, bs, imgs_txt_mismatched.shape[1], imgs_txt_mismatched.shape[2], imgs_txt_mismatched.shape[3])
        imgs_txt_mismatched_bf = imgs_txt_mismatched_tf.permute(1, 0, 2, 3, 4).contiguous()   # B x T x C x H W

        imgs_txt_mismatched_inp_res_tf = imgs_txt_mismatched_inp_res.contiguous().reshape(n_frames, bs, imgs_txt_mismatched_inp_res.shape[1], imgs_txt_mismatched_inp_res.shape[2], imgs_txt_mismatched_inp_res.shape[3])
        imgs_txt_mismatched_inp_res_bf = imgs_txt_mismatched_inp_res_tf.permute(1, 0, 2, 3, 4).contiguous()   # B x T x C x H W

        # directional loss
        directional_clip_loss = 0
        if self.global_step == 0 or self.clip_loss_lambda(self.global_step) > 0:
            # TODO: should we use vid_norm_bf or the inversion?
            directional_clip_loss = self.clip_loss.directional_loss(vid_norm_bf, txt_feat_bf, imgs_txt_mismatched_inp_res_bf, txt_feat_mismatch_bf, self.global_step, video=True)

        # consistency loss
        consistency_loss = 0
        if self.global_step == 0 or self.consistency_lambda(self.global_step) > 0:
            consistency_loss = self.clip_loss.consistency_loss(reconstruction_bf)
            consistency_loss += self.clip_loss.consistency_loss(imgs_txt_mismatched_bf)

        # lambdas
        self.log("lambda/consistency_loss", self.consistency_lambda(self.global_step), prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("lambda/vgg_loss", self.lambda_vgg(self.global_step), prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("lambda/rl_loss", self.rec_loss_lambda(self.global_step), prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("lambda/clip_loss", self.clip_loss_lambda(self.global_step), prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log("lambda/l2_latent_loss", self.l2_latent_lambda(self.global_step) , prog_bar=False, logger=True, on_step=False, on_epoch=True)

        # losses
        self.log("train/consistency_loss", consistency_loss, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("train/structure_loss", structure_loss, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("train/style_loss", style_loss, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("train/vgg_loss", vgg_loss, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log("train/rl_loss", reconstruction_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/clip_loss", directional_clip_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/l2_latent_loss", latent_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        total_loss = self.rec_loss_lambda(self.global_step) * reconstruction_loss \
                    + self.clip_loss_lambda(self.global_step) * directional_clip_loss \
                    + self.l2_latent_lambda(self.global_step) * latent_loss \
                    + self.lambda_vgg(self.global_step) * vgg_loss \
                    + self.consistency_lambda(self.global_step) * consistency_loss 

        self.log("train/total_loss", total_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return total_loss


    def validation_step(self, batch, batch_idx):
        pass


    def log_videos_as_imgs(self, vid_bf):
        # videos reshape
        bs, T, ch, height, width = vid_bf.size()
        vid_tf = vid_bf.permute(1,0,2,3,4) # T x B x C x H x W
        video_sample = vid_tf.contiguous().view(T * bs, ch, height, width) # T*B x C x H x W // range [0,1]
        video_sample_norm = video_sample * 2 - 1 # range [-1, 1] to pass to the generator and disc
        return video_sample_norm

    def log_images(self, batch, split):
        """
        return a dictionary of tensors in the range [-1, 1]
        """
        ret = dict()
        if self.global_step == 0 and split == 'train':
            return ret

        if 'attribute' in batch:
            input_desc = [f"a photo of a woman wearing {att}" for att in batch['attribute']]
        else:
            input_desc = batch['raw_desc'] # B
        vid_bf = batch['real_img'] # B x T x C x H x W

        sampleT = batch['sampleT']  # B x T
        assert torch.all(sampleT[0] == sampleT[np.random.randint(sampleT.size(0)-1)+1]), f"index: {batch['index']}"
        sampleT = sampleT[0] # B  --assumption: all batch['sampleT'] are the same
        ts = (sampleT) / self.video_length
        ts = ts - ts[0]


        bs, T, ch, height, width = vid_bf.size()
        vid_tf = vid_bf.permute(1,0,2,3,4) # T x B x C x H x W

        # inversions reshape
        inversions_bf = batch['inversion'] # B, T x n_layers x D
        bs, T, n_channels, dim = inversions_bf.shape
        inversions_tf = inversions_bf.permute(1, 0, 2, 3)
        mean_inversion = inversions_tf.mean(0, keepdims=True) # 1 x B x 18 x 512
        ref_frame = self.stylegan_G(mean_inversion[0]) # B x C x H x W

        # encode text
        txt_feat = self.clip_encode_text(input_desc)  # B x D
        txt_feat_tf = txt_feat.unsqueeze(0).repeat(T, 1, 1)
        txt_feat_tb = txt_feat_tf.contiguous().view(T*bs, -1)
        txt_feat = txt_feat_tb

        # extract frame features
        frame_feat = self.clip_loss.encode_images(ref_frame) # B x D
        zF_tf = frame_feat.unsqueeze(0).repeat(T, 1, 1)
        zF_tb = zF_tf.contiguous().view(T*bs, -1)
        frame_video_style = zF_tb

        ###### static ########
        # vae encode frames
        rep_video_dynamics = self.video_dynamic_rep(vid_bf, ts, mask=self.prepare_mask(T))
        frame_dynamics = self.sample_frames_dynamics(rep_video_dynamics, ts) # T * B x D x H' x W'
        frame_dynamics = frame_dynamics.permute(0, 2, 3, 1).contiguous().view(T*bs, -1, frame_dynamics.shape[1]) # T * B x H' * W' x D


        # roll batch-wise
        txt_feat_mismatch, _ = self.preprocess_text_feat(txt_feat, mx_roll=2) # T*B x D2
        txt_dir = self.manipulation_strength * (txt_feat_mismatch - txt_feat)

        frame_dynamics_swapped = torch.roll(frame_dynamics.view(T, bs, -1, frame_dynamics.shape[-1]).contiguous(), 1, dims=1).view(T*bs, -1, frame_dynamics.shape[-1])

        # frame rep (video_style, video_content, dynamics)

        frame_rep = (frame_video_style.unsqueeze(1), frame_dynamics) # T*B x D1+D2
        frame_rep_txt_mismatched = (txt_dir.unsqueeze(1) + frame_video_style.unsqueeze(1), frame_dynamics) # T*B x D1+D2
        frame_rep_dynamics_swapped= (frame_video_style.unsqueeze(1), frame_dynamics_swapped) # T*B x D1+D2

        conditional_vector = self.modulation_network(*frame_rep)
        conditional_vector_mismatched = self.modulation_network(*frame_rep_txt_mismatched)
        conditional_vector_dynamic_swapper= self.modulation_network(*frame_rep_dynamics_swapped)
        # mean frame
        ret['x_recon_mean']  = self.sample_frames(mean_inversion, conditional_vector)
        ret['x_mismatch_mean']  = self.sample_frames(mean_inversion, conditional_vector_mismatched)
        ret['x_recon_mean_image']  = self.sample_frames(mean_inversion, conditional_vector)
        ret['x_mismatch_mean_image']  = self.sample_frames(mean_inversion, conditional_vector_mismatched)
        ret['x_swapped_dynamics']  = self.sample_frames(mean_inversion, conditional_vector_dynamic_swapper)
        ret['img_mean_inv'] = self.stylegan_G(mean_inversion[0])

        self.log("val/video_dynamics_effectivness", 1 - torch.nn.functional.mse_loss(ret['x_swapped_dynamics'], ret['x_recon_mean']), prog_bar=False, logger=True, on_step=True, on_epoch=True)

        # first frame
        first_inversion = inversions_tf[0:1] # 1 x B x 18 x 512
        ref_frame = self.stylegan_G(first_inversion[0])
        frame_feat = self.clip_loss.encode_images(ref_frame) # B x D
        zF_tf = frame_feat.unsqueeze(0).repeat(T, 1, 1)
        zF_tb = zF_tf.contiguous().view(T*bs, -1)
        frame_video_style = zF_tb
        frame_rep = (frame_video_style.unsqueeze(1), frame_dynamics) # T*B x D1+D2
        first_frame_conditional_vector = self.modulation_network(*frame_rep)
        ret['x_recon_first']  = self.sample_frames(first_inversion, first_frame_conditional_vector)
        ret['x_recon_mismatch_ref_frame']  = self.sample_frames(first_inversion, conditional_vector)
        ret['img_first_inv'] = ref_frame

        # last frame
        last_inversion = inversions_tf[-1:] # 1 x B x 18 x 512
        ref_frame = self.stylegan_G(last_inversion[0])
        frame_feat = self.clip_loss.encode_images(ref_frame) # B x D
        zF_tf = frame_feat.unsqueeze(0).repeat(T, 1, 1)
        zF_tb = zF_tf.contiguous().view(T*bs, -1)
        frame_video_style = zF_tb
        frame_rep = (frame_video_style.unsqueeze(1), frame_dynamics) # T*B x D1+D2
        last_frame_conditional_vector = self.modulation_network(*frame_rep)
        ret['x_recon_last']  = self.sample_frames(last_inversion, last_frame_conditional_vector)
        ret['x_recon_mismatch_ref_frame']  = self.sample_frames(last_inversion, conditional_vector)
        ret['img_last_inv'] = ref_frame

        ########  interpolation ########
        interp_mask = torch.ones(T, 1)
        interp_mask[[1],:] = 0
        interp_video_dynamics = self.video_dynamic_rep(vid_bf, ts, mask=interp_mask)
        interp_frames_dynamics = self.sample_frames_dynamics(interp_video_dynamics, ts) # T * B x D x H' x W'
        interp_frames_dynamics = interp_frames_dynamics.permute(0, 2, 3, 1).contiguous().view(T*bs, -1, interp_frames_dynamics.shape[1]) # T * B x H' * W' x D
        # frame_rep = torch.cat((frame_video_style.unsqueeze(1), interp_frames_dynamics), 1) # T*B x D1+D2
        frame_rep = (frame_video_style.unsqueeze(1), interp_frames_dynamics) # T*B x D1+D2
        conditional_vector = self.modulation_network(*frame_rep)
        ret['x_interp_mean']  = self.sample_frames(mean_inversion, conditional_vector)

        # # extrapolation
        # extra_mask = torch.ones(T, 1)
        # extra_mask[-2:,:] = 0
        # extra_video_dynamics = self.video_dynamic_rep(vid_bf, ts, mask=extra_mask)
        # extra_frames_dynamics = self.sample_frames_dynamics(extra_video_dynamics, ts)
        # ret['x_exterp_mean']  = self.sample_frames(extra_frames_dynamics, mean_inversion, txt_feat)


        ret['real_image'] = self.log_videos_as_imgs(vid_bf)
        ret['real_image_image'] = self.log_videos_as_imgs(vid_bf)

        if 'inverted_img' in batch: # log inverted videos for reference
            ret['inverted_image'] = self.log_videos_as_imgs(batch['inverted_img'])
            ret['inverted_image_image'] = self.log_videos_as_imgs(batch['inverted_img'])

        C, H, W = ret['real_image'].shape[1:]

        ret = self.downsample_log(ret)
        # log gifs
        video_lst = ['real_image', 'x_recon_mean', 'x_mismatch_mean', 'x_swapped_dynamics', 'x_recon_first', 'x_recon_last', 'inverted_image']

        ret_imgs = {}
        for k, v in ret.items():
            if k in video_lst:
                self.log_gif(v.reshape(T, bs, v.shape[1], v.shape[2], W).contiguous(), range=(-1, 1), name=f'{split}/{k}_gif')
            else:
                ret_imgs[k] = v

        ret_imgs['real_image_image_caption'] = '\n'.join([f"{batch['video_name'][i]}: {el}" for i, el in enumerate(batch['raw_desc'])])
        return ret_imgs

    def downsample_log(self, ret):
        for k, v in ret.items():
            ret[k] = nn.functional.interpolate(v, size=self.frame_log_size, mode="nearest")
        return ret

    def log_gif(self, video, range, name):
        # Assuming that the current shape is T * B x C x H x W
        os.makedirs('tmp_log_gifs', exist_ok=True)
        filename = f'tmp_log_gifs/{str(uuid.uuid4())}.gif'
        with imageio.get_writer(filename, mode='I') as writer:
           for b_frames in video:
                # b_frames B x C x H x W
                frame = torchvision.utils.make_grid(b_frames,
                                nrow=b_frames.shape[0],
                                normalize=True,
                                value_range=range).detach().cpu().numpy()
                frame = (np.transpose(frame, (1, 2, 0)) * 255).astype(np.uint8)
                writer.append_data(frame)

        self.trainer.logger.experiment.log(
                {name: wandb.Video(filename, fps=2, format="gif"),
                "global_step":self.global_step})
        os.remove(filename)

    def configure_optimizers(self):
        lr = self.learning_rate
        vae_params = list(self.bVAE_enc.parameters())

        style_m_params = list(self.style_mapper.parameters()) + \
                        list(self.modulation_network.parameters())

        opt_vae = torch.optim.Adam(vae_params,
                                  lr=lr * 5,
                                  betas=(0.9, 0.999))

        opt_sm = torch.optim.Adam(style_m_params, lr=lr, betas=(0.9, 0.99))

        opt_ae = HybridOptim([opt_vae, opt_sm])


        ae_ret = {"optimizer": opt_ae, "frequency": 1}
        return ae_ret
