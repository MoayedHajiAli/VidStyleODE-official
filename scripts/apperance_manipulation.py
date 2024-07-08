import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, required=True)
parser.add_argument("--run_id", type=str, default="default")
parser.add_argument("--video_list", type=str, default="data/fashion/supp_exp/style1/list.txt")
parser.add_argument("--src_text_list", type=str, default="data/fashion/supp_exp/apperance")
parser.add_argument("--tgt_text_list", type=str, default="data/fashion/supp_exp/apperance")
parser.add_argument("--output_dir", type=str, default="supp_experiments")
parser.add_argument("--img_root", type=str, default="/scratch/users/abond19/datasets/aligned_fashion_dataset")
parser.add_argument("--inversion_root", type=str, default="/scratch/users/abond19/datasets/w+_fashion_dataset/fashion/PTI")
parser.add_argument("--n_start", type=int, default=5)
parser.add_argument("--n_frames", type=int, default=30)
parser.add_argument("--n_step", type=int, default=1)
parser.add_argument("--device", type=str, default='cuda')
parser.add_argument("--spv", type=int, default=10)
parser.add_argument("--n_samples", type=int, default=2)
parser.add_argument("--alpha", type=int, default=1.3)


# load images
import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from IPython.display import Image as IPImage
from src.utils import *
import imageio
import torchvision
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

torch.set_grad_enabled(False)

IMG_EXTENSIONS = ['.png', '.PNG']
TXT_EXTENSIONS = ['.txt']

crop = None
size = None
trans_list = []
if crop is not None:
    trans_list.append(transforms.CenterCrop(tuple(crop)))
if size is not None:
    trans_list.append(transforms.Resize(tuple(size)))
trans_list.append(transforms.ToTensor())
img_transform=transforms.Compose(trans_list)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def is_text_file(filename):
    return any(filename.endswith(extension) for extension in TXT_EXTENSIONS)

def get_image(img_path):
    img = Image.open(img_path).convert('RGB')
    return img

def get_inversion(inversion_path):
    w_vector = torch.load(inversion_path, map_location='cpu')
    if len(w_vector.shape) == 2:
        w_vector = w_vector.unsqueeze(0)
    assert (w_vector.shape == (1, 18, 512)), "Inverted vector has incorrect shape"
    return w_vector

def load_video(args, vid_path):
    images, inversions, sampleT, inversion_imgs = [], [], [], []
    fname = vid_path

    lst_vid = sorted(os.listdir(os.path.join(args.img_root, fname)))
    lst_vid = sorted([vid for vid in lst_vid if vid.endswith('png')])
    def correct(f):
        if is_image_file(f):
            return int(f.split('.')[0])
        else:
            return f

    lst_vid = list(map(correct, lst_vid))[:-1]
    inds = np.argsort(lst_vid)
    inds = inds[args.n_start:args.n_start+args.n_step*args.n_frames:args.n_step]
    for f in np.array(sorted(os.listdir(os.path.join(args.img_root, fname))))[inds]:
        if is_image_file(f):
            imname = f[:-4]
            images.append(img_transform(get_image(os.path.join(args.img_root, fname, f))))
            inversions.append(get_inversion(os.path.join(os.path.join(args.inversion_root, fname, imname + ".pt"))))
            sampleT.append(int(imname))

    with open(os.path.join(args.img_root, fname, 'updated_descriptions.txt')) as f:
        description = f.readlines()[0]

    return torch.stack(images).to(args.device), torch.cat(inversions, 0).to(args.device), torch.Tensor(sampleT).to(args.device), description


def save_img(x):
    x = x.permute(1,2,0).numpy()
    x = (255*x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x

def save_gif(video, range, save_path):
    # Assuming that the current shape is T * B x C x H x W
    frames_dir = os.path.join(f"{save_path[:-4]}")
    os.makedirs(frames_dir, exist_ok=True)
    with imageio.get_writer(save_path, mode='I') as writer:
       for i, b_frames in enumerate(video):
            # b_frames B x C x H x W
            b_frames = torch.clamp(b_frames, range[0], range[1])
            frame = torchvision.utils.make_grid(b_frames,
                            nrow=b_frames.shape[0],
                            normalize=True,
                            value_range=range).detach().cpu().numpy()
            frame = (np.transpose(frame, (1, 2, 0)) * 255).astype(np.uint8)
            writer.append_data(frame)

            # save frame
            Image.fromarray(frame).save(os.path.join(frames_dir, f"{i:06d}.png"))
#     wandb.log({name: wandb.Video(filename, fps=2, format="gif")})
#     os.remove(filename)

def concat_and_save_gif(videos, ref_frame, titles, val_range, save_path):
    # Assuming that the current shape is T * B x C x H x W
    all_frames = []
    for i in range(videos[0].shape[0]):
        frames = []
        for j, video in enumerate(videos):
            # b_frames B x C x H x W
            b_frame = torch.clamp(video[i], val_range[0], val_range[1])
            frame = torchvision.utils.make_grid(b_frame,
                            nrow=b_frame.shape[0],
                            normalize=True,
                            value_range=val_range).detach().cpu().numpy()
            frame = Image.fromarray((np.transpose(frame, (1, 2, 0)) * 255).astype(np.uint8))
            frames.append(frame)
        all_frames.append(stack_imgs(frames, titles))


    final_output = all_frames
    final_output[0].save(save_path, save_all=True, append_images =final_output[1:], duration=100, loop=0)


def display_gif(path):
    return IPImage(filename=path)

def main(args):
    # save dir
    save_root = os.path.join(args.output_dir, f"{args.model_dir.split('/')[-1]}", "apperance_manipulaiton", str(args.run_id))
    os.makedirs(save_root, exist_ok=True)
    # load model
    model = load_model_from_dir(args.model_dir).to(args.device)

    # read videos
    videos = read_lines(args.video_list)
    content_frames = np.random.choice(videos, min(len(videos), args.n_samples), replace=False)


    d_vids, vids_dyns, vids_ts = [], [], []
    all_videos = []
    # process
    for video in tqdm(content_frames):
        src_lst = np.array(read_lines(os.path.join(args.src_text_list, video, "src_txt.txt")))
        tgt_lst = np.array(read_lines(os.path.join(args.tgt_text_list, video, "tgt_txt.txt")))
        txt_inds = sorted(np.random.choice(len(src_lst), min(args.spv,len(src_lst)), replace=False))
        src_lst = src_lst[txt_inds]
        tgt_lst = tgt_lst[txt_inds]
        save_dir = os.path.join(save_root, video)
        os.makedirs(save_dir, exist_ok=True)
        write_lines(src_lst, f"{save_dir}/src_texts.txt")
        write_lines(tgt_lst, f"{save_dir}/tgt_texts.txt")


        all_videos = []
        invs, vid_imgs = [], []
        # sample driving video
        vid_img, vid_inversions, ts, src_desc = load_video(args, video)
        ts_norm = (ts) / model.video_length
        ts_norm = ts_norm -  ts_norm[0]
        dyn = model.video_dynamic_rep(vid_img.unsqueeze(0), ts_norm, mask=torch.ones(vid_img.shape[0], 1))

        # print("vid img", vid_img.shape)
        ind = np.random.randint(vid_img.shape[0]-5) + 5
        ref_frame_inversion = vid_inversions[ind:ind+1] # 1 x 1 x 18 x 512
        invs.append(ref_frame_inversion)
        vid_imgs.append(vid_img)

        invs = torch.cat(invs, 1)
        vid_imgs = torch.stack(vid_imgs, 0)
        # print("inv", invs.shape)
        ref_frame = model.stylegan_G(invs)
        frame_feat = model.clip_loss.encode_images(ref_frame) # B x D


        # print("frame feat", frame_feat.shape)
        to_PIL(ref_frame[0]).save(f'{save_dir}/ref_{video}.png')

        all_videos.append(vid_imgs[0] * 2 - 1)

        for i, (src_txt, tgt_txt) in enumerate(zip(src_lst, tgt_lst)):
            src_txt_feat = model.clip_encode_text([src_txt])
            tgt_txt_feat = model.clip_encode_text([tgt_txt])  # - model.clip_encode_text([src_desc])
            txt_dir = args.alpha * (tgt_txt_feat - src_txt_feat)
            out = model(None,
                        ts,
                        invs,
                        txt_dir.to(args.device),
                        frame_feat,
                        rep_video_dynamics=dyn)
            save_path = f'{save_dir}/{video}_txt_{i}.gif'
            save_gif(out[0], (-1, 1), save_path)

            all_videos.append(out[0])

        titles = [' '.join(desc.split(' ')[-2:]) for desc in tgt_lst]
        titles = ['Original'] + titles
        concat_and_save_gif(all_videos, ref_frame[0], titles, (-1, 1), f'{save_dir}/demo_{video}.gif')




if __name__ == '__main__':
    args, _ = parser.parse_known_args()
    main(args)
