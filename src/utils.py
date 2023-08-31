import os
import yaml
import torch
from omegaconf import OmegaConf
import PIL
from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
import requests
import importlib
import io


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config, **kwargs):
    """
    target class is expected to be at "target" and it will be initiated with params
    """
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()), **kwargs)

def to_PIL(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.)/2.
    x = x.permute(1,2,0).numpy()
    x = (255*x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def download_image_from_web(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return PIL.Image.open(io.BytesIO(resp.content))

def load_image_from_path(path, size):
    return Image.open(path).resize((size, size))

def preprocess_image(image, device):
    image = np.array(image).astype(np.uint8)
    image = (image/127.5 - 1.0).astype(np.float32)
    image = torch.unsqueeze(torch.tensor(image), 0).to(device)
    return image.permute(0, 3, 1, 2)

def load_model(config, ckpt_path=None):
  model = instantiate_from_config(config, ckpt_path=ckpt_path)
  if ckpt_path is not None:
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
  return model

def load_model_from_dir(model_dir):
    config_path = get_config_path(model_dir)
    ckpt_path = get_ckpt_path(model_dir)

    config = load_config(config_path)
    model = load_model(config.model, ckpt_path=ckpt_path)

    return model

def get_ckpt_path(dic):
    paths = os.listdir(dic)
    # print(paths)
    name = dic.split('/')[-1]
    if name in paths:
        return get_ckpt_path(os.path.join(dic, name))
    elif 'VideoManipulation' in paths:
        return get_ckpt_path(os.path.join(dic, 'VideoManipulation'))
    elif 'checkpoints' in paths:
        return get_ckpt_path(os.path.join(dic, 'checkpoints'))
    elif os.path.isfile(os.path.join(dic, sorted(paths)[-1])):
        tgt_file = sorted(paths)[-1]
        if tgt_file.startswith('last') and len(paths) > 1:
            tgt_file = sorted(paths)[-2]
        return os.path.join(dic, tgt_file)
    else:
        return  get_ckpt_path(os.path.join(dic, paths[0]))

def get_config_path(dic):
    import os
    paths = os.listdir(dic)
    if 'configs' in paths:
        return get_config_path(os.path.join(dic, 'configs'))
    else:
        paths = [path for path in paths if 'project' in path]
        return os.path.join(dic, paths[-1])

def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config

def read_lines(file_path):
    with open(file_path, 'r') as f:
        return f.read().split('\n')

def write_lines(lst, file_path):
    with open(file_path, 'w') as f:
        for el in lst:
            f.write(el)
            f.write('\n')

def stack_imgs(Xs, titles=[], c = 30):
    w, h = Xs[0].size[0], Xs[0].size[1]
    img = Image.new("RGB", (len(Xs)*w, h+c))
    for i, x in enumerate(Xs):
        img.paste(x, (i*w,c))

    for i, title in enumerate(titles):
        ImageDraw.Draw(img).text((i*w, 0), f'{title}', (255, 255, 255)) # coordinates, text, color, font
    return img

def stack_imgs_vertical(Xs, titles=[]):
    c = 25
    w, h = Xs[0].size[0], Xs[0].size[1]
    img = Image.new("RGB", (w + c, len(Xs) * h))
    for i, x in enumerate(Xs):
        img.paste(x, (c, i*h))

    for i, title in enumerate(titles):
        f = ImageFont.load_default()
        txt= Image.new('L', (h,c-5))
        d = ImageDraw.Draw(txt)
        d.text((0, 0), f'{title}',  font=f, fill=255)
        w = txt.rotate(90,  expand=1)
        w = ImageOps.colorize(w, (0,0,0), (255,255,84))
        img.paste(w, (5, i*h - 50))
        # ImageDraw.Draw(img).text((0, i*h), f'{title}', (255, 255, 255)) # coordinates, text, color, font
    return img


def image_to_batch(image, DEVICE):
    if isinstance(image, list):
        image = [np.array(img).astype(np.uint8) for img in image]
        image = [img[:, :, :3] if img.shape[-1] == 4 else img for img in image]
        image = np.stack(image)
        image = (image/127.5 - 1.0)
        image = torch.tensor(image).to(DEVICE)
    else:
        image = np.array(image).astype(np.uint8)
        image = image[:, :, :3] if image.shape[-1] == 4 else image
        image = (image/127.5 - 1.0)
        image = torch.unsqueeze(torch.tensor(image), 0).to(DEVICE)

    return {'image':image}

def reconstruct_from_PIL(model, image):
    batch = image_to_batch(image)
    res = model.log_images(batch)
    return to_PIL(res['reconstructions'][0])


def fig_to_PIL(fig):
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format='png')
    im = Image.open(img_buf)
    return im
