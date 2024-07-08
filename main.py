import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

import os
os.environ['WANDB_START_METHOD'] = 'thread'

import torch
import argparse, os, sys, datetime, glob
from omegaconf import OmegaConf
import numpy as np
from PIL import Image


import torchvision
from torch.utils.data import random_split, DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only# decorator to ensure that only the first processor execute the method

import pandas as pd
from src.utils import *
from torch.utils.data.distributed import DistributedSampler
import wandb

def get_parser(**parser_kwargs):
    """
    Get default parser for the experiment
    """
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=True,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=True,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "--sample-data",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="test code on sample data (for debugging purposes)",
    )
    parser.add_argument("-p", "--project", help="name of new or path to existing project")
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=907,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "--tags",
        '--names-list',
        type=str,
        nargs="*",
        default=[],
        help="tags specific for the experiments",
    )

    parser.add_argument(
        "--strict",
        type=str2bool,
        default=True,
        help="Being strict when loading the model from checkpoint",
    )
    parser.add_argument(
        "--resume_id",
        type=str,
        default="",
        help="Resume on the same wandb run and same directory (if exists)",
    )

    parser.add_argument(
        "--save_every",
        type=int,
        default=None,
        help="Save a ckpt every N epochs",
    )

    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataModuleFromConfig(pl.LightningDataModule):
    """
    wrap training, validation, and test with dataloaders if wrap is True
    """
    def __init__(self, batch_size, train=None, validation=None, test=None,
                 wrap=False, num_workers=None, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size*2
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader
        self.wrap = wrap
        self.dataset_kwargs = kwargs

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            print(data_cfg)
            instantiate_from_config(data_cfg, **self.dataset_kwargs)
            print("finished")

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k], **self.dataset_kwargs))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        # train_sampler = DistributedSampler(self.datasets["train"], shuffle=False, drop_last=True)
        dataloader = DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False, drop_last=True) # ATTENTION: shuffle should stay false to consistent sampling
        return dataloader

    def _val_dataloader(self):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False,  drop_last=True)

    def _test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False,  drop_last=True)


class SetupCallback(Callback):
    """
    create the setup (such as creating directories) for callbacks"
    """
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config,
                name, id, model=None, tags=[]):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config
        self.name = name
        self.tags = tags
        self.id = id
        self.model = model

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            wandb.init(project=self.config.project_title,
                    config=OmegaConf.to_container(self.config, resolve=True),
                    name=self.name,
                    id=self.id,
                    dir=self.logdir,
                    tags=self.tags,
                    resume='allow')

            # if self.model is not None:
            #     wandb.watch(self.model, log_freq=5000)
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs_2", name)
                print("creating", os.path.split(dst)[0])
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass

class LossLogger(Callback):
    """
    Generate loss figures at the end of every epoch
    """
    def __init__(self, log_dir):
        super().__init__()
        print(log_dir)
        self.loss_loggers = {
            pl.loggers.WandbLogger: self._wandb,
            # pl.loggers.TestTubeLogger: self._testtube, # Disable testtube for now
        }
        self.log_dir = log_dir
        os.makedirs(os.path.join(log_dir, 'figures') , exist_ok=True)

    @rank_zero_only
    def _wandb(self):
        pass

    @rank_zero_only
    def _testtube(self):
        metrics_path = os.path.join(self.log_dir, 'testtube', 'version_0', 'metrics.csv')
        if not os.path.exists(metrics_path):
            print('[LossLogger] metrics do not exists at path', metrics_path)
            return

        metrics = pd.read_csv(metrics_path)
        interstring_keys = [key for key in metrics.keys() if 'epoch' in key and '/' in key and 'step' not in key] # only take keys that logged over all epochs from the loss function
        interestring_keys = sorted(interstring_keys)

        def plot_all_figures_for_suffex(last_suff, cur_fig_names):
            tmp_lst = last_suff.split('/')
            if len(tmp_lst) >= 2:
                path = '/'.join(tmp_lst[:-1])
                os.makedirs(os.path.join(self.log_dir, 'figures', path), exist_ok=True)

            if not cur_fig_names:
                return

            # plot all all metrics
            fig, axs = plt.subplots(len(cur_fig_names), 1, figsize=(18, len(cur_fig_names) * 3))
            if len(cur_fig_names) == 1:
                axs = [axs] # make axs iteratible

            fig.tight_layout(pad=3.0)
            for name, ax in zip(cur_fig_names, axs):
                ax.set_title(name)
                ax.set_xlabel('epochs')
                series = metrics[f'{last_suff}/{name}'].dropna()
                ax.plot(np.arange(len(series) - 2), series.values[2:])

            # save the figures
            fig.savefig(os.path.join(self.log_dir, 'figures', f'{last_suff}.png'))

        # make all metrics that have the same suffix in a single figure
        last_suff = None
        cur_fig_names = []
        for key in interestring_keys:
            tmp_lst = key.split('/')
            suff, name = '/'.join(tmp_lst[:-1]), tmp_lst[-1]
            if last_suff and suff != last_suff:
                plot_all_figures_for_suffex(last_suff, cur_fig_names)
                cur_fig_names = []

            cur_fig_names.append(name)
            last_suff = suff
        if last_suff:
            plot_all_figures_for_suffex(last_suff, cur_fig_names)


    def on_validation_epoch_end(self, trainer, pl_module):
        logger = self.loss_loggers[type(pl_module.logger)]
        logger()

class ImageLogger(Callback):
    """
    A callback for logging sample images for every epoch.
    The calss has options of loging images from training and validation on wandb, local, test_tube ... etc.


    """
    def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.logger_log_images = {
            pl.loggers.WandbLogger: self._wandb,
            # pl.loggers.TestTubeLogger: self._testtube, #disable TestTubeLogger for now
        }
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp

    @rank_zero_only
    def _wandb(self, pl_module, images, batch_idx, split, global_step, ncol=6):
        """
        Log images with the help of wandb by combining every split images into one frame
        Params:
            pl_module: pytorch model
            images: images to be logged
            batch_idx:
            split: how many images per frames
        """
        grids = dict()
        for k in images:
            if isinstance(images[k], str):
                continue
            elif isinstance(images[k], torch.Tensor):
                sz = images[k].shape[0]
                grid = torchvision.utils.make_grid(images[k],
                                nrow=ncol,
                                normalize=True,
                                value_range=(-1, 1))
            else:
                grid = images[k]
            caption = None
            if f'{k}_caption' in images:
                caption = images[f'{k}_caption']

            grids[f"{split}/{k}"] = wandb.Image(grid, caption=caption)

        grids.update({"epoch":global_step})
        pl_module.logger.experiment.log(grids, commit=False)

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split, global_step):
        for k in images:
            if isinstance(images[k], str):
                continue
            elif isinstance(images[k], torch.Tensor):
                 grid = torchvision.utils.make_grid(images[k])
                 grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w
            else:
                grid = images[k]

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx, ncol=6):

        """
        pos-process images then save them locally by combining every split images into one frame then saving them locally
        """

        root = os.path.join(save_dir, "images", split)
        for k in images:
            if isinstance(images[k], str):
                continue
            elif isinstance(images[k], torch.Tensor):
                sz = images[k].shape[0]
                grid = torchvision.utils.make_grid(images[k], nrow=ncol)
                grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w
                grid = grid.transpose(0,1).transpose(1,2).squeeze(-1)
                grid = grid.numpy()
                grid = (grid*255).astype(np.uint8)
                grid = Image.fromarray(grid)
            else:
                grid = images[k]

            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            grid.save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        """
        obtain evaluated images from the model and pass them to the logger
        """
        log = (split == 'train' and self.check_frequency(batch_idx)) or \
                 (split == 'val' and self.check_frequency(batch_idx))
        if ( log and  # batch_idx % self.batch_freq == 0
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                self.max_images > 0):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()


            with torch.no_grad():
                # obtain a dict of inputs and reconst where it has the original images and target.
                # racon images are between -1 and 1
                images = pl_module.log_images(batch, split=split)

            for k in images: # iterate over keys (imputs, and reconst)
                if isinstance(images[k], torch.Tensor):
                    if len(images[k].shape) >= 2:
                        N = images[k].shape[0]
                        N = min(images[k].shape[0], self.max_images) # take only max_images images from batch to visualize
                        images[k] = images[k][:N]
                    images[k] = images[k].detach().cpu()
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            logger_log_images(pl_module, images, pl_module.global_step, split, global_step=pl_module.current_epoch)

            if is_train:
                pl_module.train()

    def check_frequency(self, batch_idx):
        if (batch_idx % self.batch_freq) == 0 or (batch_idx in self.log_steps):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.log_img(pl_module, batch, batch_idx, split="val")



if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()

    opt, unknown = parser.parse_known_args()

    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            idx = len(paths)-paths[::-1].index("logs")+1
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = get_ckpt_path(logdir)

        # opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs+opt.base

    # create nowtime and logdir
    if opt.resume and opt.resume_id:
        nowname = opt.resume_id
    else:
        if opt.name:
            name = "_"+opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_"+cfg_name
        else:
            name = ""

        nowname = name+opt.postfix+now
        logdir = os.path.join("logs", nowname)

    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]  # N configs, one for each given config file name (i.e base)
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli) # merge all configs

        lightning_config = config.pop("lightning", OmegaConf.create()) # prepare a config instance to be passed to torch lightning
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        # default to ddp
        trainer_config["distributed_backend"] = "ddp"
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k) # --gpus for example is not a default config
        if not "gpus" in trainer_config:
            del trainer_config["distributed_backend"]
            cpu = True
        else:
            gpuinfo = trainer_config["gpus"]
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        print("Initializing the model")
        model = instantiate_from_config(config.model)

        print("Finished initializing the model")

        if opt.resume:
            print(f"Loading model from ckpt f{ckpt}")
            model = model.load_from_checkpoint(ckpt, **config.model.params, strict=opt.strict)
            print(f"Ckpt loaded from f{ckpt}")
       
        # trainer and callbacks
        trainer_kwargs = dict()

        #default logger configs
        default_logger_cfgs = {
           "wandb": {
               "target": "pytorch_lightning.loggers.WandbLogger",
               "params": {
                   "project":config.project_title,
                   "name": nowname,
                   "save_dir": logdir,
                   "offline": opt.debug,
                   "id": nowname,
               }
           },
           # testtube was not tested
            "testtube": {
               "target": "pytorch_lightning.loggers.TestTubeLogger",
               "params": {
                   "name": "testtube",
                   "save_dir": logdir,
                   "create_git_tag" : True,
               }

            },
        }
        os.makedirs(logdir, exist_ok=True)
        os.makedirs(os.path.join(logdir, 'wandb'), exist_ok=True)
        default_logger_cfg = default_logger_cfgs["wandb"]
        if 'logger' in lightning_config:
            logger_cfg = lightning_config.logger  
        else:
            logger_cfg = OmegaConf.create()
        
        logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
        trainer_kwargs["logger"] = instantiate_from_config(logger_cfg)


        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:06}",
                "verbose": True,
                "save_last": True,
            }
        }
        if hasattr(model, "monitor"):
            print(f"Monitoring {model.monitor} as checkpoint metric.")
            default_modelckpt_cfg["params"]["monitor"] = model.monitor
            default_modelckpt_cfg["params"]["save_top_k"] = 3

        if 'modelcheckpoint' in lightning_config:
            modelckpt_cfg = lightning_config.modelcheckpoint 
        else:
            modelckpt_cfg = OmegaConf.create()
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
        print("Model ckpt args:", modelckpt_cfg)
        trainer_kwargs["checkpoint_callback"] = instantiate_from_config(modelckpt_cfg)

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                    "name" : nowname,
                    "tags": opt.tags,
                    "id": nowname,
                }
            },
            "image_logger": {
                "target": "main.ImageLogger",
                "params": {
                    "batch_frequency": 1500, # 750
                    "max_images": 15, # 5 * 3 frames
                    "clamp": True
                }
            },
            "loss_logger":{
                "target": "main.LossLogger",
                "params": {
                    "log_dir": logdir,
                }
            },
        }
        if 'callbacks' in lightning_config:
            callbacks_cfg = lightning_config.callbacks 
        else:
            callbacks_cfg = OmegaConf.create()
            
        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        trainer_kwargs["callbacks"] = [
            instantiate_from_config(callbacks_cfg[k], model=model) if k == 'setup_callback' else instantiate_from_config(callbacks_cfg[k])
            for k in callbacks_cfg]
        trainer_kwargs["callbacks"].append(instantiate_from_config(modelckpt_cfg))
        
        # no need to pass model checkpoint separately
        del trainer_kwargs['checkpoint_callback']
        
        if opt.resume:
            trainer = Trainer(**vars(trainer_opt), **trainer_kwargs, resume_from_checkpoint=ckpt)
        else:
            trainer = Trainer(**vars(trainer_opt), **trainer_kwargs)
        # configure learning rate
        bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
        print("Batch size:", bs)
        if not cpu:
            ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
        else:
            ngpu = 1
        
        accumulate_grad_batches = lightning_config.trainer.get('accumulate_grad_batches', 1)
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        print("Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
            model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))


        data = instantiate_from_config(config.data, num_gpus=ngpu)
        # load data according to config into training and testing, and wrap them with the model
        data.prepare_data()
        data.setup()
        print("Done seting up data", flush=True)

        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            # run all checkpoint hooks
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)
                # wandb.finish()

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb; pudb.set_trace()

        import signal # for asynchronous events
        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)


        # model unlearnable params
        print("Model non-learnable params:",  sum(p.numel() for p in model.parameters() if not p.requires_grad))
        print("Size of training dataset:", len(data.datasets['train']))
        print("Size of validation dataset:", len(data.datasets['validation']))
        # run
        if opt.train:
            try:
                trainer.fit(model, data)
            except Exception:
                melk()
                raise
        if not opt.no_test and not trainer.interrupted:
            trainer.test(model, data)
    except Exception:
        if opt.debug and trainer.global_rank==0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank==0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
