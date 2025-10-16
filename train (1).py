import argparse
from argparse import ArgumentParser
import os
import sys
import glob
import warnings

# 忽略requests依赖警告
warnings.filterwarnings("ignore", category=UserWarning, module="requests")

import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch

from sgmse.backbones.shared import BackboneRegistry
from sgmse.data_module_vi import SpecsDataModule
from sgmse.model import StochasticRegenerationModel

from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint

import numpy as np
import random

class CheckpointEveryNSteps(pl.Callback):
    def __init__(self, save_step_frequency, prefix="N-Step-Checkpoint", use_modelcheckpoint_filename=False):
        self.save_step_frequency = save_step_frequency
        self.prefix = prefix
        self.use_modelcheckpoint_filename = use_modelcheckpoint_filename

    def on_train_batch_end(self, trainer: pl.Trainer, *args, **kwargs):
        epoch = trainer.current_epoch
        global_step = trainer.global_step
        if global_step % self.save_step_frequency == 0:
            if self.use_modelcheckpoint_filename:
                filename = trainer.checkpoint_callback.filename
            else:
                filename = f"{self.prefix}_{epoch=}_{global_step=}.ckpt"
            ckpt_path = os.path.join(trainer.checkpoint_callback.dirpath, filename)
            trainer.save_checkpoint(ckpt_path)


def get_argparse_groups(parser):
    groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        groups[group.title] = argparse.Namespace(**group_dict)
    return groups

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

seed_everything(20)


if __name__ == '__main__':
    base_parser = ArgumentParser(add_help=False)
    parser = ArgumentParser()
    DATA_BASE_DIR = r"my_test_data//train"
    parser.set_defaults(base_dir=DATA_BASE_DIR)

    for parser_ in (base_parser, parser):
        parser_.add_argument("--mode", default="regen-joint-training", choices=["regen-joint-freeze", "score-only", "denoiser-only", "regen-freeze-denoiser", "regen-joint-training"])
        parser_.add_argument("--backbone_denoiser", type=str, choices=["none"] + BackboneRegistry.get_all_names(), default="ncsnpp_crossatt")
        parser_.add_argument("--pretrained_denoiser", default=None)
        parser_.add_argument("--backbone_score", type=str, choices=["none"] + BackboneRegistry.get_all_names(), default="ncsnpp_crossatt")
        parser_.add_argument("--pretrained_score", default=None)
        parser_.add_argument("--nolog", action="store_true")
        parser_.add_argument("--logstdout", action="store_true")
        parser_.add_argument("--discriminatively", action="store_true")
        parser_.add_argument("--debug_data", action="store_true", help="在数据加载阶段打印样本调试信息")

    temp_args, _ = base_parser.parse_known_args()
    if "regen" in temp_args.mode:
        model_cls = StochasticRegenerationModel

    backbone_cls_denoiser = BackboneRegistry.get_by_name(temp_args.backbone_denoiser) if temp_args.backbone_denoiser != "none" else None
    backbone_cls_score = BackboneRegistry.get_by_name(temp_args.backbone_score) if temp_args.backbone_score != "none" else None

    parser = pl.Trainer.add_argparse_args(parser)
    model_cls.add_argparse_args(parser.add_argument_group(model_cls.__name__, description=model_cls.__name__))

    if temp_args.backbone_denoiser != "none":
        backbone_cls_denoiser.add_argparse_args(parser.add_argument_group("BackboneDenoiser", description=backbone_cls_denoiser.__name__))
    else:
        parser.add_argument_group("BackboneDenoiser", description="none")

    if temp_args.backbone_score != "none":
        backbone_cls_score.add_argparse_args(parser.add_argument_group("BackboneScore", description=backbone_cls_score.__name__))
    else:
        parser.add_argument_group("BackboneScore", description="none")

    data_module_cls = SpecsDataModule
    data_module_cls.add_argparse_args(parser.add_argument_group("DataModule", description=data_module_cls.__name__))
    args = parser.parse_args()
    arg_groups = get_argparse_groups(parser)

    arg_groups['DataModule'].format = "custom_data"

    if (not hasattr(arg_groups['DataModule'], 'base_dir')) or (not arg_groups['DataModule'].base_dir) or (arg_groups['DataModule'].base_dir == "/mnt/scratch/datasets/new_avspeech"):
        arg_groups['DataModule'].base_dir = DATA_BASE_DIR

    print("正在初始化数据模块...")
    print(f"数据路径: {arg_groups['DataModule'].base_dir}")
    print(f"格式: custom_data")
    print(f"批次大小: {arg_groups['DataModule'].batch_size}")
    print(f"工作进程数: {arg_groups['DataModule'].num_workers}")

    if not os.path.exists(arg_groups['DataModule'].base_dir):
        print(f"错误：数据路径不存在: {arg_groups['DataModule'].base_dir}")
        sys.exit(1)

    required_dirs = ['clean_audio', 'noisy_audio', 'lip_features']
    for dir_name in required_dirs:
        dir_path = os.path.join(arg_groups['DataModule'].base_dir, dir_name)
        if not os.path.exists(dir_path):
            print(f"警告：缺少必要的数据文件夹: {dir_path}")
        else:
            print(f"找到数据文件夹: {dir_path}")

    print("正在初始化模型...")
    if "regen" in temp_args.mode:
        try:
            model = model_cls(
                mode=args.mode,
                backbone_denoiser=args.backbone_denoiser,
                backbone_score=args.backbone_score,
                data_module_cls=data_module_cls,
                **{
                    **vars(arg_groups['StochasticRegenerationModel']),
                    **vars(arg_groups['BackboneDenoiser']),
                    **vars(arg_groups['BackboneScore']),
                    **vars(arg_groups['DataModule'])
                },
                nolog=args.nolog
            )
            print("模型创建成功！")
        except Exception as e:
            print(f"模型创建失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        if temp_args.pretrained_denoiser is not None:
            model.load_denoiser_model(temp_args.pretrained_denoiser)
        if temp_args.pretrained_score is not None:
            model.load_score_model(temp_args.pretrained_score)

        print("正在准备数据模块...")
        try:
            model.datamodule.prepare_data()
            print("prepare_data() 完成")
        except Exception as e:
            print(f"prepare_data() 失败: {e}")
            import traceback
            traceback.print_exc()

        print("正在设置数据模块...")
        try:
            model.datamodule.setup()
            print("setup() 完成")

            print("检查数据集...")
            if hasattr(model.datamodule, 'train_set'):
                print(f"训练集大小: {len(model.datamodule.train_set)}")
                if args.debug_data:
                    try:
                        for i in range(min(3, len(model.datamodule.train_set))):
                            sample = model.datamodule.train_set[i]
                            print(f"样本 {i}: 类型={type(sample)}, 内容={sample if isinstance(sample, (list, tuple, dict)) else '非结构化对象'}")
                    except Exception as e:
                        print(f"⚠️ 访问 train_set 出错: {e}")
            else:
                print("警告: 没有找到训练集")

            if hasattr(model.datamodule, 'valid_set'):
                print(f"验证集大小: {len(model.datamodule.valid_set)}")
            else:
                print("警告: 没有找到验证集")
        except Exception as e:
            print(f"数据模块设置失败: {e}")
            import traceback
            traceback.print_exc()

    logger = TensorBoardLogger(save_dir=f"./.logs/", name="debug_log", flush_secs=30) if not args.nolog else None

    callbacks = []
    callbacks.append(TQDMProgressBar(refresh_rate=10))
    if not args.nolog:
        callbacks.append(ModelCheckpoint(dirpath=os.path.join(logger.log_dir, "checkpoints"), save_last=True, save_top_k=1, monitor="valid_loss", filename='{epoch}'))
        callbacks.append(ModelCheckpoint(dirpath=os.path.join(logger.log_dir, "checkpoints"), save_top_k=1, monitor="ValidationPESQ", mode="max", filename='{epoch}-{pesq:.2f}'))
        callbacks.append(CheckpointEveryNSteps(save_step_frequency=30000))

    if torch.cuda.is_available():
        accelerator = 'gpu'
        devices = torch.cuda.device_count()
    else:
        accelerator = 'cpu'
        devices = 1

    try:
        trainer = pl.Trainer.from_argparse_args(
            arg_groups['pl.Trainer'],
            strategy=DDPStrategy(find_unused_parameters=True) if accelerator == 'gpu' and devices > 1 else None,
            accelerator=accelerator,
            devices=devices,
            logger=logger,
            log_every_n_steps=5,
            num_sanity_val_steps=2,
            callbacks=callbacks,
            max_epochs=40,
            val_check_interval=0.25,
            enable_progress_bar=True,
            enable_model_summary=True,
        )
    except Exception as e:
        print(f"训练器初始化失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("开始训练...")
    try:
        trainer.fit(model)
    except KeyboardInterrupt:
        print("训练被用户中断")
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
