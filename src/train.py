import os
import sys
import json
import torch
import random
import pickle
import textwrap
import argparse

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch.distributed as dist

from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint 
from pytorch_lightning.plugins import DDPPlugin,DeepSpeedPlugin

from models import T5GenRet 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(args, train_params):
    sys.setrecursionlimit(10000)
    set_seed(args.seed)

    model = T5GenRet(args)
    trainer = pl.Trainer(**train_params)

    if args.do_train:
        if torch.cuda.current_device() == 0:
            print("=== Start Train ===")
        if args.resume_from_checkpoint is None:
            trainer.fit(model)
        else:
            trainer.fit(model, ckpt_path=args.resume_from_checkpoint)
        trainer.save_checkpoint(os.path.join(args.output_dir, "last.ckpt"))
    if args.do_test:
        if torch.cuda.current_device() == 0:
            print("=== Start Test ===")
        trainer.test(model)
    
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', default=None, type=str)
    arg_ = parser.parse_args()
    if arg_.config == None:
        raise NotImplementedError("Input Config File is Needed!")

    with open(arg_.config) as config_file:
        hparam = json.load(config_file)
    hparam = argparse.Namespace(**hparam)

    if hparam.wandb_log:
        wandb_logger = WandbLogger(project=hparam.wandb_project, name=hparam.wandb_run_name)
    else:
        wandb_logger = None

    # Set Configurations
    args_dict = dict(
        output_dir=hparam.output_dir,
        dataset=hparam.dataset,
        model_name_or_path=hparam.model,
        tokenizer_name_or_path=hparam.model,
        max_input_length=hparam.max_input_length,
        max_output_length=hparam.max_output_length,
        learning_rate=hparam.learning_rate, 
        lr_scheduler=hparam.lr_scheduler,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=0,
        num_train_epochs=hparam.num_train_epochs, 
        train_batch_size=hparam.train_batch_size, 
        eval_batch_size=hparam.eval_batch_size,
        gradient_accumulation_steps=hparam.gradient_accumulation_steps,
        n_gpu=hparam.n_gpu,
        num_workers=hparam.num_workers,
        resume_from_checkpoint=hparam.resume_from_checkpoint,
        seed=hparam.seed,
        check_val_every_n_epoch=hparam.check_val_every_n_epoch,
        train_file = hparam.train_file,
        dev_file = hparam.dev_file,
        test_file = hparam.test_file,
        prefix_tree = hparam.prefix_tree,
        constrained_decoding = hparam.constrained_decoding,
        do_train = hparam.do_train,
        do_test = hparam.do_test,
        setting = hparam.setting,
        test_model_path = hparam.test_model_path,
        test_tokenizer_path = hparam.test_model_path,
        beam_size=hparam.beam_size,
        ret_num=hparam.ret_num,
        accelerator=hparam.accelerator
    )
    args = argparse.Namespace(**args_dict)
    if torch.cuda.current_device() == 0:
        print('#'*80)
        print(args)
        print('#'*80)

    assert args.lr_scheduler in ["constant", "exponential"]
    assert args.setting in ["ret_fixed", "ret_dynamic", "LM_mem", "multihop_mem"]
    assert (args.do_train and not args.do_test) or (args.do_test and not args.do_train), f"Choose between [do_train | do_test]" 
    assert args.beam_size > 0 and args.ret_num > 0

    if args.setting in ["LM_mem", "multihop_mem"]:
        assert not args.constrained_decoding, "Don't use constrained decoding for memorization methods" 

    # V1: only support single gpu for test setting
    if args.do_test:
        args.n_gpu = 1
        
    # Save Checkpoint 
    callbacks = [pl.callbacks.ModelCheckpoint(monitor="val_em_score", mode="max", dirpath=args.output_dir, filename='{epoch:02d}-{val_loss:.2f}', save_top_k=5)]

    # Logging Learning Rate Scheduling
    if args.learning_rate is not "constant":
        callbacks.append(pl.callbacks.LearningRateMonitor())

    if args.accelerator == "ddp":
        plugins = DDPPlugin(find_unused_parameters=False)
        fp_16 = False 
    elif args.accelerator == "deepspeed":
        plugins = DeepSpeedPlugin(stage=2, load_full_weights=True)
        fp_16 = True
    else:
        raise NotImplementedError("Choose accelerator between [ddp | deepspeed]")

    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        strategy=plugins,
        max_epochs=args.num_train_epochs,
        precision=16 if fp_16 else 32,
        default_root_dir=args.output_dir,
        checkpoint_callback=True,
        val_check_interval=1.0,
        logger=wandb_logger,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        callbacks=callbacks
    )

    main(args, train_params)    