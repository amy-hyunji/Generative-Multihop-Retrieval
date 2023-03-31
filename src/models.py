import re
import os
import time
import h5py
import json
import copy
import torch
import pickle
import requests
import string
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch.distributed as dist

from typing import *
from tqdm import tqdm
from data import GenRetDataset
from datasets import load_metric
from transformers import (
    Adafactor,
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5Config,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import DataLoader, RandomSampler


class T5GenRet(pl.LightningModule):
    def __init__(self, args):
        super(T5GenRet, self).__init__()
        self.save_hyperparameters(args)

        if self.hparams.do_train:
            self.model = T5ForConditionalGeneration.from_pretrained(
                self.hparams.model_name_or_path
            )
            self.tokenizer = T5Tokenizer.from_pretrained(
                self.hparams.tokenizer_name_or_path
            )

            if self.hparams.setting in ['ret_fixed', 'ret_dynamic']:
                sp_tokens = ["<EVIDENCE>", "</EVIDENCE>", "<QUESTION>", "</QUESTION>"]
                if self.hparams.setting == "ret_dynamic":
                    sp_tokens += ["DONE"]
                self.tokenizer.add_tokens(sp_tokens, special_tokens=True)
                self.model.resize_token_embeddings(len(self.tokenizer))

            
        if self.hparams.do_test:
            self.model = T5ForConditionalGeneration.from_pretrained(
                self.hparams.test_model_path
            )
            self.tokenizer = T5Tokenizer.from_pretrained(
                self.hparams.test_tokenizer_path
            )
            self.test = {"input": [], "output": [], "predict": [], "em": [], "recall": []}

        self.output_dir = self.hparams.output_dir
        self.max_input_length = self.hparams.max_input_length
        self.max_output_length = self.hparams.max_output_length

        self.constrained_decoding = self.hparams.constrained_decoding

        if self.constrained_decoding:
            self.prefix_tree = pickle.load(
                open(os.path.join(self.hparams.dataset, self.hparams.prefix_tree), "rb")
            )
        else:
            self.prefix_tree = {}

        if self.hparams.setting in ["ret_dynamic", "ret_fixed"]:
            self.em_score_list = []; self.recall_score_list = []
        elif self.hparams.setting in ["LM_mem", "multihop_mem"]:
            self.f1_score_list = []
        else:
            raise NotImplementedError("Check the Setting!")

    def get_dataset(self, split):
        assert split in ["train", "validation", "test"]
        dataset = GenRetDataset(
            tokenizer=self.tokenizer, hparams=self.hparams, split=split
        )
        return dataset

    def train_dataloader(self):
        train_dataset = self.get_dataset(split="train")
        dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.hparams.train_batch_size,
            drop_last=True,
            num_workers=self.hparams.num_workers,
        )
        return dataloader

    def val_dataloader(self):
        val_dataset = self.get_dataset(split="validation")
        dataloader = DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=self.hparams.eval_batch_size,
            drop_last=False,
            num_workers=self.hparams.num_workers,
        )
        return dataloader

    def test_dataloader(self):
        test_dataloader = self.get_dataset(split="test")
        dataloader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=self.hparams.eval_batch_size,
            drop_last=False,
            num_workers=self.hparams.num_workers,
        )
        return dataloader

    def _loss(self, batch):
        lm_labels = copy.deepcopy(batch["target_ids"])
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch["target_mask"],
        )
        loss = outputs[0]
        return loss        

    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        lm_labels=None,
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )

    def training_step(self, batch, batch_idx):
        loss = self._loss(batch)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        return loss

    def _get_from_trie(self, input_ids, trie_dict):
        if len(input_ids) == 0:
            output = list(trie_dict.keys())
            return output
        elif input_ids[0] in trie_dict:
            return self._get_from_trie(input_ids[1:], trie_dict[input_ids[0]])
        else:
            return []

    def get(self, input_ids):
        trie_dict = self.prefix_tree
        return self._get_from_trie(input_ids, trie_dict)

    def _generate_wo_constraint(self, input_ids, input_attn, target_attn=None):
        generated_ids = self.model.generate(
            input_ids, 
            attention_mask=input_attn,
            use_cache=True,
            decoder_attention_mask=target_attn,
            max_length=self.max_output_length,
            num_beams=self.hparams.beam_size,
            num_return_sequences=self.hparams.beam_size,
            early_stopping=True
        )
        return generated_ids

    def _generate_w_constraint(self, input_ids, input_attn, target_attn=None):
        generated_ids = self.model.generate(
            input_ids,
            attention_mask=input_attn,
            use_cache=True,
            decoder_attention_mask=target_attn,
            max_length=self.max_output_length,
            num_beams=self.hparams.beam_size,
            num_return_sequences=self.hparams.beam_size,
            early_stopping=True,
            prefix_allowed_tokens_fn=lambda batch_id, ids: self.get(
                ids.tolist(),
            )
        )
        return generated_ids

    def _score(self, batch):
        if self.constrained_decoding:
            generated_ids = self._generate_w_constraint(batch["source_ids"], batch["source_mask"], batch["target_mask"])
        else:                
            generated_ids = self._generate_wo_constraint(batch["source_ids"], batch["source_mask"], batch["target_mask"])
        _preds = np.array(self.ids_to_text(generated_ids))
        preds = list(_preds.reshape((-1, self.hparams.beam_size)))
        targets = self.ids_to_text(batch["target_ids"])
        inputs = self.ids_to_text(batch["source_ids"])
        assert len(preds) == len(targets) == len(inputs)

        em_score, recall_score = self.calculate_ret_scores(
            preds, targets
        )
        return em_score, recall_score

    def normalize_answer(self, s):
        def remove_sp(text):
            text = text.replace("<EVIDENCE>", "").replace("</EVIDENCE>", "").replace("<QUESTION>", "").replace("</QUESTION>", "")
            return text

        def remove_space_at_ends(text):
            while (text.endswith(" ")): 
                text = text[:-1]
            while (text.startswith(" ")):
                text = text[1:]
            return text

        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(remove_space_at_ends(remove_sp(s))))))

    def _remove_dup(self, sen_list):
        ret_list = []
        for sen in sen_list:
            if sen in ret_list:
                continue 
            else:
                ret_list.append(sen)
        return ret_list

    def _calculate_em(self, pred: str, gt: str):
        return 100 if pred == gt else 0

    def _calculate_recall(self, pred: List[str], gt: str):
        return 100 if gt in pred else 0

    def _calculate_hop_f1(self, pred: List[str], gt: List[str]):
        assert False

    def calculate_ret_scores(self, preds: List[str], targets: str, is_test=False):
        preds = self._remove_dup([[self.normalize_answer(_el) for _el in el] for el in preds])
        targets = [self.normalize_answer(el) for el in targets]            
        if is_test:
            raise NotImplementedError("Test Score for Ret is not Implemented Yet!")
            return _recall_list, _f1_list 
        else:
            _em_list = [self._calculate_em(pred[0], gt) for (pred, gt) in zip(preds, targets)]
            _recall_list = [self._calculate_recall(pred, gt) for (pred, gt) in zip(preds, targets)]
            return _em_list, _recall_list

    def lmap(self, f, x):
        return list(map(f, x))

    def ids_to_text(self, ids):
        text = self.tokenizer.batch_decode(
            ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )
        return self.lmap(str.strip, text)

    def _val_step(self, batch):
        _em_list, _recall_list = self._score(batch)
        return _em_list, _recall_list

    def _lm_mem_step(self, batch):
        ids = self._generate_wo_constraint(batch["source_ids"], batch["source_mask"])
        _preds = np.array(self.ids_to_text(generated_ids)) 
        preds = list(_preds.reshape((-1, self.hparams.beam_size)))
        targets = self.ids_to_text(batch["target_ids"])
        inputs = self.ids_to_text(batch["source_ids"])
        assert len(preds) == len(targets) == len(inputs)

        f1_score = self.calculate_mem_scores(
            preds, targets
        )
        return f1_score

    def _tok_f1_score(self, pred, gt):
        pred_tok = self.normalize_answer(pred).split()
        gt_tok = self.normalize_answer(gt).split()

        common = Counter(pred_tok) & Counter(gt_tok)
        num_same = sum(common.values())
        if num_same == 0: return 0
        
        precision = 1.0 * num_same / len(pred_tok)
        recall = 1.0 * num_same / len(gt_tok)
        f1 = (2*precision*recall) / (precision+recall) 
        return f1       

    def _calculate_tok_f1(self, pred_list, gt):
        return np.array([self._tok_f1_score(pred, gt) for pred in pred_list]).mean()

    def calculate_mem_scores(self, preds, targets):
        preds = self._remove_dup([[self.normalize_answer(_el) for _el in el] for el in preds])
        targets = [self.normalize_answer(el) for el in targets]
        _f1_list = [self._calculate_tok_f1(pred, gt) for (pred, gt) in zip(preds, targets)]
        return _f1_list

    def validation_step(self, batch, batch_idx):
        if self.hparams.setting in ["ret_dynamic", "ret_fixed"]:
            _em_list, _recall_list = self._val_step(batch)
            self.em_score_list.extend(_em_list) 
            self.recall_score_list.extend(_recall_list) 
            return
        elif self.hparams.setting in ["LM_mem", "multihop_mem"]:
            self.f1_score_list.extend(self._lm_mem_step(batch))
            return
        else:
            raise NotImplementedError(f"Check the Setting! {self.hparams.setting}")

    def validation_epoch_end(self, outputs):
        avg_em = np.mean(self.em_score_list)
        avg_recall = np.mean(self.recall_score_list)
        self.em_score_list = []; self.recall_score_list = []
        self.log(
            "val_em_score",
            avg_em,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )
        self.log(
            "val_recall_score",
            avg_recall,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )
        return 

    # TODO: remove previous branch
    def _dynamic_test_step(self, batch):
        q_list = batch["input"]
        gts = batch["output"]
        input_ids_list = batch["source_ids"] 
        input_att_list = batch["source_mask"] 

        state = [False]*len(q_list)
        gen_func = _generate_w_constraint if self.hparams.constrained_decoding else _generate_wo_constraint

        pred_dict = defaultdict(list)
        for _ in range(self.hparams.ret_num):
            generated_ids = gen_func(input_ids_list, input_att_list)
            preds = list(np.array(self.ids_to_text(generated_ids)).reshape((-1, self.hparams.beam_size)))
            assert len(preds) == len(q_list)
            for i in range(preds):
                cur_pred = preds[i]
                if state[i]:
                    continue 
                else:
                    pred_dict[i].append(cur_pred)
                    if cur_pred == "DONE":
                        state[i] = True 
                    else:
                        q_list[i] = f"{q_list[i]} <EVIDENCE> {preds[i]} </EVIDENCE>"
            input_ids_list, input_att_list = self._temporal_encode(q_list)

        recall_score, f1_score = self.calculate_ret_scores(                  
            list(pred_dict.values()), targets, is_test=True 
        )

        return em_list, recall_list, f1_list, pred_list

    def _fixed_test_step(self, batch):                                    
        q_list = batch["input"]
        gts = batch["output"]
        input_ids_list = batch["source_ids"]
        input_att_list = batch["source_mask"]

        gen_func = _generate_w_constraint if self.hparams.constrained_decoding else _generate_wo_constraint 

        pred_dict = defaultdict(list) 
        for _ in range(self.hparams.ret_num):
            generated_ids = gen_func(input_ids_list, input_att_list)                
            preds = list(np.array(self.ids_to_text(generated_ids)).reshape((-1, self.hparams.beam_size)))
            assert len(preds) == len(q_list)
            for i in range(preds):
                pred_dict[i].append(preds[i])
                q_list[i] = f"{q_list[i]} <EVIDENCE> {preds[i]} </EVIDENCE>"
            input_ids_list, input_att_list = self._temporal_encode(q_list) 

        recall_score, f1_score = self.calculate_ret_scores(                  
            list(pred_dict.values()), targets, is_test=True 
        )
        return em_score, recall_score, f1_score, list(pred_dict.values())

    def _temporal_encode(self, input_list: List[str]):            
        input_ids_list = []; input_att_list = []

        for q in input_list:
            tok_ret = self.tokenizer(q, return_tensors="pt")
            input_ids_list.append(tok_ret["input_ids"].to(self.device))
            input_att_list.append(tok_ret["attention_mask"].to(self.device))

        return input_ids_list, input_att_list

    def test_step(self, batch, batch_idx):
        q = batch["input"]
        evs = batch["output"]
        input_ids = batch["source_ids"]
        attention_mask = batch["source_mask"]
        if self.hparams.ret_setting == "fixed":
            em_list, recall_list, f1_list, pred_list = self._fixed_test_step(batch)
        elif self.hparams.ret_setting == "dynamic":
            em_list, recall_list, f1_list, pred_list = self._dynamic_test_step(batch)
        else:
            raise NotImplementedError("Choose Retrieval Setting from [fixed | dynamic]")

        self.test["input"].extend(q)
        self.test["output"].extend(evs)
        self.test["predict"].extend(pred_list)
        self.test["em"].extend(em_list)
        self.test["recall"].extend(recall_list)
        self.test["f1"].extend(f1_list)
        return 

    def test_epoch_end(self, outputs):            
        save_path = os.path.join(self.hparams.output_dir, f"beam_{self.hparams.beam_size}_output.json")
        os.makedirs(self.hparams.output_dir, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(self.test, f)
        print(f"Done Saving Test Output in {save_path}")

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = Adafactor(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            warmup_init=False,
            scale_parameter=False,
            relative_step=False,
        )
        self.opt = optimizer

        if self.hparams.lr_scheduler == "constant":
            return [optimizer]
        elif self.hparams.lr_scheduler == "exponential":
            len_data = len(self.train_dataloader())
            denominator = self.hparams.n_gpu
            steps_per_epoch = (
                (len_data // denominator) + 1
            ) // self.hparams.gradient_accumulation_steps
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.learning_rate,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.1,
                epochs=self.hparams.num_train_epochs,
                anneal_strategy="linear",
                cycle_momentum=False,
            )
            return [optimizer], [
                {"scheduler": scheduler, "interval": "step", "name": "learning_rate"}
            ]

    def on_save_checkpoint(self, checkpoint):
        save_path = os.path.join(
            self.output_dir,
            f"best_tfmr_epoch_{self.current_epoch}_step_{self.global_step}",
        )
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
