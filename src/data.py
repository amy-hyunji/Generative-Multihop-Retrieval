import os
import sys
import math
import json
import copy
import torch
import pickle
import pandas as pd
import numpy as np

from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

class GenRetDataset(Dataset):
    def __init__(self, tokenizer, hparams, split):

        if split == "train":
            data_path = hparams.train_file 
        elif split == "validation":
            data_path = hparams.dev_file
        elif split == "test":
            data_path = hparams.test_file
        else:
            raise NotImplementedError(f"Check the Split: {split}")

        assert data_path.endswith(".csv"), f"only CSV form is supported"

        self.dataset = pd.read_csv(os.path.join(hparams.dataset, data_path))
        
        if self.hparams.setting == "ret_dynamic":
            self.dataset = self.convert_fixed_to_dynamic(self.dataset)
        
        self.len = len(self.dataset["input"])

        self.tokenizer = tokenizer 
        self.hparams = hparams 

    def __len__(self):
        return self.len

    def convert_to_features(self, batch, idx):
        input_ = batch["input"]
        output_ = batch["output"]

        source = self.tokenizer.batch_encode_plus(
            [input_],
            max_length=self.hparams.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        target = self.tokenizer.batch_encode_plus(
            [output_],
            max_length=self.hparams.max_output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        if idx == 0:
            print("#"*60)
            print(f"input: {input_}")
            print(f"output: {output_}")
            print("#"*60)

        return source, target

    def convert_fixed_to_dynamic(self, dataset):
        input2output = defaultdict(list) 
        for _input, _output in zip(dataset["input"], dataset["output"]):
            input2output[_input].append(_output)    

        df = {"input": [], "output": []}
        for _input, _output_list in input2output.items():
            for _output in _output_list: 
                df["input"].append(_input)
                df["output"].append(_output)
            df["input"].append(_input)
            df["output"].append("DONE")

        return pd.DataFrame(df)

    def __getitem__(self, idx):
        batch = self.dataset.iloc[idx]
        source, target = self.convert_to_features(batch, idx)

        return {
            "source_ids": source["input_ids"].squeeze(),
            "source_mask": source["attention_mask"].squeeze(),
            "target_ids": target["input_ids"].squeeze(),
            "target_mask": target["attention_mask"].squeeze(),
            "input": batch["input"], 
            "output": batch["output"] 
        }

