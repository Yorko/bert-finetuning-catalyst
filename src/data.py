import logging
from pathlib import Path
from typing import List, Mapping, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from catalyst.utils import set_global_seed


class TextClassificationDataset(Dataset):
    """
    Wrapper around Torch Dataset to perform text classification
    """

    def __init__(
        self,
        texts: List[str],
        labels: List[str] = None,
        label_dict: Mapping[str, int] = None,
        max_seq_length: int = 512,
        model_name: str = "distilbert-base-uncased",
    ):
        """
        Args:
            texts (List[str]): a list with texts to classify or to train the
                classifier on
            labels List[str]: a list with classification labels (optional)
            label_dict (dict): a dictionary mapping class names to class ids,
                to be passed to the validation data (optional)
            max_seq_length (int): maximal sequence length in tokens,
                texts will be stripped to this length
            model_name (str): transformer model name, needed to perform
                appropriate tokenization

        """

        self.texts = texts
        self.labels = labels
        self.label_dict = label_dict
        self.max_seq_length = max_seq_length

        if self.label_dict is None and labels is not None:
            # {'class1': 0, 'class2': 1, 'class3': 2, ...}
            # using this instead of `sklearn.preprocessing.LabelEncoder`
            # no easily handle unknown target values
            self.label_dict = dict(zip(sorted(set(labels)), range(len(set(labels)))))

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # suppresses tokenizer warnings
        logging.getLogger("transformers.tokenization_utils").setLevel(logging.FATAL)

        # special tokens for transformers
        # in the simplest case a [CLS] token is added in the beginning
        # and [SEP] token is added in the end of a piece of text
        # [CLS] <indexes text tokens> [SEP] .. <[PAD]>
        self.sep_vid = self.tokenizer.vocab["[SEP]"]
        self.cls_vid = self.tokenizer.vocab["[CLS]"]
        self.pad_vid = self.tokenizer.vocab["[PAD]"]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.texts)

    def __getitem__(self, index) -> Mapping[str, torch.Tensor]:
        """Gets element of the dataset

        Args:
            index (int): index of the element in the dataset
        Returns:
            Single element by index
        """

        # encoding the text
        x = self.texts[index]

        # a dictionary with `input_ids` and `attention_mask` as keys
        output_dict = self.tokenizer.encode_plus(
            x,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.max_seq_length,
            return_tensors="pt",
            truncation=True,
            return_attention_mask=True,
        )

        # for Catalyst, there needs to be a key called features
        output_dict["features"] = output_dict["input_ids"].squeeze(0)
        del output_dict["input_ids"]

        # encoding target
        if self.labels is not None:
            y = self.labels[index]
            y_encoded = torch.Tensor([self.label_dict.get(y, -1)]).long().squeeze(0)
            output_dict["targets"] = y_encoded

        return output_dict


def read_data(params: dict) -> Tuple[dict, dict]:
    """
    A custom function that reads data from CSV files, creates PyTorch datasets and
    data loaders. The output is provided to be easily used with Catalyst

    :param params: a dictionary read from the config.yml file
    :return: a tuple with 2 dictionaries
    """
    # reading CSV files to Pandas dataframes
    train_df = pd.read_csv(
        Path(params["data"]["path_to_data"]) / params["data"]["train_filename"]
    )
    valid_df = pd.read_csv(
        Path(params["data"]["path_to_data"]) / params["data"]["validation_filename"]
    )
    test_df = pd.read_csv(
        Path(params["data"]["path_to_data"]) / params["data"]["test_filename"]
    )

    # creating PyTorch Datasets
    train_dataset = TextClassificationDataset(
        texts=train_df[params["data"]["text_field_name"]].values.tolist(),
        labels=train_df[params["data"]["label_field_name"]].values,
        max_seq_length=params["model"]["max_seq_length"],
        model_name=params["model"]["model_name"],
    )

    valid_dataset = TextClassificationDataset(
        texts=valid_df[params["data"]["text_field_name"]].values.tolist(),
        labels=valid_df[params["data"]["label_field_name"]].values,
        max_seq_length=params["model"]["max_seq_length"],
        model_name=params["model"]["model_name"],
    )

    test_dataset = TextClassificationDataset(
        texts=test_df[params["data"]["text_field_name"]].values.tolist(),
        labels=test_df[params["data"]["label_field_name"]].values,
        max_seq_length=params["model"]["max_seq_length"],
        model_name=params["model"]["model_name"],
    )

    set_global_seed(params["general"]["seed"])

    # creating PyTorch data loaders and placing them in dictionaries (for Catalyst)
    train_val_loaders = {
        "train": DataLoader(
            dataset=train_dataset,
            batch_size=params["training"]["batch_size"],
            shuffle=True,
        ),
        "valid": DataLoader(
            dataset=valid_dataset,
            batch_size=params["training"]["batch_size"],
            shuffle=False,
        ),
    }

    test_loaders = {
        "test": DataLoader(
            dataset=test_dataset,
            batch_size=params["training"]["batch_size"],
            shuffle=False,
        )
    }

    return train_val_loaders, test_loaders
