import os
import warnings
import logging
from typing import Mapping, List
from pprint import pprint

# Numpy and Pandas 
import numpy as np
import pandas as pd

# PyTorch 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Transformers 
from transformers import AutoConfig, AutoModel, AutoTokenizer

# Catalyst
from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import AccuracyCallback, F1ScoreCallback, OptimizerCallback
from catalyst.dl.callbacks import CheckpointCallback, InferCallback
from catalyst.utils import set_global_seed, prepare_cudnn

from model import BertForSequenceClassification

from data import TextClassificationDataset
# model = BertForSequenceClassification("bert-base-uncased")

# checkpoint = torch.load("logdir/checkpoints/train.2.pth")
# model.load_state_dict(torch.load('logdir/checkpoints/train.2.pth')['state_dict'])
# model.eval()

def clearConsole():
    command = 'clear'
    if os.name in ('nt', 'dos'): 
        command = 'cls'
    os.system(command)

MODEL_NAME = 'distilbert-base-uncased'

test_df = pd.read_csv("data/motions_laws/test.csv").fillna('')


test_dataset = TextClassificationDataset(
    texts=test_df['description'].values.tolist(),
    labels=None,
    label_dict=None,
    max_seq_length=32,
    model_name=MODEL_NAME
)


test_loaders = {
    "test": DataLoader(dataset=test_dataset,
                        batch_size=16, 
                        shuffle=False) 
}

model = BertForSequenceClassification(pretrained_model_name=MODEL_NAME,
                                            num_classes=3)

runner = SupervisedRunner(
    input_key=(
        "features",
        "attention_mask"
    )
)

runner.infer(
    model=model,
    loaders=test_loaders,
    callbacks=[
        CheckpointCallback(
            resume="logdir/checkpoints/best.pth"
        ),
        InferCallback(),
    ],   
    verbose=True
)


predicted_probs = runner.callbacks[0].predictions['logits']
predictions = []






#convert predictions from confidence to T, F
for pred in predicted_probs:
    max = 0
    maxidx = 0
    # print(pred)
    for i in range(len(pred)):
        
        if pred[i] > max:
            max = pred[i]
            maxidx = i
        # print(maxidx)
        
    if maxidx == 2: predictions.append("Other")
    elif maxidx == 1: predictions.append("Motion to Dismiss")        
    else: predictions.append("Motion for Summary Judgement")




### Calculate accuracy on test set
import csv

filename = 'data/motions_laws/test.csv'
# print(len(predictions))


with open(filename, 'r') as csvfile:
    datareader = csv.reader(csvfile)
    total_dismiss = 0
    total_sumjudge = 0
    total_other = 0
    correct_dismiss = 0
    correct_sumjudge = 0
    correct_other = 0
    false_dismiss = 0
    false_sumjudge = 0
    false_other = 0

    total_other_guessed = 0
    total_dismiss_guessed = 0
    total_sumjudge_guessed = 0    
    
    wrong = []
    c=0
    correct=0
    first = True
    for row in datareader:
        if first:
            first=False
        else:
            if row[4] == predictions[c]:
                correct += 1
            else:
                wrong.append(row)
            
            if row[4] == 'Other':
                total_other += 1
                if row[4] == predictions[c]:
                    correct_other += 1
                else:
                    false_other += 1
            
            if row[4] == 'Motion to Dismiss':
                total_dismiss += 1
                if row[4] == predictions[c]:
                    correct_dismiss += 1
                else:
                    false_dismiss += 1
            
            if row[4] == 'Motion for Summary Judgement':
                total_sumjudge += 1
                if row[4] == predictions[c]:
                    correct_sumjudge += 1
                else:
                    false_sumjudge += 1

            if predictions[c] == "Other": total_other_guessed += 1
            elif predictions[c] == "Motion to Dismiss": total_dismiss_guessed += 1
            else: total_sumjudge_guessed += 1


            c+=1


other_precision = correct_other / total_other_guessed
dismiss_precision = correct_dismiss / total_dismiss_guessed
sumjudge_precision = correct_sumjudge / total_sumjudge_guessed

other_recall = correct_other / total_other
dismiss_recall = correct_dismiss / total_dismiss
sumjudge_recall = correct_sumjudge / total_sumjudge


precision = (other_precision + dismiss_precision + sumjudge_precision) / 3
recall = (other_recall + dismiss_recall + sumjudge_recall) / 3





clearConsole()
print("\n\nAccuracy on Test Set: " + str(correct/c))
print("Precision: " + str(precision))
print("Recall: "+ str(recall))
print("F-Measure: " + str((2*precision*recall)/(precision+recall)))
print("----------------")
print("Total Observations: " + str(c))
print("Total Correct Observations:" + str(correct))
print("\n\n")
