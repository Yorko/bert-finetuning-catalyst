from pathlib import Path

import numpy as np
import torch
import yaml
from catalyst.callbacks.metrics.accuracy import AccuracyCallback
from catalyst.dl import (
    CheckpointCallback,
    OptimizerCallback,
    SchedulerCallback,
    SupervisedRunner,
)
from catalyst.utils import prepare_cudnn, set_global_seed

from data import read_data
from model import BertForSequenceClassification
from utils import get_project_root

# loading config params
project_root: Path = get_project_root()
with open(str(project_root / "config.yml")) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

# read and process data
train_val_loaders, test_loaders = read_data(params)

# initialize the model
model = BertForSequenceClassification(
    pretrained_model_name=params["model"]["model_name"],
    num_classes=params["model"]["num_classes"],
)

# specify criterion for the multi-class classification task, optimizer and scheduler
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=float(params["training"]["learn_rate"]))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

# reproducibility
set_global_seed(params["general"]["seed"])
prepare_cudnn(deterministic=True)

# here we specify that we pass masks to the runner. So model's forward method will be called with
# these arguments passed to it.
runner = SupervisedRunner(input_key=("features", "attention_mask"))

# finally, training the model with Catalyst
runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=train_val_loaders,
    callbacks=[
        AccuracyCallback(num_classes=int(params["model"]["num_classes"]), input_key="logits", target_key="targets"),
        OptimizerCallback(accumulation_steps=int(params["training"]["accum_steps"]), metric_key="loss"),
        SchedulerCallback(loader_key="valid", metric_key="loss"),
        CheckpointCallback(logdir=params["training"]["log_dir"], loader_key="valid", metric_key="loss", minimize=True),
    ],
    logdir=params["training"]["log_dir"],
    num_epochs=int(params["training"]["num_epochs"]),
    verbose=True,
)

# and running inference
torch.cuda.empty_cache()

# getting validation metrics
metrics = runner.evaluate_loader(
    loader=train_val_loaders["valid"],
    callbacks=[AccuracyCallback(input_key="logits", target_key="targets")],
)
print(metrics)

# lastly, saving predicted scores for the test set
test_pred_scores = np.concatenate(
    [pred["logits"].detach().cpu().numpy() for pred in runner.predict_loader(loader=test_loaders["test"])]
)
np.savetxt(X=test_pred_scores, fname=params["data"]["path_to_test_pred_scores"])
