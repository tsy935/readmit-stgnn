import logging
import numpy as np
import os
import pickle
import sys
import torch
import json
import time
import random
import queue
import shutil
import tqdm
import math
import dgl
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import scipy.sparse as sp
import pandas as pdƒ
from scipy.sparse import linalg
from collections import defaultdict
from datetime import datetime
from itertools import repeat
from collections import OrderedDict, defaultdict
from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.metrics import precision_recall_curve, average_precision_score


def last_relevant_pytorch(output, lengths, batch_first=False):
    lengths = lengths.cpu()

    # masks of the true seq lengths
    masks = (lengths - 1).view(-1, 1).expand(len(lengths), output.size(2))
    time_dimension = 1 if batch_first else 0
    masks = masks.unsqueeze(time_dimension)
    masks = masks.type(torch.int64)
    masks = masks.to(output.device)
    last_output = output.gather(time_dimension, masks).squeeze(time_dimension)
    last_output.to(output.device)

    return last_output


def seed_torch(seed=123):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_logger(log_dir, name):
    """Get a `logging.Logger` instance that prints to the console
    and an auxiliary file.
    Args:
        log_dir (str): Directory in which to create the log file.
        name (str): Name to identify the logs.
    Returns:
        logger (logging.Logger): Logger instance for logging events.
    """

    class StreamHandlerWithTQDM(logging.Handler):
        """Let `logging` print without breaking `tqdm` progress bars.
        See Also:
            > https://stackoverflow.com/questions/38543506
        """

        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Log everything (i.e., DEBUG level and above) to a file
    log_path = os.path.join(log_dir, "log.txt")
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Log everything except DEBUG level (i.e., INFO level and above) to console
    console_handler = StreamHandlerWithTQDM()
    console_handler.setLevel(logging.INFO)

    # Create format for the logs
    file_formatter = logging.Formatter(
        "[%(asctime)s] %(message)s", datefmt="%m.%d.%y %H:%M:%S"
    )
    file_handler.setFormatter(file_formatter)
    console_formatter = logging.Formatter(
        "[%(asctime)s] %(message)s", datefmt="%m.%d.%y %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)

    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_save_dir(base_dir, training, id_max=5000):
    """Get a unique save directory by appending the smallest positive integer
    `id < id_max` that is not already taken (i.e., no dir exists with that id).
    Args:
        base_dir (str): Base directory in which to make save directories.
        name (str): Name to identify this training run. Need not be unique.
        training (bool): Save dir. is for training (determines subdirectory).
        id_max (int): Maximum ID number before raising an exception.
    Returns:
        save_dir (str): Path to a new directory with a unique name.
    """
    for uid in range(1, id_max):
        subdir = "train" if training else "test"
        save_dir = os.path.join(base_dir, subdir, "{}-{:02d}".format(subdir, uid))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            return save_dir

    raise RuntimeError(
        "Too many save directories created with the same name. \
                       Delete old save directories or use another name."
    )


class CheckpointSaver:
    """Class to save and load model checkpoints.
    Save the best checkpoints as measured by a metric value passed into the
    `save` method. Overwrite checkpoints with better checkpoints once
    `max_checkpoints` have been saved.
    Args:
        save_dir (str): Directory to save checkpoints.
        metric_name (str): Name of metric used to determine best model.
        maximize_metric (bool): If true, best checkpoint is that which maximizes
            the metric value passed in via `save`. Otherwise, best checkpoint
            minimizes the metric.
        log (logging.Logger): Optional logger for printing information.
    """

    def __init__(self, save_dir, metric_name, maximize_metric=False, log=None):
        super(CheckpointSaver, self).__init__()

        self.save_dir = save_dir
        self.metric_name = metric_name
        self.maximize_metric = maximize_metric
        self.best_val = None
        self.ckpt_paths = queue.PriorityQueue()
        self.log = log
        self._print(
            "Saver will {}imize {}...".format(
                "max" if maximize_metric else "min", metric_name
            )
        )

    def is_best(self, metric_val):
        """Check whether `metric_val` is the best seen so far.
        Args:
            metric_val (float): Metric value to compare to prior checkpoints.
        """
        if metric_val is None:
            # No metric reported
            return False

        if self.best_val is None:
            # No checkpoint saved yet
            return True

        return (self.maximize_metric and self.best_val <= metric_val) or (
            not self.maximize_metric and self.best_val >= metric_val
        )

    def _print(self, message):
        """Print a message if logging is enabled."""
        if self.log is not None:
            self.log.info(message)

    def save(self, epoch, model, optimizer, metric_val):
        """Save model parameters to disk.
        Args:
            step (int): Total number of examples seen during training so far.
            model (torch.nn.DataParallel): Model to save.
            metric_val (float): Determines whether checkpoint is best so far.
            device (torch.device): Device where model resides.
        """
        ckpt_dict = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }

        checkpoint_path = os.path.join(self.save_dir, "last.pth.tar")
        torch.save(ckpt_dict, checkpoint_path)

        best_path = ""
        if self.is_best(metric_val):
            # Save the best model
            self.best_val = metric_val
            best_path = os.path.join(self.save_dir, "best.pth.tar")
            shutil.copy(checkpoint_path, best_path)
            self._print("New best checkpoint at epoch {}...".format(epoch))

    def save_multi(self, epoch, model_dict, optimizer_dict, metric_val):
        """Save multiple model parameters to disk.
        Args:
            step (int): Total number of examples seen during training so far.
            model (torch.nn.DataParallel): Model to save.
            metric_val (float): Determines whether checkpoint is best so far.
            device (torch.device): Device where model resides.
        """

        ckpt_dict = {
            "epoch": epoch,
            # 'model_state': model.state_dict(),
            # 'optimizer_state': optimizer.state_dict()
        }

        for model_name, model in model_dict.items():
            ckpt_dict[model_name + "_model_state"] = model.state_dict()
        for optimizer_name, optimizer in optimizer_dict.items():
            ckpt_dict[optimizer_name + "_optimizer_state"] = optimizer.state_dict()

        checkpoint_path = os.path.join(self.save_dir, "last.pth.tar")
        torch.save(ckpt_dict, checkpoint_path)

        best_path = ""
        if self.is_best(metric_val):
            # Save the best model
            self.best_val = metric_val
            best_path = os.path.join(self.save_dir, "best.pth.tar")
            shutil.copy(checkpoint_path, best_path)
            self._print("New best checkpoint at epoch {}...".format(epoch))


def load_model_checkpoint(checkpoint_file, model, optimizer=None):
    checkpoint = torch.load(checkpoint_file)
    try:
        model.load_state_dict(checkpoint["model_state"])
    except:
        model.load_state_dict(checkpoint["model_state"], strict=False)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        return model, optimizer

    return model


def count_parameters(model):
    """
    Counter total number of parameters, for Pytorch
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter:
    """Keep track of average values over time.
    Adapted from:
        > https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """Reset meter."""
        self.__init__()

    def update(self, val, num_samples=1):
        """Update meter with new value `val`, the average of `num` samples.
        Args:
            val (float): Average value to update the meter with.
            num_samples (int): Number of samples that were averaged to
                produce `val`.
        """
        self.count += num_samples
        self.sum += val * num_samples
        self.avg = self.sum / self.count


def eval_dict(
    y, y_pred, y_prob, average="binary", thresh_search=False, best_thresh=0.5
):
    """
    Args:
        y : labels, shape (num_examples, num_classes)
        y_pred: per-time-step predictions, shape (num_examples, num_classes)
        y-prob: per-time-step probabilities, shape (num_examples, num_classes)
        average: 'weighted', 'micro', 'macro' etc. to compute F1 score etc.
    Returns:
        scores_dict: Dictionary containing scores such as F1, acc etc.
    """
    if thresh_search:
        best_thresh = thresh_max_f1(y_true=y, y_prob=y_prob)
        y_pred = (y_prob >= best_thresh).astype(int)

    scores_dict = {}
    if len(np.unique(y)) == 2:  # binary case
        scores_dict["auroc"] = roc_auc_score(y_true=y, y_score=y_prob)
        scores_dict["aupr"] = average_precision_score(y, y_prob)
        tn, fp, fn, tp = confusion_matrix(y_true=y, y_pred=y_pred).ravel()
        scores_dict["specificity"] = tn / (tn + fp)

    scores_dict["acc"] = accuracy_score(y_true=y, y_pred=y_pred)
    scores_dict["F1"] = f1_score(y_true=y, y_pred=y_pred, average=average)
    scores_dict["precision"] = precision_score(
        y_true=y, y_pred=y_pred, average=average, zero_division=0
    )
    scores_dict["recall"] = recall_score(y_true=y, y_pred=y_pred, average=average)

    scores_dict["best_thresh"] = best_thresh

    return scores_dict


def thresh_max_f1(y_true, y_prob):
    """
    Find best threshold based on precision-recall curve to maximize F1-score.
    Binary calssification only
    """
    if len(np.unique(y_true)) > 2:
        raise NotImplementedError

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    thresh_filt = []
    fscore = []
    n_thresh = len(thresholds)
    for idx in range(n_thresh):
        curr_f1 = (2 * precision[idx] * recall[idx]) / (precision[idx] + recall[idx])
        if not (np.isnan(curr_f1)):
            fscore.append(curr_f1)
            thresh_filt.append(thresholds[idx])
    # locate the index of the largest f score
    ix = np.argmax(np.array(fscore))
    best_thresh = thresh_filt[ix]
    return best_thresh


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def get_config(model_name, args):
    if model_name == "stgnn":
        config = {
            "hidden_dim": args.hidden_dim,
            "num_gcn_layers": args.num_gcn_layers,
            "g_conv": args.g_conv,
            "num_gru_layers": args.num_rnn_layers,
            "rnn_hidden_dim": args.rnn_hidden_dim,
            "add_bias": True,
            "dropout": args.dropout,
            "activation_fn": args.activation_fn,
            "aggregator_type": args.aggregator_type,
            "final_pool": args.final_pool,
            "t_model": args.t_model,
        }
    elif model_name == "graphsage":
        config = {
            "hidden_dim": args.hidden_dim,
            "num_gcn_layers": args.num_gcn_layers,
            "g_conv": args.g_conv,
            "num_gru_layers": args.num_rnn_layers,
            "rnn_hidden_dim": args.rnn_hidden_dim,
            "add_bias": True,
            "dropout": args.dropout,
            "activation_fn": args.activation_fn,
            "aggregator_type": args.aggregator_type,
        }
    else:
        raise NotImplementedError

    return config
