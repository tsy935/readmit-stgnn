import numpy as np
import os
import pickle
import torch
import json
from argparse import Namespace
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import math
import utils
import dgl
from dgl.data.utils import load_graphs
from args import get_args
from collections import OrderedDict, defaultdict
from json import dumps
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.model import GraphRNN
from model.fusion import JointFusionModel
from dotted_dict import DottedDict
from constants import CATEGORICAL_DIMS, CATEGORICAL_IDXS


def evaluate(
    args,
    model,
    graph,
    features,
    labels,
    nid,
    loss_fn,
    best_thresh=0.5,
    save_file=None,
    thresh_search=False,
    img_features=None,
    ehr_features=None,
):
    model.eval()
    with torch.no_grad():
        if "fusion" in args.model_name:
            logits = model(graph, img_features, ehr_features)
        elif args.model_name != "stgcn":
            assert len(graph) == 1
            features_avg = features[:, -1, :]
            logits, _ = model(graph[0], features_avg)
        else:
            logits, _ = model(graph, features)
        logits = logits[nid]

        if logits.shape[-1] == 1:
            logits = logits.view(-1)  # (batch_size,)

        labels = labels[nid]
        loss = loss_fn(logits, labels)

        logits = logits.view(-1)  # (batch_size,)
        probs = torch.sigmoid(logits).cpu().numpy()  # (batch_size, )
        preds = (probs >= best_thresh).astype(int)  # (batch_size, )

        eval_results = utils.eval_dict(
            y=labels.data.cpu().numpy(),
            y_pred=preds,
            y_prob=probs,
            average="binary",
            thresh_search=thresh_search,
            best_thresh=best_thresh,
        )
        eval_results["loss"] = loss.item()

    if save_file is not None:
        with open(save_file, "wb") as pf:
            pickle.dump(
                {
                    "probs": probs,
                    "labels": labels.cpu().numpy(),
                    "preds": preds,
                    "node_indices": nid,
                },
                pf,
            )

    return eval_results


def main(args):

    args.cuda = torch.cuda.is_available()

    if args.cuda:
        torch.cuda.set_device(args.gpu_id)
        device = "cuda:{}".format(args.gpu_id)
    else:
        device = "cpu"

    # set random seed
    utils.seed_torch(seed=args.rand_seed)

    # get save directories
    args.save_dir = utils.get_save_dir(
        args.save_dir, training=True if args.do_train else False
    )

    # save args
    args_file = os.path.join(args.save_dir, "args.json")
    with open(args_file, "w") as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)

    logger = utils.get_logger(args.save_dir, "train")
    logger.info("Args: {}".format(dumps(vars(args), indent=4, sort_keys=True)))

    # load graph
    logger.info("Loading dataset...")
    g = load_graphs(args.graph_dir)
    g = g[0][0]

    if args.feature_type != "multimodal":
        features = g.ndata[
            "feat"
        ]  # features for each graph are the same, including temporal info
        img_features = None
        ehr_features = None
        cat_idxs = CATEGORICAL_IDXS
        cat_dims = CATEGORICAL_DIMS
    else:
        img_features = g.ndata["img_feat"]
        ehr_features = g.ndata["ehr_feat"]
        features = None
        cat_idxs = CATEGORICAL_IDXS
        cat_dims = CATEGORICAL_DIMS
    labels = g.ndata["label"]  # labels are the same
    train_mask = g.ndata["train_mask"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]

    # ensure self-edges
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
    n_edges = g.number_of_edges()
    n_nodes = g.number_of_nodes()
    logger.info(
        """----Graph Stats------
            # Nodes %d
            # Undirected edges %d
            # Average degree %d """
        % (
            n_nodes,
            int(n_edges / 2),
            g.in_degrees().float().mean().item(),
        )
    )

    train_nid = torch.nonzero(train_mask).squeeze().to(device)
    val_nid = torch.nonzero(val_mask).squeeze().to(device)
    test_nid = torch.nonzero(test_mask).squeeze().to(device)

    train_labels = labels[train_nid]
    val_labels = labels[val_nid]
    test_labels = labels[test_nid]

    logger.info(
        "#Train samples: {}; positive percentage :{:.2f}".format(
            train_mask.int().sum().item(),
            (train_labels == 1).sum().item() / len(train_labels) * 100,
        )
    )
    logger.info(
        "#Val samples: {}; positive percentage :{:.2f}".format(
            val_mask.int().sum().item(),
            (val_labels == 1).sum().item() / len(val_labels) * 100,
        )
    )
    logger.info(
        "#Test samples: {}; positive percentage :{:.2f}".format(
            test_mask.int().sum().item(),
            (test_labels == 1).sum().item() / len(test_labels) * 100,
        )
    )

    if args.cuda:
        if args.feature_type != "multimodal":
            features = features.to(device)
        else:
            img_features = img_features.to(device)
            ehr_features = ehr_features.to(device)
        labels = labels.to(device)
        train_mask = train_mask.to(device)
        val_mask = val_mask.to(device)
        test_mask = test_mask.to(device)
        g = g.int().to(device)

    if args.model_name == "stgcn":
        in_dim = features.shape[-1]
        print("Input dim:", in_dim)
        config = utils.get_config(args.model_name, args)
        model = GraphRNN(
            in_dim=in_dim,
            n_classes=args.num_classes,
            device=device,
            is_classifier=True,
            ehr_encoder_name="embedder" if args.feature_type != "imaging" else None,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=args.cat_emb_dim,
            **config
        )

    elif args.model_name == "joint_fusion":
        img_config = utils.get_config("stgcn", args)
        ehr_config = utils.get_config("stgcn", args)
        img_in_dim = img_features.shape[-1]
        ehr_in_dim = ehr_features.shape[-1]
        model = JointFusionModel(
            img_in_dim=img_in_dim,
            ehr_in_dim=ehr_in_dim,
            img_config=img_config,
            ehr_config=ehr_config,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            ehr_encoder_name=args.ehr_encoder_name,
            cat_emb_dim=args.cat_emb_dim,
            joint_hidden=args.joint_hidden,
            num_classes=args.num_classes,
            dropout=args.dropout,
            device=device,
        )
    else:
        in_dim = features.shape[-1]
        print("Input dim:", in_dim)
        config = utils.get_config(args.model_name, args)
        model = GConvLayers(
            in_dim=in_dim,
            num_classes=args.num_classes,
            is_classifier=True,
            device=device,
            **config
        )

    model.to(device)

    # define optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.l2_wd
    )

    # load model checkpoint
    if args.load_model_path is not None:
        model, optimizer = utils.load_model_checkpoint(
            args.load_model_path, model, optimizer
        )

    # count params
    params = utils.count_parameters(model)
    logger.info("Trainable parameters: {}".format(params))

    # loss func
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor(args.pos_weight)).to(
        device
    )

    # checkpoint saver
    saver = utils.CheckpointSaver(
        save_dir=args.save_dir,
        metric_name=args.metric_name,
        maximize_metric=args.maximize_metric,
        log=logger,
    )

    # scheduler
    logger.info("Using cosine annealing scheduler...")
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    # average meter for validation loss
    nll_meter = utils.AverageMeter()

    if args.do_train:
        # Train
        logger.info("Training...")
        model.train()
        epoch = 0
        prev_val_loss = 1e10
        patience_count = 0
        early_stop = False

        while (epoch != args.num_epochs) and (not early_stop):

            epoch += 1
            logger.info("Starting epoch {}...".format(epoch))

            # forward
            # if no temporal dim
            if "fusion" in args.model_name:
                logits = model(g, img_features, ehr_features)
            elif args.model_name != "stgcn":
                assert len(g) == 1
                features_avg = features[:, -1, :]
                logits, _ = model(g, features_avg)
            else:
                logits, _ = model(g, features)

            if logits.shape[-1] == 1:
                logits = logits.view(-1)
            loss = loss_fn(logits[train_nid], labels[train_nid])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # evaluate on val set
            if epoch % args.eval_every == 0:
                logger.info("Evaluating at epoch {}...".format(epoch))
                eval_results = evaluate(
                    args=args,
                    model=model,
                    graph=g,
                    features=features,
                    labels=labels,
                    nid=val_nid,
                    loss_fn=loss_fn,
                    img_features=img_features,
                    ehr_features=ehr_features,
                )
                model.train()
                saver.save(epoch, model, optimizer, eval_results[args.metric_name])
                # accumulate patience for early stopping
                if eval_results["loss"] < prev_val_loss:
                    patience_count = 0
                else:
                    patience_count += 1
                prev_val_loss = eval_results["loss"]

                # Early stop
                if patience_count == args.patience:
                    early_stop = True

                # Log to console
                results_str = ", ".join(
                    "{}: {:.4f}".format(k, v) for k, v in eval_results.items()
                )
                logger.info("VAL - {}".format(results_str))

            # step lr scheduler
            scheduler.step()

        logger.info("Training DONE.")
        best_path = os.path.join(args.save_dir, "best.pth.tar")
        model = utils.load_model_checkpoint(best_path, model)
        model.to(device)

    # evaluate
    val_results = evaluate(
        args=args,
        model=model,
        graph=g,
        features=features,
        labels=labels,
        nid=val_nid,
        loss_fn=loss_fn,
        save_file=os.path.join(args.save_dir, "val_predictions.pkl"),
        thresh_search=args.thresh_search,
        img_features=img_features,
        ehr_features=ehr_features,
    )
    val_results_str = ", ".join(
        "{}: {:.4f}".format(k, v) for k, v in val_results.items()
    )
    logger.info("VAL - {}".format(val_results_str))

    # eval on test set
    test_results = evaluate(
        args=args,
        model=model,
        graph=g,
        features=features,
        labels=labels,
        nid=test_nid,
        loss_fn=loss_fn,
        save_file=os.path.join(args.save_dir, "test_predictions.pkl"),
        best_thresh=val_results["best_thresh"],
        img_features=img_features,
        ehr_features=ehr_features,
    )
    test_results_str = ", ".join(
        "{}: {:.4f}".format(k, v) for k, v in test_results.items()
    )
    logger.info("TEST - {}".format(test_results_str))

    logger.info("Results saved to {}".format(args.save_dir))

    return val_results[args.metric_name]


if __name__ == "__main__":
    main(get_args())
