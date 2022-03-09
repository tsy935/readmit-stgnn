import argparse


def str2bool(x):
    if x == "true" or x == "True":
        return True
    else:
        return False


def get_args():
    parser = argparse.ArgumentParser(
        description="Spatiotemporal GNN for readmission prediction"
    )

    # general args
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Directory to save the outputs and checkpoints.",
    )
    parser.add_argument(
        "--graph_dir", type=str, default=None, help="Dir to preprocessed graph."
    )
    parser.add_argument(
        "--load_model_path",
        type=str,
        default=None,
        help="Model checkpoint to start training/testing from.",
    )
    parser.add_argument(
        "--do_train",
        default=True,
        type=str2bool,
        help="Whether perform training.",
    )
    parser.add_argument(
        "--gpu_id",
        default=0,
        type=int,
        help="Which GPU to use? If None, use default GPU.",
    )
    parser.add_argument("--rand_seed", type=int, default=123, help="Random seed.")
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Number of patience epochs before early stopping.",
    )

    # parser.add_argument(
    #     "--edge_top_perc",
    #     default=None,
    #     type=float,
    #     help="Top percentage edges to be kept.",
    # )
    # parser.add_argument(
    #     "--use_gauss_kernel",
    #     default=False,
    #     type=str2bool,
    #     help="Whether or not to use thresholded Gaussian kernel for edges",
    # )
    # parser.add_argument(
    #     "--max_seq_len",
    #     type=int,
    #     default=3,
    #     help="Maximum sequence lengt.",
    # )
    # parser.add_argument(
    #     "--max_seq_len_img",
    #     type=int,
    #     default=3,
    #     help="Maximum sequence length for images.",
    # )
    # parser.add_argument(
    #     "--max_seq_len_ehr",
    #     type=int,
    #     default=8,
    #     help="Maximum sequence length for ehr.",
    # )
    # parser.add_argument(
    #     "--label_cutoff", type=int, default=7, help="Cutoff days for binary labels."
    # )
    # parser.add_argument(
    #     "--dist_measure",
    #     type=str,
    #     default="euclidean",
    #     choices=("cosine", "hamming", "euclidean", "correlation", "jaccard"),
    #     help="Which distance measure? cosine, distance, or correlation.",
    # )
    # parser.add_argument(
    #     "--img_by",
    #     type=str,
    #     default="cxr",
    #     choices=("day", "cxr"),
    #     help="Each time step is by one day or by one cxr.",
    # )

    # parser.add_argument(
    #     "--edge_modality",
    #     type=str,
    #     nargs="+",
    #     default=["demo", "cpt", "icd", "lab", "med"],
    #     help="Modalities used for constructing edges.",
    # )

    parser.add_argument(
        "--feature_type",
        default="imaging",
        choices=("imaging", "non-imaging", "multimodal"),
        type=str,
        help="Options: imaging, pxs, cpt etc.",
    )
    # parser.add_argument(
    #     "--ehr_types",
    #     default=["demo", "cpt", "icd", "lab", "med"],
    #     nargs="+",
    #     type=str,
    #     help="Types of EHR for node features.",
    # )
    parser.add_argument(
        "--standardize",
        type=str2bool,
        default=True,
        help="Whether to standardize input to zero mean and unit variance.",
    )

    # model args
    parser.add_argument(
        "--model_name",
        type=str,
        default="stgcn",
        choices=(
            "stgcn",
            "graphsage",
            "joint_fusion",
        ),
        help="Name of the model.",
    )

    parser.add_argument(
        "--ehr_encoder_name",
        type=str,
        default=None,
        choices=("embedder", None),
        help="Name of ehr encoder.",
    )
    parser.add_argument(
        "--cat_emb_dim",
        type=int,
        default=1,
        help="Embedding dimension for categorical variables.",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=64, help="Hidden size of GCN layers."
    )
    parser.add_argument(
        "--joint_hidden",
        nargs="+",
        type=int,
        default=[128],
        help="List of hidden dims for joint fusion model.",
    )

    parser.add_argument(
        "--num_gcn_layers", type=int, default=1, help="Number of GCN layers."
    )
    parser.add_argument(
        "--g_conv",
        type=str,
        default="graphsage",
        choices=("graphsage"),
        help="Type of GRU layers.",
    )
    parser.add_argument(
        "--num_rnn_layers", type=int, default=1, help="Number of RNN (GRU) layers."
    )
    parser.add_argument(
        "--rnn_hidden_dim", type=int, default=64, help="Hidden size of RNN layers."
    )
    parser.add_argument(
        "--add_bias",
        type=str2bool,
        default=True,
        help="Whether to add bias to GraphGRU cell.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=1,
        help="Number of output class. 1 for binary classification.",
    )
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout proba.")
    parser.add_argument(
        "--activation_fn",
        type=str,
        choices=("relu", "elu"),
        default="relu",
        help="Activation function name.",
    )
    parser.add_argument(
        "--aggregator_type",
        type=str,
        default="mean",
        choices=("mean", "gcn", "pool", "lstm"),
        help="Aggregator type. For GraphSAGE only.",
    )
    parser.add_argument(
        "--final_pool",
        type=str,
        default="last",
        choices=("last", "mean", "max", "cat"),
        help="How to pool time step results?",
    )
    parser.add_argument(
        "--t_model",
        type=str,
        default="gru",
        choices=("gru"),
        help="Which temporal model to use?",
    )

    # training args
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs.")
    parser.add_argument(
        "--eval_every", type=int, default=1, help="Evaluate on dev set every x epoch."
    )
    parser.add_argument(
        "--metric_name",
        type=str,
        default="F1",
        choices=("F1", "acc", "loss", "auroc", "aupr"),
        help="Name of dev metric to determine best checkpoint.",
    )
    parser.add_argument("--l2_wd", type=float, default=5e-4, help="L2 weight decay.")
    parser.add_argument(
        "--pos_weight",
        type=float,
        nargs="+",
        default=1,
        help="Positive class weight or list of class weights to weigh the loss function.",
    )
    parser.add_argument(
        "--thresh_search",
        type=str2bool,
        default=True,
        help="Whether or not to perform threshold search on validation set.",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=64, help="Training batch size."
    )
    parser.add_argument(
        "--test_batch_size", type=int, default=64, help="Test batch size."
    )
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers.")
    parser.add_argument(
        "--which_img",
        type=str,
        default="last",
        choices=("last", "mean", "all"),
        help="Which image to use for the patient for non-temporal models.",
    )

    args = parser.parse_args()

    # which metric to maximize
    if args.metric_name == "loss":
        # Best checkpoint is the one that minimizes loss
        args.maximize_metric = False
    elif args.metric_name in ("F1", "acc", "auroc", "aupr"):
        # Best checkpoint is the one that maximizes F1 or acc
        args.maximize_metric = True
    else:
        raise ValueError('Unrecognized metric name: "{}"'.format(args.metric_name))

    # must provide load_model_path if testing only
    if (args.load_model_path is None) and (not (args.do_train)):
        raise ValueError(
            "For prediction only, please provide trained model checkpoint in argument load_model_path."
        )

    return args
