# Spatiotemporal graph neural networks for improved prediction of 30-day all-cause hospital readmission

Siyi Tang, Amara Tariq, Jared Dunnmon, Umesh Sharma, Praneetha Elugunti, Daniel Rubin, Bhavik N. Patel, Imon Banerjee, *arXiv*, 2022

## Background
Measures to predict 30-day readmission are considered an important quality factor for hospitals as they can reduce the overall cost of care through identification of high risk patients and allow allocation of resources accordingly. In this study, we proposed a spatiotemporal graph neural network (STGNN) for prediction of 30-day all-cause hospital readmission by utilizing longitudinal chest radiographs or electronic health records (EHR) during hospitalizations. To leverage both imaging and non-imaging modalities, we further designed a multimodal (MM) fusion framework (MM-STGNN) that fused longitudinal chest radiographs and EHR.

## Data
We provide graphs preprocessed from the public [MIMIC-IV v1.0](https://physionet.org/content/mimiciv/1.0/) *hosp* module for 3 modalities presented in our paper: imaging (chest radiographs from [MIMIC-CXR-JPG v2.0.0](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)), EHR, and fusion (imaging+EHR).

Download the preprocessed graphs from the following links to your local directory:
* [Imaging-based graph](https://drive.google.com/file/d/1FGIzGTAJlO3TFsIUxbAh_d62wOFFaimJ/view?usp=sharing)
* [EHR-based graph](https://drive.google.com/file/d/1zMtHgRKrcaGUErrYNHrv8F5Jxd6CKqRD/view?usp=sharing)
* [Fusion graph](https://drive.google.com/file/d/1PTCRO4DWTm_36vQSx0r7a6MMc5mVd_vi/view?usp=sharing)

## Conda Environment Setup
To install required packages, run the following on terminal:

    conda env create -f stgnn.yml
    conda activate stgnn

## Models
The following commands reproduce the results on MIMIC-IV in the paper.

### Imaging-based Model

    python3 train.py --save_dir <save-dir> --graph_dir <imaging-graph-dir> --feature_type imaging --model_name stgcn -t_model gru --hidden_dim 128 --num_gcn_layers 1 --num_rnn_layers 1 --final_pool last --add_bias True --g_conv graphsage --aggregator_type mean --num_classes 1 --dropout 0.5 --activation_fn elu --metric_name auroc --lr 5e-3 --l2_wd 5e-4 --pos_weight 3 --num_epochs 100
where `<save-dir>` is the directory to save model checkpoints and `<imaging-graph-dir>` is the path to the downloaded imaging-based graph.

### EHR-based Model

    python3 train.py --save_dir <save-dir> --graph_dir <ehr-graph-dir> --feature_type non-imaging --model_name stgcn --t_model gru --ehr_encoder_name embedder --cat_emb_dim 3 --hidden_dim 128 --num_gcn_layers 1 --num_rnn_layers 1 --add_bias True --g_conv graphsage --aggregator_type mean --num_classes 1 --dropout 0.5 --activation_fn elu --metric_name auroc --lr 5e-3 --l2_wd 5e-4 --pos_weight 3 --num_epochs 100 --final_pool last
where `<save-dir>` is the directory to save model checkpoints and `<ehr-graph-dir>` is the path to the downloaded EHR-based graph.

### Fusion Model

    python3 train.py --save_dir <save-dir> --graph_dir <fusion-graph-dir> --feature_type multimodal --model_name joint_fusion --t_model gru --ehr_encoder_name embedder --cat_emb_dim 3 --hidden_dim 128 --joint_hidden 256 --num_gcn_layers 1 --num_rnn_layers 1 --add_bias True --g_conv graphsage --aggregator_type mean --num_classes 1 --dropout 0.5 --activation_fn elu --metric_name auroc --lr 5e-3 --l2_wd 5e-4 --patience 10 --pos_weight 3 --num_epochs 100 --final_pool last
where `<save-dir>` is the directory to save model checkpoints and `<fusion-graph-dir>` is the path to the downloaded fusion graph.

To directly evaluate a trained model, specify `--do_train False --load_model_path <model-checkpoint-dir>`.

## GNN Explainer
**TO DO**

## Reference
If you use this codebase, or otherwise find our work valuable, please cite:
**TO DO**