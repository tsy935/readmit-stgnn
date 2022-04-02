# Multimodal spatiotemporal graph neural networks for improved prediction of 30-day all-cause hospital readmission

Siyi Tang, Amara Tariq, Jared Dunnmon, Umesh Sharma, Praneetha Elugunti, Daniel Rubin, Bhavik N. Patel, Imon Banerjee, *arXiv*, 2022

## Background
Measures to predict 30-day readmission are considered an important quality factor for hospitals as they can reduce the overall cost of care through identification of high risk patients and allow allocation of resources accordingly. In this study, we proposed a spatiotemporal graph neural network (STGNN) for prediction of 30-day all-cause hospital readmission by utilizing longitudinal chest radiographs or electronic health records (EHR) during hospitalizations. To leverage both imaging and non-imaging modalities, we further designed a multimodal (MM) fusion framework (MM-STGNN) that fused longitudinal chest radiographs and EHR.

## Conda Environment Setup
To install required packages, run the following on terminal:

    conda env create -f stgnn.yml
    conda activate stgnn

## Data
### Downloading MIMIC-IV
We use the public [MIMIC-IV v1.0](https://physionet.org/content/mimiciv/1.0/) *hosp* module and [MIMIC-CXR-JPG v2.0.0](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) in our study. Both datasets are publicly available for downloading after fulfilling certain requirements, e.g., registering on [physionet](https://physionet.org/) and completing its required training, and signing the data use agreement. For more details, see [MIMIC-IV v1.0](https://physionet.org/content/mimiciv/1.0/) and [MIMIC-CXR-JPG v2.0.0](https://physionet.org/content/mimic-cxr-jpg/2.0.0/).


### Data Preprocessing
#### Cohort Selection
To select the MIMIC cohort used in our study, run the following on terminal:
```
python ehr/get_mimic_cohort.py --raw_data_dir <mimic-iv-data-dir> --cxr_data_dir <mimic-cxr-jpg-dir> --save_dir <preproc-save-dir>
```
where `<mimic-iv-data-dir>` is the directory of the downloaded MIMIC-IV data, `<mimic-cxr-jpg-dir>` is the directory of the downloaded MIMIC-CXR-JPG data, and `<preproc-save-dir>` is the directory where the filtered cohort (.csv files) will be saved.

#### Preprocessing EHR Features
To preprocess EHR features, run the following on terminal:
```
python ehr/preprocess_ehr.py --demo_file <preproc-save-dir>/mimic_admission_demo.csv --icd_file <preproc-save-dir>/mimic_hosp_icd_subgroups.csv --lab_file <preproc-save-dir>/mimic_hosp_lab_filtered.csv --med_file <preproc-save-dir>/mimic_hosp_med_filtered.csv
```

#### Preprocessing Imaging Features
To extract imaging features from DenseNet121 pretrained using MoCo self-supervised pretraining protocol on CheXpert data, please download the pretrained model [checkpoint](https://storage.googleapis.com/moco-cxr/d1w-00001.pth.tar) provided in the MoCo-CXR [repository](https://github.com/stanfordmlgroup/MoCo-CXR).

Next, run the following on terminal:
```
python cxr/extract_cxr_features.py --feature_dir <cxr-feature-dir> --model_checkpoint_dir <pretrained-model-dir> --csv_file <preproc-save-dir>/mimic_admission_demo.csv
```
where `<cxr-feature-dir>` is the directory to save extracted imaging features, `<pretrained-model-dir>` is the directory where the pretrained DenseNet121 checkpoint is saved, and `<preproc-save-dir>` is the directory where selected MIMIC-IV cohort is saved.

#### Converting DICOM files to PNG files (Optional)
If you have chest radiographs in DICOM format that follow the [DICOM](https://www.dicomstandard.org/) standard file structure, you may use the following command to convert DICOM files to PNG files prior to extract imaging features:
```
python cxr/dicom2png.py --DICOMHome <dicom-data-dir> --OutputDirectory <png-save-dir>
```
where `<dicom-data-dir>` is the directory of folders with DICOM files and `<png-save-dir>` is the directory to save converted PNG files.


### Downloading Preprocessed Graphs
Alternatively, we provide graphs preprocessed from the public [MIMIC-IV v1.0](https://physionet.org/content/mimiciv/1.0/) *hosp* module for 3 modalities presented in our paper: imaging (chest radiographs from [MIMIC-CXR-JPG v2.0.0](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)), EHR, and fusion (imaging+EHR).

Download the preprocessed graph (best MM-STGNN) from the following links to your local directory:
* [Fusion graph](https://drive.google.com/file/d/1PTCRO4DWTm_36vQSx0r7a6MMc5mVd_vi/view?usp=sharing)


## Models
The following commands reproduce the results on MIMIC-IV in the paper.

### Fusion MM-STGNN

    python3 train.py --save_dir <save-dir> --graph_dir <fusion-graph-dir> --feature_type multimodal --model_name joint_fusion --t_model gru --ehr_encoder_name embedder --cat_emb_dim 3 --hidden_dim 128 --joint_hidden 256 --num_gcn_layers 1 --num_rnn_layers 1 --add_bias True --g_conv graphsage --aggregator_type mean --num_classes 1 --dropout 0.5 --activation_fn elu --metric_name auroc --lr 5e-3 --l2_wd 5e-4 --patience 10 --pos_weight 3 --num_epochs 100 --final_pool last
where `<save-dir>` is the directory to save model checkpoints and `<fusion-graph-dir>` is the path to the downloaded fusion graph.

### Imaging-based STGNN

    python3 train.py --save_dir <save-dir> --graph_dir <imaging-graph-dir> --feature_type imaging --model_name stgcn -t_model gru --hidden_dim 128 --num_gcn_layers 1 --num_rnn_layers 1 --final_pool last --add_bias True --g_conv graphsage --aggregator_type mean --num_classes 1 --dropout 0.5 --activation_fn elu --metric_name auroc --lr 5e-3 --l2_wd 5e-4 --pos_weight 3 --num_epochs 100
where `<save-dir>` is the directory to save model checkpoints and `<imaging-graph-dir>` is the path to the downloaded imaging-based graph.

### EHR-based STGNN

    python3 train.py --save_dir <save-dir> --graph_dir <ehr-graph-dir> --feature_type non-imaging --model_name stgcn --t_model gru --ehr_encoder_name embedder --cat_emb_dim 3 --hidden_dim 128 --num_gcn_layers 1 --num_rnn_layers 1 --add_bias True --g_conv graphsage --aggregator_type mean --num_classes 1 --dropout 0.5 --activation_fn elu --metric_name auroc --lr 5e-3 --l2_wd 5e-4 --pos_weight 3 --num_epochs 100 --final_pool last
where `<save-dir>` is the directory to save model checkpoints and `<ehr-graph-dir>` is the path to the downloaded EHR-based graph.

To use our preprocessed graphs, specify `--graph_dir <preprocessed-graph-dir>`.

To directly evaluate a trained model, specify `--do_train False --load_model_path <model-checkpoint-dir>`.

## GNNExplainer
To explain a node's prediction by MM-STGNN using [GNNExplainer](https://arxiv.org/pdf/1903.03894.pdf), run the following:
```
python gnn_explainer.py --graph_dir <preprocessed-graph-dir> --model_dir <trained-model-dir> --node_to_explain <node-id-to-explain> --modality fusion --save_dir <save-dir>
```

## Reference
If you use this codebase, or otherwise find our work valuable, please cite:
**TO DO**