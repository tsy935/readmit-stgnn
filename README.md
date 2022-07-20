# Multimodal spatiotemporal graph neural networks for improved prediction of 30-day all-cause hospital readmission

Siyi Tang, Amara Tariq, Jared Dunnmon, Umesh Sharma, Praneetha Elugunti, Daniel Rubin, Bhavik N. Patel, Imon Banerjee, *arXiv*, 2022. http://arxiv.org/abs/2204.06766.

## Background
Measures to predict 30-day readmission are considered an important quality factor for hospitals as they can reduce the overall cost of care through identification of high risk patients and allow allocation of resources accordingly. In this study, we propose a multimodal spatiotemporal graph neural network (MM-STGNN) for prediction of 30-day all-cause hospital readmission by fusing longitudinal chest radiographs and electronic health records (EHR) during hospitalizations.

## Conda Environment Setup
To install required packages, run the following on terminal:

    conda env create -f stgnn.yml
    conda activate stgnn
Note that you may need to install a different version of DGL depending on the CUDA version. See [DGL documentation](https://www.dgl.ai/pages/start.html) for details.

## Data
### Downloading MIMIC-IV
We use the public [MIMIC-IV v1.0](https://physionet.org/content/mimiciv/1.0/) *hosp* module and [MIMIC-CXR-JPG v2.0.0](https://physionet.org/content/mimic-cxr-jpg/2.0.0/) in our study. Both datasets are publicly available for downloading after fulfilling certain requirements, e.g., registering on [physionet](https://physionet.org/), completing its required training, and signing the data use agreement. For more details, see [MIMIC-IV v1.0](https://physionet.org/content/mimiciv/1.0/) and [MIMIC-CXR-JPG v2.0.0](https://physionet.org/content/mimic-cxr-jpg/2.0.0/).


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
python ehr/preprocess_ehr.py --demo_file <preproc-save-dir>/mimic_admission_demo.csv --icd_file <preproc-save-dir>/mimic_hosp_icd_subgroups.csv --lab_file <preproc-save-dir>/mimic_hosp_lab_filtered.csv --med_file <preproc-save-dir>/mimic_hosp_med_filtered.csv --save_dir <ehr-feature-dir>
```
where `<ehr-feature-dir>` is where the preprocessed EHR features will be saved. 

#### Preprocessing Imaging Features
To extract imaging features from DenseNet121 pretrained using MoCo self-supervised pretraining protocol on CheXpert data, please download the pretrained model [checkpoint](https://storage.googleapis.com/moco-cxr/d1w-00001.pth.tar) provided in the MoCo-CXR [repository](https://github.com/stanfordmlgroup/MoCo-CXR).

Next, run the following on terminal:
```
python cxr/extract_cxr_features.py --feature_dir <cxr-feature-dir> --model_checkpoint_dir <pretrained-model-dir> --csv_file <preproc-save-dir>/mimic_admission_demo.csv
```
where `<cxr-feature-dir>` is the directory to save extracted imaging features, `<pretrained-model-dir>` is the directory where the pretrained DenseNet121 checkpoint is saved, and `<preproc-save-dir>` is the directory where selected MIMIC-IV cohort is saved.

#### Converting DICOM Files to PNG Files (Optional)
If you have chest radiographs in DICOM format that follow the [DICOM](https://www.dicomstandard.org/) standard file structure, you may use the following command to convert DICOM files to PNG files prior to extract imaging features:
```
python cxr/dicom2png.py --DICOMHome <dicom-data-dir> --OutputDirectory <png-save-dir>
```
where `<dicom-data-dir>` is the directory of folders with DICOM files and `<png-save-dir>` is the directory to save converted PNG files.

#### Unseen test set (Optional)
Our problem setup is based on settings of semisupervised node classification. This means that all of the train/validation/test nodes' features are available to the model during training, while only the training nodes' labels are available to the model at training time. However, in real-world scenarios, we often want to predict an unseen patient's (i.e., unseen node) readmission risk, which means that the model does not have access to the unseen node's features at training time. 

Here, we simulate this scenario by splitting the original test set into half. The following command will create two files, where the first file contains train, validation, and half of the original test patients (saved to `<preproc-save-dr>/mimic_admission_demo_half_test.csv`) and can be used to train a model, the second file contains the train, validation, and the other half of the original test patients (saved to `<preproc-save-dr>/mimic_admission_demo_unseen_test.csv`) and can be used to evaluate the model.
```
python ehr/split_test_set.py --demo_file <preproc-save-dir>/mimic_admission_demo.csv --save_dir <preproc-save-dir>
```

## Models
The following commands reproduce the results on MIMIC-IV in the paper.

### Fusion MM-STGNN

    python train.py --save_dir <save-dir> --demo_file <preproc-save-dir>/mimic_admission_demo.csv --edge_modality 'demo' --feature_type 'multimodal' --ehr_feature_file <ehr-feature-dir>/ehr_preprocessed_seq_by_day_cat_embedding.pkl \
    --edge_ehr_file <ehr-feature-dir>/ehr_preprocessed_seq_by_day_one_hot.pkl --img_feature_dir <cxr-feature-dir> --ehr_types 'demo' 'icd' 'lab' 'med' --edge_top_perc 0.01 --sim_measure 'euclidean' --use_gauss_kernel True \
    --max_seq_len_img 9 --max_seq_len_ehr 9 --hidden_dim 256 --joint_hidden 128 --num_gcn_layers 1 --num_rnn_layers 1 --add_bias True --g_conv graphsage --aggregator_type mean --num_classes 1 --dropout 0.2 --activation_fn elu \
    --metric_name auroc --lr 3e-3 --l2_wd 5e-4 --patience 10 --pos_weight 4 --num_epochs 100 --final_pool last --model_name 'joint_fusion' --t_model 'gru' --ehr_encoder_name 'embedder' --cat_emb_dim 3
where `<save-dir>` is the directory to save model checkpoints, and `<preproc-save-dir>`, `<ehr-feature-dir>` and `<cxr-feature-dir>` are preprocessed directories from previous steps.

### Imaging-based STGNN

    python train.py --save_dir <save-dir> --demo_file <preproc-save-dir>/mimic_admission_demo.csv --edge_modality 'demo' --feature_type 'imaging' --edge_ehr_file <ehr-feature-dir>/ehr_preprocessed_seq_by_day_one_hot.pkl \
    --img_feature_dir <cxr-feature-dir> --edge_top_perc 0.01 --sim_measure 'euclidean' --use_gauss_kernel True --max_seq_len_img 9 --hidden_dim 256 --num_gcn_layers 1 --num_rnn_layers 1 --add_bias True \
    --g_conv graphsage --aggregator_type mean --num_classes 1 --dropout 0.2 --activation_fn elu --metric_name auroc --lr 3e-3 --l2_wd 5e-4 --patience 10 --pos_weight 4 --num_epochs 100 \
    --final_pool last --model_name 'stgnn' --t_model 'gru' --ehr_encoder_name 'embedder' --cat_emb_dim 3

### EHR-based STGNN

    python train.py --save_dir <save-dir> --demo_file <preproc-save-dir>/mimic_admission_demo.csv --edge_modality 'demo' --feature_type 'non-imaging' --ehr_feature_file <ehr-feature-dir>/ehr_preprocessed_seq_by_day_cat_embedding.pkl \
    --edge_ehr_file <ehr-feature-dir>/ehr_preprocessed_seq_by_day_one_hot.pkl --ehr_types 'demo' 'icd' 'lab' 'med' --edge_top_perc 0.01 --sim_measure 'euclidean' --use_gauss_kernel True --max_seq_len_ehr 9 \
    --hidden_dim 256 --num_gcn_layers 1 --num_rnn_layers 1 --add_bias True --g_conv graphsage --aggregator_type mean --num_classes 1 --dropout 0.2 --activation_fn elu --metric_name auroc --lr 3e-3 \
    --l2_wd 5e-4 --patience 10 --pos_weight 4 --num_epochs 100 --final_pool last --model_name 'stgnn' --t_model 'gru' --ehr_encoder_name 'embedder' --cat_emb_dim 3

### Model Evaluation on Full Cohort
To directly evaluate a trained model, keep other args the same, and specify `--do_train False --load_model_path <model-checkpoint-file>`. For example:

    python train.py --save_dir <save-dir> --demo_file <preproc-save-dir>/mimic_admission_demo.csv --edge_modality 'demo' --feature_type 'multimodal' --ehr_feature_file <ehr-feature-dir>/ehr_preprocessed_seq_by_day_cat_embedding.pkl \
    --edge_ehr_file <ehr-feature-dir>/ehr_preprocessed_seq_by_day_one_hot.pkl --img_feature_dir <cxr-feature-dir> --ehr_types 'demo' 'icd' 'lab' 'med' --edge_top_perc 0.01 --sim_measure 'euclidean' --use_gauss_kernel True \
    --max_seq_len_img 9 --max_seq_len_ehr 9 --hidden_dim 256 --joint_hidden 128 --num_gcn_layers 1 --num_rnn_layers 1 --add_bias True --g_conv graphsage --aggregator_type mean --num_classes 1 --dropout 0.2 --activation_fn elu \
    --metric_name auroc --lr 3e-3 --l2_wd 5e-4 --patience 10 --pos_weight 4 --num_epochs 100 --final_pool last --model_name 'joint_fusion' --t_model 'gru' --ehr_encoder_name 'embedder' --cat_emb_dim 3 \
    --do_train False --load_model_path <model-checkpoint-file>

### Model Evaluation on Unseen Patients
In order to evaluate the model on unseen patients, the model should not see the unseen patients' features during training. First, train the model by specifying a different cohort csv file that does not contain the unseen patients, e.g.,  `--demo_file <preproc-save-dir>/mimic_admission_demo_half_test.csv`. Once the model training is done, evaluate the model on the unseen patients by specifying `--demo_file <preproc-save-dir>/mimic_admission_demo_unseen_test.csv --do_train False --load_model_path <model-checkpoint-file>`.

If you would like to use a trained model to provide predictions on your own data, you may create a new csv file by appending your data to the bottom of the cohort file used to train the model (e.g., `<preproc-save-dir>/mimic_admission_demo.csv`), and make sure to specify `test` in the  `splits` column. In this way, the dataloader will include this additional datapoint as a **new** node in the graph and the model predictions will include this node.

## GNNExplainer
To explain a node's prediction by MM-STGNN using [GNNExplainer](https://arxiv.org/pdf/1903.03894.pdf), run the following:
```
python gnn_explainer.py --model_dir <trained-model-dir> --node_to_explain <node-id-to-explain> --modality fusion --save_dir <save-dir>
```

## Reference
If you use this codebase, or otherwise find our work valuable, please cite:
```
@ARTICLE{Tang2022-jt,
   title         = "Multimodal spatiotemporal graph neural networks for improved
                    prediction of 30-day all-cause hospital readmission",
   author        = "Tang, Siyi and Tariq, Amara and Dunnmon, Jared and Sharma,
                    Umesh and Elugunti, Praneetha and Rubin, Daniel and Patel,
                    Bhavik N and Banerjee, Imon",
   month         =  apr,
   year          =  2022,
   archivePrefix = "arXiv",
   primaryClass  = "cs.LG",
   eprint        = "2204.06766"
 }
```
