# XDNet Dataset

This repository hosts the dataset that is adapted from the MVTec data (originally used for anomaly detection) for few-shot cross-domain meta-learning, as presented in the paper "XDNet: A Few-Shot Meta-Learning Approach for Cross-Domain Visual Inspection" presented in the CVPR 2023 Vision Workshop".

Please use the following link to access the re-formulated dataset used in the paper : https://drive.google.com/drive/folders/14NS0eSGo-rlko1Bs3L7vqDJ9OjNYaKIc?usp=sharing

The code used to perform the cross-domain analysis is derived from https://github.com/DustinCarrion/cd-metadl 

To perform 14 fold cross-validation analysis, please replace the "*meta_splits.txt*" file in the "*info*" folder within the "*mvtec_anomaly_detection*" dataset (downloaded using above link) with "*XDNet-Dataset/data/fold_xx/meta_splits.txt*" based on each fold.

To run the experiment, modify the "*XDNet-Dataset/XDNet/model.py*" by inserting an appropriate model, and run:
```
python -m XDNet-Dataset.run --input_data_dir=mvtec_anomaly_detection --submission_dir=XDNet --output_dir_ingestion=ingestion_output --output_dir_scoring=scoring_output --verbose=True --overwrite_previous_results=False --test_tasks_per_dataset=50
```



