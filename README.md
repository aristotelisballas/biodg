# Welcome to BioDG!

BioDG is a PyTorch suite containing benchmark datasets and algorithms for domain generalization, 
as introduced in [Towards Domain Generalization for ECG and EEG Classification: Algorithms and Benchmarks](https://arxiv.org/pdf/2303.11338.pdf).

The available algorithms were based on [DomainBed](https://github.com/facebookresearch/DomainBed) and modified
for 1D biosignal classification.

[//]: # ( ## Current results)
[//]: # (![Result table]&#40;domainbed/results/2020_10_06_7df6f06/results.png&#41;)
[//]: # ()
[//]: # (Full results for [commit 7df6f06]&#40;https://github.com/facebookresearch/DomainBed/tree/7df6f06a6f9062284812a3f174c306218932c5e4&#41; in LaTeX format available [here]&#40;domainbed/results/2020_10_06_7df6f06/results.tex&#41;.)

## Available algorithms

The [currently available ECG algorithms](biosignals/ecg/algorithms.py)
and [currently available EEG algorithms](biosignals/eeg/algorithms.py) are:

* Baseline - Empirical Risk Minimization (ERM, [Vapnik, 1998](https://www.wiley.com/en-fr/Statistical+Learning+Theory-p-9780471030034))
* Invariant Risk Minimization (IRM, [Arjovsky et al., 2019](https://arxiv.org/abs/1907.02893))
* Maximum Mean Discrepancy (MMD, [Li et al., 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Domain_Generalization_With_CVPR_2018_paper.pdf))
* Deep CORAL (CORAL, [Sun and Saenko, 2016](https://arxiv.org/abs/1607.01719))
* Representation Self-Challenging (RSC, [Huang et al., 2020](https://arxiv.org/abs/2007.02454))

[//]: # (* Group Distributionally Robust Optimization &#40;GroupDRO, [Sagawa et al., 2020]&#40;https://arxiv.org/abs/1911.08731&#41;&#41;)
[//]: # (* Interdomain Mixup &#40;Mixup, [Yan et al., 2020]&#40;https://arxiv.org/abs/2001.00677&#41;&#41;)
[//]: # (* Marginal Transfer Learning &#40;MTL, [Blanchard et al., 2011-2020]&#40;https://arxiv.org/abs/1711.07910&#41;&#41;)
[//]: # (* Meta Learning Domain Generalization &#40;MLDG, [Li et al., 2017]&#40;https://arxiv.org/abs/1710.03463&#41;&#41;)
[//]: # (* Domain Adversarial Neural Network &#40;DANN, [Ganin et al., 2015]&#40;https://arxiv.org/abs/1505.07818&#41;&#41;)
[//]: # (* Conditional Domain Adversarial Neural Network &#40;CDANN, [Li et al., 2018]&#40;https://openaccess.thecvf.com/content_ECCV_2018/papers/Ya_Li_Deep_Domain_Generalization_ECCV_2018_paper.pdf&#41;&#41;)
[//]: # (* Style Agnostic Networks &#40;SagNet, [Nam et al., 2020]&#40;https://arxiv.org/abs/1910.11645&#41;&#41;)
[//]: # (* Adaptive Risk Minimization &#40;ARM, [Zhang et al., 2020]&#40;https://arxiv.org/abs/2007.02931&#41;&#41;, contributed by [@zhangmarvin]&#40;https://github.com/zhangmarvin&#41;)
[//]: # (* Variance Risk Extrapolation &#40;VREx, [Krueger et al., 2020]&#40;https://arxiv.org/abs/2003.00688&#41;&#41;, contributed by [@zdhNarsil]&#40;https://github.com/zdhNarsil&#41;)
[//]: # (* Representation Self-Challenging &#40;RSC, [Huang et al., 2020]&#40;https://arxiv.org/abs/2007.02454&#41;&#41;, contributed by [@SirRob1997]&#40;https://github.com/SirRob1997&#41;)
[//]: # (* Spectral Decoupling &#40;SD, [Pezeshki et al., 2020]&#40;https://arxiv.org/abs/2011.09468&#41;&#41;)
[//]: # (* Learning Explanations that are Hard to Vary &#40;AND-Mask, [Parascandolo et al., 2020]&#40;https://arxiv.org/abs/2009.00329&#41;&#41;)
[//]: # (* Out-of-Distribution Generalization with Maximal Invariant Predictor &#40;IGA, [Koyama et al., 2020]&#40;https://arxiv.org/abs/2008.01883&#41;&#41;)
[//]: # (* Gradient Matching for Domain Generalization &#40;Fish, [Shi et al., 2021]&#40;https://arxiv.org/pdf/2104.09937.pdf&#41;&#41;)
[//]: # (* Self-supervised Contrastive Regularization &#40;SelfReg, [Kim et al., 2021]&#40;https://arxiv.org/abs/2104.09841&#41;&#41;)
[//]: # (* Smoothed-AND mask &#40;SAND-mask, [Shahtalebi et al., 2021]&#40;https://arxiv.org/abs/2106.02266&#41;&#41;)
[//]: # (* Invariant Gradient Variances for Out-of-distribution Generalization &#40;Fishr, [Rame et al., 2021]&#40;https://arxiv.org/abs/2109.02934&#41;&#41;)
[//]: # (* Learning Representations that Support Robust Transfer of Predictors &#40;TRM, [Xu et al., 2021]&#40;https://arxiv.org/abs/2110.09940&#41;&#41;)
[//]: # (* Invariance Principle Meets Information Bottleneck for Out-of-Distribution Generalization &#40;IB-ERM , [Ahuja et al., 2021]&#40;https://arxiv.org/abs/2106.06607&#41;&#41;)
[//]: # (* Invariance Principle Meets Information Bottleneck for Out-of-Distribution Generalization &#40;IB-IRM, [Ahuja et al., 2021]&#40;https://arxiv.org/abs/2106.06607&#41;&#41;)
[//]: # (* Optimal Representations for Covariate Shift &#40;CAD & CondCAD, [Ruan et al., 2022]&#40;https://arxiv.org/abs/2201.00057&#41;&#41;, contributed by [@ryoungj]&#40;https://github.com/ryoungj&#41;)
[//]: # (* Quantifying and Improving Transferability in Domain Generalization &#40;Transfer, [Zhang et al., 2021]&#40;https://arxiv.org/abs/2106.03632&#41;&#41;, contributed by [@Gordon-Guojun-Zhang]&#40;https://github.com/Gordon-Guojun-Zhang&#41;)
[//]: # (* Invariant Causal Mechanisms through Distribution Matching &#40;CausIRL with CORAL or MMD, [Chevalley et al., 2022]&#40;https://arxiv.org/abs/2206.11646&#41;&#41;, contributed by [@MathieuChevalley]&#40;https://github.com/MathieuChevalley&#41;)


## Available datasets

The currently available datasets are:
### ECG
The datasets for the ECG Domain Generalization setup were taken from the [2020 PhysioNet Challenge](https://moody-challenge.physionet.org/2020/) and are the:
* China Physiological Signal Challenge 2018 ([CPSC and CPSC Extra](http://2018.icbeb.org/Challenge.html))  
* PTB and PTB-XL Diagnostic ECG Database ([PTB and PTB-XL](https://www.physionet.org/content/ptbdb/1.0.0/)) 
* St Petersburg INCART 12-lead Arrhythmia Database ([INCART](https://physionet.org/content/incartdb/1.0.0/)) 
* Georgia 12-Lead ECG Challenge Database ([G12EC](https://www.kaggle.com/datasets/bjoernjostein/georgia-12lead-ecg-challenge-database) 

To download the above datasets run the following in a terminal:
```shell
# Download all files
wget -r -N -c -np https://physionet.org/files/challenge-2020/1.0.2/

# INCART Annotations -- Extract data and rename folder to 'annotations'
wget -r -N -c -np https://physionet.org/files/incartdb/1.0.0/training/

mv physionet.org/files/challenge-2020/1.0.2/training/ .
mv physionet.org/files/challenge-2020/1.0.0/training/ annotations/
```
After downloading the datasets, please also download the following files:
* PhysioNet 2020 scored classes [csv](https://github.com/physionetchallenges/evaluation-2020/blob/master/dx_mapping_scored.csv)
* PhysioNet 2020 unscored classes [csv](https://github.com/physionetchallenges/evaluation-2020/blob/master/dx_mapping_unscored.csv)

After extracting all datasets, the ECG data directory should follow the below tree structure:
```shell
├── annotations               
├── cpsc_2018
├── cpsc_2018_extra
├── georgia
├── ptb
├── ptb-xl
├── st_petersburg_incart
├── dx_mapping_scored.csv
└── dx_mapping_unscored.csv
```

### EEG
The datasets for the EEG Domain Generalization are provided by the [BCMI](https://bcmi.sjtu.edu.cn/) laboratory of the Shanghai Jiao Tong University, 
and are the following:
* [SEED](https://bcmi.sjtu.edu.cn/home/seed/)
* [SEED-FRA](https://bcmi.sjtu.edu.cn/home/seed/seed-FRA.html)
* [SEED-GER](https://bcmi.sjtu.edu.cn/home/seed/seed-GER.html)

The datasets are available for research purposes, after applying [here](https://bcmi.sjtu.edu.cn/ApplicationForm/apply_form/).

## Quick start

### ECG
1) First set the following variables in the bioconfig.py [file](ECG.bioconfig.py):
* hostname  --> Create a block with your hostname to set data paths
* scripts_root --> Root of the code package
* _root_ecg_path --> Root path of the ecg data

2) Convert the .mat ECG signal files to the appropriate format for the PyTorch DataLoader by running:

```sh
python3 ECG/convert_to_pickles.py --hostname user --outpath 'path to directory where converted data will be stored'
```
3) Set the following variables in the ECG bioconfig.py [file](ECG.bioconfig.py):
* pickle_data_dir         --> should be same path as the above output path
* ecg_results_dir         --> experiment results path

4) Train a DG model:

```sh
python3 experiments/ecg_dg_train.py\
        --network RSC --algorithm RSC\
        --c "Flags are mentioned in the experiment file"
```

5) Train our proposed model:
```shell
python3 experiments/ecg.py --model biodg_resnet18 --epochs 30 --optim adam --batch_size 128
```

### EEG
1) After downloading the datasets, run the following script to split and convert the EEG DE features to the appropriate format for the PyTorch DataLoader
```sh
python3 EEG/split_de_features.py\
        --eeg_de_features_path 'path to 1s_de_feature folder in EEG dataset'
        --split_data_path 'output path for split data'
        --dataset 'one between china, fra or ger'
```
2) Set the following variables in the EEG config.py [file](EEG.config.py):
* pickle_data_dir        --> should be the same as the split_data_path above
* eeg_results_dir        --> experiment results path
3) Train a model:
```sh
python3 experiments/eeg_dg_train.py\
        --network ERM --algorithm ERM\
        --c "Flags are mentioned in the experiment file"
```
4) Train our proposed model:
```shell
python3 experiments/eeg.py --epochs 20 --optim adam --batch_size 128
```

## Cite Us
If you use the above code for your research please cite our paper, which as of the 22nd of June 2023 has been accepted at [IEEE TETCI](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=7433297):
```citation
@ARTICLE{10233054,
  author={Ballas, Aristotelis and Diou, Christos},
  journal={IEEE Transactions on Emerging Topics in Computational Intelligence}, 
  title={Towards Domain Generalization for ECG and EEG Classification: Algorithms and Benchmarks}, 
  year={2023},
  volume={},
  number={},
  pages={1-11},
  doi={10.1109/TETCI.2023.3306253}}
```

## License

This source code is released under the MIT license.
