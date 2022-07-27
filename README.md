# MTD4SemiRE
Wanli Li and Tieyun Qian: "[From Consensus to Disagreement: Multi-Teacher Distillation for Semi-Supervised Relation Extraction](https://arxiv.org/abs/2112.01048)"
## 1. Requirements
 To reproduce the reported results accurately, please install the specific version of each package.
* python 3.7.10
* torch 1.7.1
* numpy 1.19.2
* transformers 4.7.0
* apex 0.1

All data should be put into `dataset/$data_name` folder in a similar format as `dataset/sample`, with a naming convention such that (1) `train-$ratio.json` indicates that certain percentage of training data are used. (2) `raw-$ratio.json` is a part of original training data, in which we assume the labels are unknown to model.

To replicate the experiments, first prepare the required dataset as below:

- SemEval: SemEval 2010 Task 8 data (included in `dataset/semeval`)
- TACRED: The TAC Relation Extraction Dataset ([download](https://catalog.ldc.upenn.edu/LDC2018T24))
  - Put the official dataset (in JSON format) under folder `dataset/tacred` in a similar format like [here](https://github.com/yuhaozhang/tacred-relation/tree/master/dataset/tacred).

We provide our partitioned data included in dataset/semeval path for reproducing the reported results.


You can use :
```python
python train.py --labeled_ratio 0.05 --seed 41 --teacher_num 0
python train.py --labeled_ratio 0.05 --seed 42 --teacher_num 1
```

to get teacher1 and teacher2.

Next, using get_teacher_features.py to get feature.pkl:
```python
python get_teacher_features.py --labeled_ratio 0.05 --model_num 2
```

Finally, you can train a student model by:
```python
python train.py --labeled_ratio 0.05 --is_student
```
If you have any question, please let me know.

## Code Overview
The main entry for all models is in `train.py`. We provide the "self-trainng + MTD" version.

## Citation
If you find our code and datasets useful, please cite our paper.
```latex
@article{DBLP:journals/corr/abs-2112-01048,
  author    = {Wanli Li and
               Tieyun Qian},
  title     = {From Consensus to Disagreement: Multi-Teacher Distillation for Semi-Supervised
               Relation Extraction},
  journal   = {CoRR},
  volume    = {abs/2112.01048},
  year      = {2021},
  url       = {https://arxiv.org/abs/2112.01048},
  eprinttype = {arXiv},
  eprint    = {2112.01048},
  timestamp = {Tue, 07 Dec 2021 12:15:54 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2112-01048.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
