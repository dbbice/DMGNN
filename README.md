# Overview

We use [Python](https://www.python.org/) and [Pytorch](https://www.pytorch.org/) to implement a Disentangled Multi-view Graph Neural Network for Multilingual Knowledge Graph Completion model named **DMGNN**. 

## Getting Started

### Datasets

We use [DBP-5L] and [E-PKG] datasets. The structure of DBP-5L dataset is listed as follows:


```
datasetdbp5l/:
├── entities/
│   ├── el.tsv: entity names for language 'el'
├── kg/
│   ├── el-train.tsv: the train dataset for the completion task
│   ├── el-val.tsv: the train dataset for the completion task
│   ├── el-test.tsv: the train dataset for the completion task
├── seed_train_pairs/
│   ├── el-en.tsv: alignment training seeds
├── seed_train_pairs/
│   ├── el-en.tsv: alignment test seeds
├── relation.txt: set of relations
```

# install dependencies
pip install -r requirements.txt
```

## Experiments
### Training and Testing

To reproduce our experiments, please use the following script:

```bash
# w/ SI
python train.py --data_path datasetdbp5l/ --target_language ja
# w/o SI
python train.py --data_path datasetdbp5l/ --target_language ja --no_name_info --dropout 0.1
```


