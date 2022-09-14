# Learning Granularity-Unified Representations for Text-to-Image Person Re-identification

This is the codebase for our [ACM MM 2022 paper](https://arxiv.org/abs/2207.07802).
```bash
datasets
└── cuhkpedes
    ├── captions.json
    └── imgs
        ├── cam_a
        ├── cam_b
        ├── CUHK01
        ├── CUHK03
        ├── Market
        ├── test_query
        └── train_query
└──icfgpedes
    ├── ICFG-PEDES.json
    └── ICFG_PEDES
        ├── test
        └── train

```

### Download DeiT-small weights
```bash
wget https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth
```
### Process image and text datasets
```bash
python processed_data_singledata_CUHK.py
python processed_data_singledata_ICFG.py
```


### Train
```bash
python train_mydecoder_pixelvit_txtimg_3_bert.py
```

## Citation
If you find this project useful for your research, please use the following BibTeX entry.
```
@inproceedings{shao2022learning,
  title={Learning Granularity-Unified Representations for Text-to-Image Person Re-identification},
  author={Shao, Zhiyin and Zhang, Xinyu and Fang, Meng and Lin, Zhifeng and Wang, Jian and Ding, Changxing},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  year={2022}
}
```
