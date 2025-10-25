# RFOD for CS5344 Project
## Overview
1. 确认data文件夹下有完整的train, valid, test的csv。git LFS有点麻烦我就不传这些数据了
2. data_process.py: 数据预处理
3. rfod.py: 训练模型
4. infer.py： 得到提交kaggle csv
5. PDF是所用到的方法的论文，可以大致看看


## Quick Start
训练，本次参数最佳模型在model文件夹下
```console
python rfod.py
```

推理，得到的结果在result文件夹下
```console
python infer.py
```