# Semantic Segmentation using PyTorch Lightning
Repository for implementation and training of semantic segmentation models using PyTorch Lightning. This repo was contributed as a [full example](https://github.com/Lightning-AI/lightning/blob/master/examples/pl_domain_templates/semantic_segmentation.py) in the official [PyTorch Lightning](https://github.com/Lightning-AI/lightning) repository. However there have been further changes (majorly w.r.t. coding practices) to that example since my initial pull requests were merged. Thus, this repo is no longer being maintained.

## Dataset Used
The [KITTI semantic segmentation dataset](http://www.cvlibs.net/datasets/kitti/eval_semseg.php?benchmark=semantics2015) is used in this. So, it needs to be downloaded from above link. It is used for it's small size (200 semantically annotated images and 200 test images). Data preprocessing and loading is as per this dataset. This dataset claims to be same in format to [Cityscapes](https://www.cityscapes-dataset.com/) (a much larger dataset of road scenes), so same code can be used with some modifications (not done in this yet, due to memory limitations).

## Models Implemented
- FCN ResNet50/101 (available through [torchvision.models.segmentation](https://pytorch.org/docs/stable/torchvision/models.html#semantic-segmentation))
- DeepLabv3 ResNet50/101 (available through [torchvision.models.segmentation](https://pytorch.org/docs/stable/torchvision/models.html#semantic-segmentation))
- [U-net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)
- [ENet](https://arxiv.org/abs/1606.02147)
