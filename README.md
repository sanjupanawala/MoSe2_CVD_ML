# 2DM-Ensegnet

Source code for optical-image segmentation of CVD-grown MoSe₂ and the separate exfoliated-data evaluation.

The `ensegnet` package contains model definitions, data loading and augmentation, training, ensemble prediction, evaluation, and the Han et al. and Leger et al. comparison implementations. The ensemble combines SegFormer MiT-B1, ResNet18 U-Net, and ResNet50 U-Net with scSE attention.

Training protocols are defined in `ensegnet/protocols.py`. Software dependencies are listed in `requirements.txt`.

The `weights/cvd` directory contains the nine CVD-trained 2DM-Ensegnet member checkpoints, stored with Git LFS.
