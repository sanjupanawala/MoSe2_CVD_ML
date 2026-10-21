# 2DM-Ensegnet

Source code for optical-image segmentation of CVD-grown MoSe₂ and the separate exfoliated-data evaluation.

The `ensegnet` package contains model definitions, data loading and augmentation, training, ensemble prediction, evaluation, and the Han et al. and Leger et al. comparison implementations. The ensemble combines SegFormer MiT-B1, ResNet18 U-Net, and ResNet50 U-Net with scSE attention.

Training protocols are defined in `ensegnet/protocols.py`. Software dependencies are listed in `requirements.txt`.

The `weights/cvd` directory contains the nine CVD-trained 2DM-Ensegnet member checkpoints, stored with Git LFS.

## References

1. Han, B., et al. (2020). Deep-learning-enabled fast optical identification and characterization of 2D materials. *Advanced Materials*, **32**, 2000953. [doi:10.1002/adma.202000953](https://doi.org/10.1002/adma.202000953).
2. Leger, P. A., Ramesh, A., Ulloa, T., and Wu, Y. (2024). Machine learning enabled fast optical identification and characterization of 2D materials. *Scientific Reports*, **14**, 27808. [doi:10.1038/s41598-024-79386-z](https://doi.org/10.1038/s41598-024-79386-z).
3. Ronneberger, O., Fischer, P., and Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *MICCAI 2015*, 234–241. [doi:10.1007/978-3-319-24574-4_28](https://doi.org/10.1007/978-3-319-24574-4_28).
4. He, K., Zhang, X., Ren, S., and Sun, J. (2016). Deep residual learning for image recognition. *CVPR 2016*, 770–778. [doi:10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90).
5. Xie, E., et al. (2021). SegFormer: Simple and efficient design for semantic segmentation with transformers. *Advances in Neural Information Processing Systems*, **34**, 12077–12090. [Paper](https://proceedings.neurips.cc/paper_files/paper/2021/hash/64f1f27bf1b4ec22924fd0acb550c235-Abstract.html).
6. Roy, A. G., Navab, N., and Wachinger, C. (2018). Concurrent spatial and channel ‘squeeze & excitation’ in fully convolutional networks. *MICCAI 2018*, 421–429. [doi:10.1007/978-3-030-00928-1_48](https://doi.org/10.1007/978-3-030-00928-1_48).

The code uses [segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch), [Hugging Face Transformers](https://github.com/huggingface/transformers), and [PyTorch](https://github.com/pytorch/pytorch).
