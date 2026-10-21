# 2DM-Ensegnet: optical segmentation of CVD-grown MoSe₂

Research code for the final nine-member ensemble, Han and Leger architecture comparisons, a neutral RGB histogram reference and the separate exfoliated-data evaluation. The CVD dataset was acquired and annotated by the authors. Raman measurements of representative regions guided optical annotation; labels are not pixelwise Raman maps.

The older standalone U-Net workflow has been removed from this branch and remains recoverable through Git history. This cleanup preserves the recorded results; it does not retrain or replace saved models.

## Files

```
ensegnet/     models, data loading, training, evaluation, metrics and plotting
configs/      the three recorded training protocols
records/      portable split identities, validation selection and verified scores
tests/        deterministic checks without private images or downloaded weights
```

## Method and comparison conditions

2DM-Ensegnet averages softmax probabilities from three independently trained fits of each member, giving nine models:

- MiT-B1 SegFormer (`segformer_b1`), with ImageNet input normalization.
- ResNet18 U-Net (`student_control`), using RGB / 255.
- ResNet50 U-Net with scSE decoder attention (`scse`), using RGB / 255.

Composition was selected using CVD validation scores and retained for the separate source-data experiment. Training uses Adam and cross entropy plus macro soft Dice with equal coefficients. Each checkpoint maximizes pooled three-class validation mIoU; exact ties retain the earlier epoch. Inference averages probabilities equally, then takes argmax, without test-time augmentation.

**Han et al.** is a local VGG16/index-unpool encoder–decoder reconstruction; **Leger et al.** is a local grayscale ResNet34 U-Net implementation. Final author-trained checkpoints were not available to this study. CVD comparisons use implementations retrained on our acquired CVD images under the common CVD protocol. Their reported results are means of three separately scored fits, not scores of a probability-averaged three-fit ensemble.

On released exfoliated data, the Leger reproduction uses grayscale 0–255, 20 epochs and cross entropy. The separately trained ensemble uses RGB / 255, 50 epochs and CE + Dice. These recorded protocols are **not a matched-budget architecture-only comparison**. Neither source experiment uses CVD images or CVD-trained initialization. Published metrics computed differently must not be relabeled as locally measured argmax mIoU.

## Install

The recorded environment used Python 3.9.24, PyTorch 2.8.0 + CUDA 12.8 and torchvision 0.23.0. Keep segmentation-models-pytorch 0.2.1 and timm 0.4.12 for historical checkpoint compatibility.

```bash
python -m venv .venv
# Activate .venv using your shell's activation command.
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m ensegnet --help
```

CPU inference is available with `--device cpu --full-precision`; numerical scores can differ slightly from recorded CUDA precision. Inference never downloads weights. Training initializes generic ImageNet encoders; `--asset` accepts a local MiT-B1 directory or, for Han, a local VGG16 weight file.

## Data and weights

The manuscript links [CVD data](https://doi.org/10.5281/zenodo.18921676) and [segmentation resources](https://doi.org/10.5281/zenodo.18646700). These are existing deposits; this update does not replace their assets. Check their inventory before choosing weights: an older single-model checkpoint cannot substitute for nine final member checkpoints.

Released exfoliated images come from the data release associated with [Leger et al. (2024)](https://doi.org/10.1038/s41598-024-79386-z), [source repository](https://github.com/aramesh10/MLOC-Machine-Learning-Enabled-Fast-Optical-Identification-and-Characterization-of-2D-Materials), commit `91d584bb82507e07a8ea65a870916aaadec95192`. Preserve the provided split and the release's attribution/licensing terms. No images, third-party data, manuscripts or trained weights are bundled here.

```
CVD_DATA/{train,val,test}/images/<name>.<extension>
CVD_DATA/{train,val,test}/masks/<name>_mask.png
SOURCE_DATA/{train,val,test}/images/<name>.jpg
SOURCE_DATA/{train,val,test}/labels/<name>.png
```

CVD: 97 / 12 / 12 tiles, 512 × 512, mask values 0 / 20 / 40. Source: 1024 / 256 / 320 images, 100 × 100, mask values 1 / 2 / 3. The released top-left 96 × 96 crop is retained without resizing. Internal class indices 0 / 1 / 2 mean substrate / monolayer / multilayer. Test images keep their stored orientation. Portable manifests in `records/` identify the exact partitions by filenames and SHA256 hashes.

## Train and evaluate

Run from the repository root. Choose new output directories; existing results are protected against overwriting.

```bash
# Nine CVD fits; training reads train/val only.
python -m ensegnet train --data CVD_DATA --config configs/cvd.json --output runs/cvd
python -m ensegnet evaluate --data CVD_DATA --weights-dir runs/cvd --output outputs/cvd

# Comparator fits on our acquired CVD images.
python -m ensegnet train --data CVD_DATA --config configs/cvd.json --model han --output runs/han
python -m ensegnet train --data CVD_DATA --config configs/cvd.json --model leger --output runs/leger

# Evaluate each fit separately; repeat for the other two checkpoint files.
python -m ensegnet evaluate --data CVD_DATA --models leger --checkpoints runs/leger/leger__20261010.pth --output outputs/leger_fit1
python -m ensegnet summarize outputs/leger_fit1/metrics.json outputs/leger_fit2/metrics.json outputs/leger_fit3/metrics.json --output outputs/leger_mean.json

# Neutral training-fitted, validation-selected RGB reference.
python -m ensegnet fit-rgb --data CVD_DATA --output weights/rgb.npz
python -m ensegnet evaluate-rgb --data CVD_DATA --checkpoint weights/rgb.npz --output outputs/rgb

# Separate source-data experiments; fresh generic encoder initialization.
python -m ensegnet train --data SOURCE_DATA --config configs/exfoliated.json --output runs/source_ensemble
python -m ensegnet evaluate --data SOURCE_DATA --domain exfoliated --weights-dir runs/source_ensemble --batch-size 16 --reduction source --output outputs/source_ensemble
python -m ensegnet train --data SOURCE_DATA --config configs/leger_source.json --model leger --output runs/source_leger
python -m ensegnet evaluate --data SOURCE_DATA --domain exfoliated --models leger --checkpoints runs/source_leger/leger__20261008.pth --source-gray --full-precision --batch-size 4 --output outputs/source_leger
```

`--weights-dir` expects `<architecture>__<seed>.pth` for the three member IDs and three fit identifiers in the configs. Alternatively, list explicit `--checkpoints` and corresponding `--models`, one architecture ID per checkpoint. Historical `state` wrappers and direct Leger source state dictionaries load strictly. Evaluation saves class-index PNGs, pooled/per-image scores, checkpoint hashes and the evaluated manifest.

The frozen source-to-CVD diagnostic uses the Leger source checkpoint without CVD training or tuning:

```bash
python -m ensegnet transfer --data CVD_DATA --checkpoint SOURCE_LEGER_CHECKPOINT --normalized-images ARCHIVED_NORMALIZED_PNGS --output outputs/transfer
```

Primary inference uses exact archived upstream MATLAB Lab-normalized PNGs and nonoverlapping 96-pixel patches with bottom/right reflect padding. Supply those PNGs explicitly; the code does not substitute another Lab conversion. Omitting `--normalized-images` runs the raw-patch sensitivity. Adding `--full-field` with normalized images runs the fully convolutional sensitivity.

## Figures and metrics

```bash
python -m ensegnet plot --metrics outputs/han_mean.json outputs/leger_mean.json outputs/cvd/metrics.json --labels "Han et al." "Leger et al." "2DM-Ensegnet" --output outputs/comparison
```

This exports editable SVG text, PDF and 600 dpi PNG point-estimate graphs. These regenerate numerical comparisons, not experimental characterization artwork.

Confusion matrices have reference classes in rows and predictions in columns. IoU = TP / (TP + FP + FN); mIoU is the unweighted mean of three class IoUs after pooling pixels across evaluated tiles. A class absent from both maps contributes zero, matching the recorded implementation. Pixel accuracy is the fraction of correct pixels and is a different metric.

Coverage is class fraction × 100. Signed error is predicted minus annotated coverage. Coverage mean absolute error (MAE) is averaged across tiles, in **percentage points**. Boundary F1 uses 3 × 3 morphological gradients with a two-pixel square tolerance. A boundary absent from both maps is omitted from the boundary mean; a boundary on only one side scores zero.

The CVD split is random at tile level: source fields can occur in multiple partitions, and the test set had been inspected in earlier work. The released source split also contains related derived crops. Scores characterize the preserved partitions; tiles are not independent growth-run replicates. `records/results.json` stores primary scores and checkpoint identities. The validation record documents composition selection without bundling the exploratory workflow.

Cleanup verification recomputed the archived maps' scores exactly and checked strict checkpoint loading and identical member outputs against the original implementations. Fresh CUDA ensemble inference differed from the archived mIoUs by less than 0.000004 (both still round to 0.888 for CVD and 0.801 for the source data). Mixed-precision kernels can change a few borderline pixel assignments. The recorded study results remain unchanged. The Leger source reproduction and frozen source-to-CVD diagnostic reproduced their archived confusion matrices exactly.

## References

- Han et al., *Deep-Learning-Enabled Fast Optical Identification and Characterization of 2D Materials*, Advanced Materials (2020), [10.1002/adma.202000953](https://doi.org/10.1002/adma.202000953).
- Leger et al., Scientific Reports (2024), [10.1038/s41598-024-79386-z](https://doi.org/10.1038/s41598-024-79386-z).
- Implementations use [segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch), [Transformers](https://github.com/huggingface/transformers) and [PyTorch](https://pytorch.org/). Their licenses apply to their code/assets. No framework or author-released training scripts are vendored here.
