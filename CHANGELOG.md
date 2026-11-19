# Changelog

## 1.1.0 - 2026-11-19

### Added

- A guided curriculum of eight sections, from pixels and a linear layer through convolutions, training dynamics, transfer learning, evaluation, and the hardware that runs the arithmetic. Each section states its objectives and ends with an exercise that names which settings to change.
- Diagrams drawn in the app: channel values, softmax, a one-number step-size sketch, a hand-built filter, the residual block, and parameter and multiply-add counts read from this ResNet-18.
- Wall-clock time per training epoch and images per second on the learner's own run. Forward multiply-adds, parameter bytes, Adam state, and a forward-output parts list are computed from the model. A CUDA allocator peak is reported only when the run used CUDA.

### Changed

- The concept panels on the main page are now the curriculum. One-line help remains on the settings.
- The version is 1.1.0.

## 1.0.0 - 2026-11-18

First teaching release of ImagePro.

### Fixed

- Novel mode trains every ResNet-18 parameter from random initialization. The previous code loaded random weights and then froze the backbone, so only the new final layer learned, on features that never moved.
- Augmentation runs when an image is loaded for a batch. Each epoch draws a new horizontal flip and a new small rotation. Previously the random transforms ran once at upload, and every epoch reused those same tensors.
- The plotted loss is the mean cross-entropy per image. Previously the chart showed the sum of per-batch means, which grew when the data was split into more batches.

### Added

- Two or more classes, with add and remove controls.
- A per-class validation holdout, and train versus validation loss and accuracy on one figure.
- A validation confusion matrix and per-class accuracy.
- Plain-language explanations for learning rate, batch size, epochs, transfer learning, augmentation, overfitting, confidence, the holdout, shuffle, and the seed.
- A seed that fixes initialization, the holdout, shuffle, and augmentation.
- A built-in set of circle, square, and triangle drawings so the app can be tried without photos.
- Parameter counts for both modes, shown before training.
- Tests, Ruff, and GitHub Actions on Python 3.11–3.13.

### Changed

- Dependencies are the ones the app imports: PyTorch, torchvision, Streamlit, Matplotlib, and Pillow.
- Number inputs and sliders replace free-text settings.
- Transfer mode loads ImageNet weights with the `weights=` argument. The deprecated `pretrained=` flag is gone.
- In transfer mode the frozen backbone's BatchNorm layers stay in evaluation mode, so their ImageNet statistics do not drift while the final layer trains.
