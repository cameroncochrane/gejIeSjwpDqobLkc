# CNN Layer Types — Reference

A glossary of every layer type used or experimented with across the SFM model iterations (`model_1` → `model_3_2_3`, see [project_summary.md](project_summary.md)), what each one does, and whether it made it into the final architecture.

**Final model architecture** (`model_3`, reused unchanged by `model_3_2`, `model_3_2_2`, `model_3_2_3`): `Input` → data augmentation block → 4× (`Conv2D` → `BatchNormalization` → `Conv2D` → `MaxPooling2D` → `SpatialDropout2D`) → `Conv2D` (1×1 bottleneck) → `MaxPooling2D` → `Flatten` → `Dense` → `Dropout` → `Dense` (output).

## Feature extraction

| Layer | What it does | Status in this project |
|---|---|---|
| **Conv2D** | Slides small learnable filters/kernels across the image; each filter learns to detect a spatial feature (edges, textures, shapes). Core building block of every model here. | **Used in every model, kept in final.** In `model_3`, also used with a 1×1 kernel as a "bottleneck" to reduce channel depth (128→16) while keeping the spatial grid intact. |

## Downsampling / pooling

| Layer | What it does | Status in this project |
|---|---|---|
| **MaxPooling2D** | Shrinks the spatial size of feature maps (e.g. halves H and W) by taking the max value in each small window. Reduces computation and adds a little translation invariance. | **Used in every model, kept in final.** |
| **GlobalAveragePooling2D** | Collapses an entire H×W feature map down to a single average value per channel, producing a 1D vector with no spatial information left. | **Used in `model_1` (and the early `model_2_1`/`model_2_2` lineage). Removed from `model_2_3` onward** — suspected of discarding spatial location info (e.g. *where* the hand/page-edge is), which was judged important for this problem. Replaced by an extra `Conv2D` + `Flatten`. |

## Regularization

| Layer | What it does | Status in this project |
|---|---|---|
| **Dropout** | Randomly zeroes out a proportion of individual neuron activations during training, forcing the network not to over-rely on any one unit. Used to fight overfitting. | **Used throughout**, including in the final model's classifier head (`Dropout(0.4)` after the dense layer). |
| **SpatialDropout2D** | Like `Dropout`, but drops entire feature *channels* at once rather than individual pixels — more appropriate inside convolutional blocks, where neighbouring pixels in a dropped channel are highly correlated anyway. | **Introduced in `model_3`, kept in final.** Replaced plain `Dropout` inside the convolutional blocks (final classifier head still uses plain `Dropout`). |
| *(L2 kernel regularization — not a layer, but worth noting)* | Penalizes large weights in `Conv2D`/`Dense` layers to discourage overfitting on the relatively small dataset. | Added in `model_3` (`kernel_regularizer=l2(1e-4)`) across most `Conv2D`/`Dense` layers, kept in final. |

## Normalization

| Layer | What it does | Status in this project |
|---|---|---|
| **BatchNormalization** | Normalizes activations within a batch (zero mean, unit variance, then learnable scale/shift). Stabilizes and speeds up training. | **Used in every model, kept in final.** |

## Reshaping

| Layer | What it does | Status in this project |
|---|---|---|
| **Flatten** | Converts a multi-dimensional feature map (H × W × C) into a single 1D vector so it can feed into `Dense` layers. | **Not used while `GlobalAveragePooling2D` handled this role in `model_1`. Reintroduced from `model_2_3` onward** (after `GlobalAveragePooling2D` was removed) and kept in the final model. |

## Fully connected

| Layer | What it does | Status in this project |
|---|---|---|
| **Dense** | Standard fully-connected layer — every input connects to every output neuron. Used both for a hidden "classifier head" layer and for the final output layer. | **Used in every model, kept in final.** Output layer changed shape/activation along the way: `model_1` used `Dense(2, activation="sigmoid")` with one-hot labels (a mismatch for a binary problem); from `model_2_x` onward this was corrected to `Dense(1, activation="sigmoid")` with `binary_crossentropy` loss. |

## Input

| Layer | What it does | Status in this project |
|---|---|---|
| **Input** | Explicitly declares the model's input shape. `model_1` instead passed `input_shape=` directly to the first `Conv2D`; later models switched to an explicit `Input` layer, which is cleaner and required for prepending the data-augmentation block. | **Adopted from `model_2_1`/`model_2_3` onward, kept in final.** |

## Data augmentation (training-time only image layers)

Bundled into a small `data_augmentation` sub-model, applied only during training (never at inference), and inserted right after `Input`.

| Layer | What it does | Status in this project |
|---|---|---|
| **RandomRotation** | Randomly rotates the image slightly (±~3.6° here) to simulate camera/book misalignment. | **Used from `model_2_2` onward, kept in final.** |
| **RandomTranslation** | Randomly shifts the image vertically/horizontally (±5% here) to simulate the book/hand not being perfectly centred. | **Used from `model_2_2` onward, kept in final.** |
| **RandomZoom** | Randomly zooms in/out slightly (±5% here) to simulate camera-distance variation. | **Used from `model_2_2` onward, kept in final.** |
| **RandomContrast** | Randomly varies image contrast (±10% here) to simulate different lighting/page reflectance. | **Used from `model_2_2` onward, kept in final.** |
| **GaussianNoise** | Adds small random noise to pixel values, simulating camera sensor/compression noise. | **Used from `model_2_2` onward, kept in final.** |
| **RandomBrightness** | Would randomly vary image brightness to simulate exposure/lighting changes. | **Considered (listed in the augmentation plan) but deliberately not used** — brightness augmentation was judged risky for the Sobel-enhanced images specifically, since it can artificially lift the whole black background. Not reintroduced after the switch back to grayscale either. |
| *Mild blur / random crop / perspective warp / random erasing* | Other augmentations considered in the same planning pass (simulate focus variation, prevent reliance on exact pixel locations, camera-angle differences, and over-reliance on one local feature, respectively). | **Considered only — never implemented in code.** |
