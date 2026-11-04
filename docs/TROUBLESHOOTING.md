# Troubleshooting

## "Model weights required" on the home screen
No weights file exists at `model.weights` (default `models/brick_yolo11n.pt`). Place a YOLO11 detection model trained on bricks there, or update the path in `config/settings.yaml`.

## "Could not load '<file>'"
The file is not a valid Ultralytics checkpoint (corrupt, partial download, or another framework's format). Re-export the model with Ultralytics.

## "... is a 'segment' / 'classify' model"
BrickFlow needs an object-detection model. Export or train a `detect` task model.

## Compute device shows CPU although an NVIDIA GPU is installed
The card's reason line explains why:
- *Installed PyTorch build has no CUDA support* – the CPU wheel is installed. Install a CUDA wheel from pytorch.org.
- *CUDA present but unusable* – the GPU or driver is too old for the installed PyTorch build. Update the NVIDIA driver; very old GPUs may not be supported by current PyTorch CUDA builds.

CPU processing works but is slower.

## Configuration error at startup
The message names the invalid setting (for example `'model.confidence' must be between 0.01 and 1.0`). Correct it in `config/settings.yaml` and reload the page.

## `py -3.12` fails with "Unable to create process"
A stale Python 3.12 registration remains after uninstall. Use `py -3.13` or reinstall 3.12.
