# SSONDO: S-SONDO: Self-Supervised Knowledge Distillation for General Audio Foundation Models

This repository contains the code for training and inference of SSONDO models - small distilled audio representation models.

## Repository Structure

- **`training_ssondo/`**: Contains all code and dependencies for training the models
  - Uses `uv` for dependency management
  - Includes scripts for extracting teacher knowledge, training, and evaluation
  
- **`inference_ssondo/`**: A pip-installable package for using the trained distilled models
  - Can be installed with `pip install -e inference_ssondo/`
  - Provides easy-to-use APIs for inference

- **`assets/`**: Contains figures and visualizations from the paper

## Setup

### Training Environment

1. Navigate to the training directory:
   ```bash
   cd training_ssondo
   ```

2. Install dependencies using `uv`:
   ```bash
   uv sync
   ```

3. Set up environment variables (see `training_ssondo/README.md` for details)

### Inference Package

Install the inference package:
```bash
pip install -e inference_ssondo/
```

## Usage

See the README files in each subdirectory for detailed usage instructions.
