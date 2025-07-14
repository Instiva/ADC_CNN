

# Defect Classification from Optical Images using CNN

This project implements a Convolutional Neural Network (CNN) to classify defects (e.g., scratch, particle, stain, no defect) from optical images captured by a defect metrology tool. The solution is built in Python using PyTorch, chosen for its flexibility in model design and strong community support for image classification tasks. The code supports grayscale or RGB images (256x256 pixels) and includes data preprocessing, model training, evaluation, and visualization.

This project was initialized around the Kaggle NEU Surface Defects Databse, which contains labeled .jpgs of six kinds of typical surface defects of the hot-rolled steel strip (300 images of each category).
  - https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database

The project uses the ./ADC_CNN/kaggle/NEU-DET/ path for train and validation sets. To replace with a new dataset, rework the path and data loaders. Please be mindful of structural and contentual differences between the original and replacement dataset.

## Project Overview

- **Objective**: Develop a CNN to achieve high accuracy (>95%) in classifying defect types from optical images, with robustness to variable conditions.
- **Dataset**: Assumes a dataset like the NEU Surface Defect Database or synthetic images, organized with labeled defect classes.
- **Framework**: PyTorch is used for its dynamic computation graphs and ease of debugging, ideal for iterative model design.
- **Key Features**:
  - Data augmentation and preprocessing (resizing, normalization, rotation, flipping).
  - K-fold cross-validation (5 folds) for robust performance.
  - Early stopping based on validation loss, accuracy, or F1-score (configurable).
  - Evaluation with accuracy, precision, recall, F1-score, and confusion matrix visualization.
  - Model saving with security (weights_only=True) to mitigate unpickling risks.

## Requirements

- **Python 3.8+**
- **Libraries**:
  - `torch` (PyTorch)
  - `torchvision`
  - `numpy`
  - `pillow` (PIL)
  - `matplotlib`
  - `seaborn`
  - `sklearn`
  - `pynvml`
  - `psutil`
  - `tqdm`
  - `flask`
  - `opencv-python`

- For GPU training, CUDA 12.1 and cuDNN (for CUDA 12.1) installed on an NVIDIA CUDA-enabled GPU.
    
- **Hardware**: GPU with ≥6GB VRAM recommended for training (CUDA support). CPU can be used for training but will be slower.

Install dependencies:
```bash
pip install torch torchvision numpy pillow matplotlib seaborn scikit-learn pynvml psutil tqdm flask
```



To use the flask dashboard, run the flask_dashboard.py script in its own powershell window and connect to localhost:5000

To use the tensorboard dashboard, run **tensorboard --logdir=runs** in the its own powershell window (with the .venv activated) and connect to localhost:6006, then run a tensorboard-enabled python script (these training scripts are tensorboard enabled)


## Dataset Preparation

- **Structure**: Place images in a `kaggle/NEU-DET` directory with subfolders:
  - `train/images/{crazing,inclusion,patches,pitted_surface,rolled-in_scale,scratches}`
  - `validation/images/{crazing,inclusion,patches,pitted_surface,rolled-in_scale,scratches}`
- **Format**: Grayscale JPG images (256x256 pixels).
- **Alternative**: If no dataset is available, generate synthetic images or download the NEU Surface Defect Database (adjust class names if needed).

## Usage Instructions

### 0. Setup Virtual Environment (Optional, but recommended)
It is advised to create a virtual environment for this project's dependencies.
To create the virtual environment, run python -m venv .venv
To activate the virtual environment, run .venv\Scripts\activate. The following instructions assume the .venv has been activated and has had the appropriate dependencies installed.

### 1. Clone the Repository
```bash
git clone https://github.com/Instiva/ADC_CNN.git
cd ADC_CNN
```

### 2. Prepare the Environment
- Ensure the datasets for training and validation are placed in the `kaggle/NEU-DET` folder relative to the script.
- Install dependencies as listed above.

### 3. Choose Early Stopping Strategy
Three scripts are provided to compare different early stopping criteria:
- `cuda_kfold_ValLoss.py`: Stops on validation loss plateau.
- `cuda_kfold_ValAcc.py`: Stops on validation accuracy plateau.
- `cuda_kfold_F1.py`: Stops on F1-score plateau (use the F1 script from prior interactions).

### 4. Run the Training
Execute each script separately to compare results. Example for loss-based:
```bash
python cuda_kfold_ValLoss.py
```

- **Output**:
  - Training logs (epoch, loss, accuracy, LR) printed to console.
  - TensorBoard logs in `runs/fold_{fold}_loss` (or `_acc`, `_f1`). These are saved into the ../runs/ folder.
  - Saved models: `best_model_fold_{fold}_(val_loss).pth`, `defect_classifier_val_loss.pth` (and similar for acc/f1).
    - Final model states are saved into ./models/ folder.
    - Partial states (i.e. for each k-fold) are saved into the ./models/training_cache/ folder.
  - Evaluation metrics and confusion matrix at the end.

### 5. Monitor Training
- Use TensorBoard to visualize metrics:
  ```bash
  tensorboard --logdir runs
  ```
  - Open http://localhost:6006 in a browser.
  - Check `val_loss`, `val_acc`, and `val_f1` (for F1 script) trends.
- Observe epoch results in Terminal:
  - ex: `Epoch 1/100, Train Loss: 13.3142, Val Loss: 9.0696, Train Acc: 0.3139, Val Acc: 0.4889, Val F1: 0.4511, LR: 0.001000`

### 6. Evaluate Results
- After each training run, the script displays:
  - Test Accuracy, Precision, Recall, F1-score.
  - Confusion matrix (heatmap and grid format).
  - Sample images with predicted vs. true labels.
- Compare across strategies using the saved `defect_classifier_{val_loss,val_acc,f1}.pth` models.

### 7. Inference on New Images
- Use the `dynamic_inference_dataloader.py` and/or `dynamic_inference_folder.py` scripts to test

  -or-
- Load the chosen model (e.g., `defect_classifier_val_loss.pth` using `torch.load()`) and modify the script to process new images:
  - Update `/targets/` to include a new image path or folder(s) containing images to classify.
  - Runs inference in the `with torch.no_grad()` block.

### Configuration Options
- **Batch Size**: Adjust `batch_size=16` in `DataLoader` based on memory.
- **Patience**: Modify `train_patience=15` and `scheduler_patience=5` in the loop, if needed.
    - `train_patience` dictates how many epochs may pass without an improvement to the target parameter before early stopping is triggered.
    - `scheduler_patience` dictates how many epochs may pass without an improvement to the target parameter before the learn rate is decreased.
- **Learning Rate**: Change `lr=0.001` in `optimizer = optim.Adam(model.parameters(), lr=0.001`, if needed.
    - Training loop will decrement the learning rate by 15% whenever `scheduler_patience` is tripped.
- **Epochs**: Adjust `for epoch in range(100)` if needed.

## Example Output
- **Console Log** (partial):
  ```
  Fold 1/5
  Epoch 1/100, Train Loss: 1.2345, Val Loss: 1.1234, Train Acc: 0.5678, Val Acc: 0.6123, LR: 0.001000
  ...
  Early stopping triggered
  Test Accuracy: 0.7501, Precision: 0.7402, Recall: 0.7456, F1-Score: 0.7429
  ```
- **Confusion Matrix**: Visualized and printed as a grid.

## Comparison of Early Stopping Strategies
- **Validation Loss**: Extends training if validation loss improves (decreases), rolling back to lowest loss state.
- **Validation Accuracy**: Extends training if validation accuracy improves (increases), rolling back to highest accuracy state.
- **F1 Score**: Balances precision/recall. Extends training if F1 Score improves (increases), rolling back to highest F1 state.
- Analyze results post-training to determine the best fit for your accuracy goal.
    - Without k-fold cross-validation, validation accuracy was topping out around ~75-78%.
    - With k-fold cross-validation, validation accuracies have surpassed ~98-99%.
        - All scripts in the initial project are k-fold implementations. Feel free to explore additional techniques!

## Troubleshooting
- **Dataset Not Found**: Ensure `kaggle/NEU-DET/` exists and contains images. Ideally, these images will be organized into subfolders holding their labels.
- **CUDA Errors**: Verify GPU setup or switch to `device = torch.device('cpu')`.
- **Early Stopping Too Soon**: Increase `train_patience` or adjust `factor` in `ReduceLROnPlateau`.
    - `factor` will be the fraction of the learning rate that is kept, i.e. a `factor=0.85` indicates a 15% reduction in learning rate, when triggered.

## Future Improvements
- Add hyperparameter tuning (e.g., grid search).
- Support RGB images by adjusting `Conv2d` input channels.
- Implement hybrid early stopping (e.g., both loss and F1).
- Enhance dynamic inference scripts with visualization or subfolder support.

## License

MIT License

Copyright (c) [2025] [Instiva]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
