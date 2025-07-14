To activate the virtual environment, run .venv\Scripts\activate. The following instructions assume the .venv has been activated and is 

To use the flask dashboard, run the flask_dashboard.py script in its own powershell window and connect to localhost:5000

To use the tensorboard dashboard, run tensorboard --logdir=runs in the its own powershell window (with the .venv activated) and connect to localhost:6006, then run the tensorboard-enabled python scrpits (naming convention: %_tensorboard or %_tb)


--

This project currently uses the Kaggle NEU Surface Defects Databse, where six kinds of typical surface defects of the hot-rolled steel strip have been collected and labeled.
- Currently uses ~/kaggle/NEU-DET/ path for train and validation sets. To replace with a new dataset, rework the path and data loaders


--

# Defect Classification from Optical Images using CNN

This project implements a Convolutional Neural Network (CNN) to classify defects (e.g., scratch, particle, stain, no defect) from optical images captured by a defect metrology tool. The solution is built in Python using PyTorch, chosen for its flexibility in model design and strong community support for image classification tasks. The code supports grayscale or RGB images (256x256 pixels) and includes data preprocessing, model training, evaluation, and visualization.

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
  - `json`
- **Hardware**: GPU recommended for training (CUDA support).

Install dependencies:
```bash
pip install torch torchvision numpy pillow matplotlib seaborn scikit-learn
```

## Dataset Preparation

- **Structure**: Place images in a `kaggle/NEU-DET` directory with subfolders:
  - `train/images/{crazing,inclusion,patches,pitted_surface,rolled-in_scale,scratches}`
  - `validation/images/{crazing,inclusion,patches,pitted_surface,rolled-in_scale,scratches}`
- **Format**: Grayscale JPG images (256x256 pixels).
- **Alternative**: If no dataset is available, generate synthetic images or download the NEU Surface Defect Database (adjust class names if needed).

## Usage Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd <your-repo-directory>
```

### 2. Prepare the Environment
- Ensure the dataset is placed in the `kaggle/NEU-DET` folder relative to the script.
- Install dependencies as listed above.

### 3. Choose Early Stopping Strategy
Three scripts are provided to compare different early stopping criteria:
- `defect_classifier_loss.py`: Stops on validation loss plateau.
- `defect_classifier_acc.py`: Stops on validation accuracy plateau.
- `defect_classifier_f1.py`: Stops on F1-score plateau (use the F1 script from prior interactions).

### 4. Run the Training
Execute each script separately to compare results. Example for loss-based:
```bash
python defect_classifier_loss.py
```

- **Output**:
  - Training logs (epoch, loss, accuracy, LR) printed to console.
  - TensorBoard logs in `runs/fold_{fold}_loss` (or `_acc`, `_f1`).
  - Saved models: `best_model_fold_{fold}_loss.pth`, `defect_classifier_loss.pth` (and similar for acc/f1).
  - Evaluation metrics and confusion matrix at the end.

### 5. Monitor Training
- Use TensorBoard to visualize metrics:
  ```bash
  tensorboard --logdir runs
  ```
  - Open http://localhost:6006 in a browser.
  - Check `val_loss`, `val_acc`, and `val_f1` (for F1 script) trends.

### 6. Evaluate Results
- After each run, the script displays:
  - Test Accuracy, Precision, Recall, F1-score.
  - Confusion matrix (heatmap and grid format).
  - Sample images with predicted vs. true labels.
- Compare across strategies using the saved `defect_classifier_{loss,acc,f1}.pth` models.

### 7. Inference on New Images
- Load the best model (e.g., `defect_classifier_loss.pth`) and modify the script to process new images:
  - Update `NEUDataset` to include a new image path.
  - Run inference in the `with torch.no_grad()` block.

### Configuration Options
- **Batch Size**: Adjust `batch_size=16` in `DataLoader` based on memory.
- **Patience**: Modify `train_patience=15` and `scheduler_patience=5` in the loop.
- **Learning Rate**: Change `lr=0.001` in `optim.Adam`.
- **Epochs**: Increase `range(100)` if needed.

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
- **Validation Loss**: May stop early (e.g., 25-30 epochs) if loss plateaus, rolling back to best loss state.
- **Validation Accuracy**: Extends training if accuracy improves, rolling back to best accuracy state.
- **F1 Score**: Balances precision/recall, potentially training longest, rolling back to best F1 state.
- Analyze results post-training to determine the best fit for your >75% accuracy goal.

## Troubleshooting
- **Dataset Not Found**: Ensure `kaggle/NEU-DET` exists and contains images.
- **CUDA Errors**: Verify GPU setup or switch to `device = torch.device('cpu')`.
- **Early Stopping Too Soon**: Increase `train_patience` or adjust `factor` in `ReduceLROnPlateau`.

## Future Improvements
- Add hyperparameter tuning (e.g., grid search).
- Support RGB images by adjusting `Conv2d` input channels.
- Implement hybrid early stopping (e.g., both loss and F1).

## License
[Specify your license, e.g., MIT] - Optional, add if applicable.
