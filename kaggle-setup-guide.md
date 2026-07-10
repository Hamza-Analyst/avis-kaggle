# Kaggle Setup & Execution Guide for AVIS/AVISM

This guide explains how to set up the environment, compile the custom CUDA operators, and run the training/evaluation for the AVISM model on Kaggle using a **dual T4 GPU** accelerator.

---

## ⚙️ Kaggle Notebook Settings
Before running the notebook cells, make sure you configure your Kaggle notebook draft with the following settings:
1. **Accelerator:** Select **GPU T4 x2** (Dual T4 GPUs).
2. **Internet:** Toggle **Internet on** (required to pull the GitHub repository, install packages, and download the pre-trained ImageNet backbone).
3. **Dataset:** Add your private dataset `hamza987/my-avis-dataset` (which mounts under `/kaggle/input/avigseg-usingforowncode/`).

---

## 📓 Notebook Cells (Top to Bottom)

Paste and run the following four cells in your Kaggle notebook. These are optimized to hide verbose downloading/compiling progress bars to keep your background logs clean and easy to read.

### 🔹 Cell 1: Clone Repository and Link Dataset
Clones the repository if it's a fresh session, or resets and pulls changes if the folder is already present. It also creates a symbolic link to the dataset.
```python
import os
%cd /kaggle/working

# Clone if folder doesn't exist; otherwise update it
if not os.path.exists("avis-kaggle"):
    print("Cloning repository...")
    !git clone -q https://github.com/Hamza-Analyst/avis-kaggle.git > /dev/null
    print("Cloning... Done!")
else:
    print("Updating repository...")
    %cd avis-kaggle
    !git reset -q --hard origin/main > /dev/null
    !git pull -q > /dev/null
    %cd /kaggle/working
    print("Updating... Done!")

# Ensure dataset symlink is created
!ln -sf /kaggle/input/avigseg-usingforowncode/datasets /kaggle/working/avis-kaggle/datasets
print("Dataset symlink... OK!")
```

### 🔹 Cell 2: Install Dependencies
Upgrades `setuptools` to avoid `distutils` errors in Python 3.12, installs `detectron2` from source, and installs requirements from `requirements.txt` quietly.
```python
print("Installing dependencies...")
%cd /kaggle/working/avis-kaggle

# Upgrade setuptools to avoid distutils module error in Python 3.12
!pip install -q -U setuptools > /dev/null

# Install Detectron2 from official source
!python -m pip install -q 'git+https://github.com/facebookresearch/detectron2.git' > /dev/null

# Install other requirements and compile helpers
!pip install -q -r requirements.txt > /dev/null
!pip install -q -U opencv-python ninja > /dev/null

print("Installing dependencies... Done!")
```

### 🔹 Cell 3: Compile CUDA Deformable Attention Operators
Compiles the custom CUDA operators for Deformable Attention quietly using the compute capability for Nvidia T4 GPUs (`7.5`).
```python
print("Compiling CUDA Deformable Attention operators...")
%cd /kaggle/working/avis-kaggle/mask2former/modeling/pixel_decoder/ops
!export TORCH_CUDA_ARCH_LIST="7.5" && export CUDA_HOME=/usr/local/cuda && sh make.sh > /dev/null
print("Compilation... Done!")
```

### 🔹 Cell 4: Launch Training (Background Execution)
Launches the training process using both T4 GPUs.
```python
%cd /kaggle/working/avis-kaggle
print("Launching training...")
!python train_net.py --num-gpus 2 --config-file configs/avism/Base-AVIS.yaml
```

---

## 🚀 Running the Background Training (Recommended)

Since the training run takes several hours, **do not run Cell 4 interactively** (as your browser session will time out and terminate the training).

Follow these steps to run the training in the background:
1. Paste all **4 cells** in the notebook.
2. In the top-right corner of the Kaggle notebook editor, click **`Save Version`**.
3. Choose the **`Save & Run All (Commit)`** option.
4. Click **`Save`**.
5. Once the run starts, you can monitor the progress by clicking the three dots `...` next to the running version in the bottom-left corner and selecting **`Open Logs`**.
6. You can safely close your browser window; Kaggle will run the training to completion on their servers and save all checkpoints and the final `eval_results.txt` file in the output directory.
