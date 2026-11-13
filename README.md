# COVID-19 Detection from Chest X-ray Images

An educational image-classification project using TensorFlow/Keras and MobileNetV2 transfer learning. It classifies chest X-ray images into the class folders provided by your dataset (for example, `COVID`, `NORMAL`, and `PNEUMONIA`).

> **Medical limitation:** This is a learning/research project, not a medical device. Do not use it to diagnose, screen, treat, or make decisions about anyone. Clinical interpretation requires qualified healthcare professionals.

## Project structure

```text
COVID-19-Detection-Deep-Learning/
├── README.md
├── requirements.txt
├── .gitignore
├── app.py
├── data/README.md
├── models/README.md
├── reports/README.md
└── src/
    ├── __init__.py
    ├── data_loader.py
    ├── preprocessing.py
    ├── train.py
    ├── evaluate.py
    └── predict.py
```

## 1. Requirements and installation

Use Python 3.10 or 3.11. From the project root:

### Windows PowerShell
```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If activation is blocked, use Command Prompt and run `.venv\Scripts\activate.bat`.

### macOS/Linux
```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Dataset preparation

Arrange a dataset like this. Folder names are the class labels:

```text
data/chest_xray/
├── train/
│   ├── COVID/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── COVID/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── COVID/
    ├── NORMAL/
    └── PNEUMONIA/
```

Use the same class folder names in all three splits. Supported image formats include JPG, JPEG, PNG, and BMP.

**Dataset attribution:** No dataset is bundled or assumed. Obtain data from a source whose current terms permit your intended use. Record the exact dataset title, authors, version, source URL, access date, required citation, and permitted-use conditions in `data/README.md`. Do not upload restricted or patient-identifiable images to GitHub.

## 3. Train

Run from the repository root:

```bash
python -m src.train --data-dir data/chest_xray --epochs 10 --batch-size 32
```

For all options: `python -m src.train --help`.

The best checkpoint is saved as `models/covid_xray_model.keras`; class labels are saved in `models/class_names.json`; training plots are saved in `reports/`. The first run may download ImageNet weights. To avoid downloading them, use `--weights none`.

## 4. Evaluate

```bash
python -m src.evaluate --data-dir data/chest_xray/test
```

This prints test loss, accuracy, per-class precision/recall/F1, and a confusion matrix. Keep the test set separate from training and model tuning.

## 5. Predict one image

```bash
python -m src.predict --image path/to/chest_xray.jpg
```

Or:

```bash
python app.py --image path/to/chest_xray.jpg
```

The displayed score is a model output, not a clinically calibrated probability or diagnosis.

## 6. Responsible use and reproducibility

- Split data at patient level where possible to reduce data leakage.
- Check class balance, duplicate images, label quality, and dataset provenance.
- Document dataset version, split strategy, random seed, preprocessing, and training settings.
- Accuracy alone is not enough; inspect per-class metrics and the confusion matrix.
- External validation on data from other institutions/devices is needed to assess generalization.
- Do not claim performance numbers until you have produced them through a reproducible evaluation.

## Troubleshooting

- **No images found:** check directory names and image files.
- **Class mismatch:** ensure train/val/test contain identical class folder names.
- **Out of memory:** reduce `--batch-size` to 8 or 16.
- **TensorFlow install errors:** use Python 3.10/3.11 and consult TensorFlow's current installation guidance for your OS/hardware.
