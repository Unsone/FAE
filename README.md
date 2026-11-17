# FAE

FeAture Explorer (FAE), a radiomics (or medical analysis) tool that helps radiologists extract features, preprocess feature matrix, develop machine learning models (Binary Classification & Survival Analysis) with one-click, and evaluate models qualitatively  and quantitatively. This project was inspired on the [Radiomics](http://www.radiomics.io/), and provides a GUI with convenient process. FAE was initially developed by East China Normal University and Siemens Healthineers Ltd. 

If FAE could help in your project, We appreciate that you could cite this work:

> Y. Song, J. Zhang, Y. Zhang, Y. Hou, X. Yan, Y. Wang, M. Zhou, Y. Yao, G. Yang. FeAture Explorer (FAE): A tool for developing and comparing radiomics models. PLoS One. 2020. DOI: https://doi.org/10.1371/journal.pone.0237587
        
        
        
        
        
        

Welcome any issues and PR. 

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

## Release

The Windows64 version and the Ubuntut 20.04 release could be found [Google Drive](https://drive.google.com/open?id=1htts7YsfaxKtN1NeDcNU4iksXfjr_XyK) or [SourceForce](https://sourceforge.net/projects/feature-explorer/). A short [tutorial video](https://www.bilibili.com/video/BV1yt4y1S79S/) with Chinese version may help.

## Installation

FAE ships two dependency sets:

| Use case | Install |
| --- | --- |
| Headless modelling (Linux server, notebooks, CI) | `requirements.txt` |
| Workstation with the Qt GUI | `requirements-gui.txt` |

### Headless (recommended for servers)

```bash
git clone https://github.com/Unsone/FAE.git
cd FAE
conda create -n fae python=3.11 -y
conda activate fae
pip install -r requirements.txt
```

The headless set does not include PySide6, SimpleITK or pyradiomics, so it installs on any Python 3.11 environment without a display.

### Workstation with GUI

```bash
pip install -r requirements-gui.txt
```

PySide6 6.11 wheels require glibc >= 2.34 (Ubuntu 22.04 or newer). pyradiomics is excluded from both lists: version 3.0.1 has no CPython 3.11 wheel and has to be built manually; only the feature-extraction pages need it.

## Running (headless CLI)

```bash
python -m fae info
python -m fae bc --data Example/numeric_feature.csv --output fae_runs/example --seed 2026
```

A more explicit pipeline:

```bash
python -m fae bc \
  --data Example/numeric_feature.csv \
  --test-ratio 0.3 --seed 2026 \
  --balance none --normalizer zscore \
  --reducer pcc --selector anova \
  --feature-numbers 5 10 20 \
  --classifier svm --cv 5 \
  --output fae_runs/example
```

Useful flags: `--test-data reference.csv` to split by a reference case list instead of randomly, `--no-hyper-param` to skip the nested grid search, and `--help` to list every choice. The console prints CV and test metrics; `pipeline_info.csv`, `*_results.csv`, predictions, models and `run_config.json` are written to the output folder.

## Running (GUI)

```bash
python -m pip install -r requirements-gui.txt
python MainFrameCall.py
```

## Generating UI Files

The `.py` modules corresponding to each `.ui` file must be generated locally (they are git-ignored, so a fresh clone has none of them). Use the `pyside6-uic` tool that ships with PySide6:

```bash
pyside6-uic HomeUI/HomePage.ui                    -o HomeUI/HomePage.py
pyside6-uic Feature/GUI/FeatureExtraction.ui      -o Feature/GUI/FeatureExtraction.py
pyside6-uic Feature/GUI/FeatureMerge.ui           -o Feature/GUI/FeatureMerge.py
pyside6-uic Feature/GUI/IccEstimation.ui          -o Feature/GUI/IccEstimation.py
pyside6-uic BC/GUI/Prepare.ui                     -o BC/GUI/Prepare.py
pyside6-uic BC/GUI/Process.ui                     -o BC/GUI/Process.py
pyside6-uic BC/GUI/Visualization.ui               -o BC/GUI/Visualization.py
pyside6-uic BC/GUI/ModelPrediction.ui             -o BC/GUI/ModelPrediction.py
pyside6-uic SA/GUI/Prepare.ui                     -o SA/GUI/Prepare.py
pyside6-uic SA/GUI/Process.ui                     -o SA/GUI/Process.py
pyside6-uic SA/GUI/Visualization.ui               -o SA/GUI/Visualization.py
```

> After regenerating, verify that the generated files import from `PySide6` (e.g. `from PySide6 import QtCore, QtGui, QtWidgets`) before committing.

## Project Structure

```
FAE/
├── MainFrameCall.py          # Application entry point
├── MainFrameCall_opt.py      # Entry point (compatibility mode, disables native dialogs)
├── HomeUI/                   # Main window and navigation hub
│   ├── HomePageForm.py       # Central controller; owns all submodule instances
│   ├── HomePage.ui / .py     # Qt Designer layout and generated code
│   └── VersionConstant.py    # Version number (MAJOR.MINOR.PATCH)
├── Feature/                  # Radiomics feature engineering
│   ├── FileMatcher.py        # Case / image / ROI file matching
│   ├── SeriesMatcher.py      # Series string matching
│   ├── RadiomicsParamsConfig.py
│   └── GUI/                  # Feature extraction, merge, and ICC estimation forms
├── BC/                       # Binary Classification pipeline
│   ├── DataContainer/        # Feature matrix, labels, and case IDs
│   ├── FeatureAnalysis/      # Balance → Normalize → Reduce → Select → Classify → CV
│   ├── Visualization/        # ROC curves, AUC plots, SHAP bar/beeswarm
│   ├── Description/          # PDF report generation
│   ├── HyperParameters/      # Classifier and selector hyperparameter configs (JSON)
│   └── GUI/                  # Preprocessing, model exploration, and visualization forms
├── SA/                       # Survival Analysis pipeline (mirrors BC structure)
│   ├── PipelineManager.py    # SA pipeline orchestrator
│   ├── DataContainer.py
│   └── GUI/                  # Model exploration and visualization forms
├── Plugin/                   # Plugin launcher
│   └── PluginManager.py      # Discovers plugins via config.json, runs via os.system()
├── Utility/                  # Shared utilities (logging, path helpers, radiomics utils)
├── fae/                      # Headless command line interface (python -m fae)
├── scripts/                  # Maintenance scripts (requirement export)
├── src/                      # Image assets (PNG icons for the UI)
├── Example/                  # Sample data
├── pyproject.toml            # Dependencies (uv)
├── uv.lock                   # Locked dependency graph
├── requirements.txt          # Server / headless pip list (generated)
└── requirements-gui.txt      # GUI pip list (generated)
```

## Dependencies

Dependencies are managed with [uv](https://docs.astral.sh/uv/) and locked in `pyproject.toml` / `uv.lock`. The pip requirement files are generated, never edited by hand:

```bash
uv add <package>                        # runtime dependency
uv add --dev <package>                  # test / tooling dependency
uv sync                                 # local headless environment
uv sync --extra gui                     # local environment with the Qt GUI
python scripts/export_requirements.py   # regenerate requirements*.txt
```

Commit `pyproject.toml`, `uv.lock`, `requirements.txt` and `requirements-gui.txt` together.

## License 
This project is licensed under the Apache 2.0 License

## Contributor and Acknowledge List
- Developer:
  - [**Yang Song**](https://github.com/salan668)
  - [**Jing Zhang**](https://github.com/zhangjingcode)
  - [**Guang Yang**](https://github.com/yg88)
  - **Chengxiu Zhang**
  - [**Xue-xiang Cao**](mailto:xuer_cao@hotmail.com)
- Tester: 
  - Xu Yan
  - [**Tian-jing Zhang**](mailto:tianjingz@nvidia.com)
  - Yi-lai Pei
  - [**jmtaysom**](https://github.com/jmtaysom)
  - Zhiyong Zhang
  - Yu-dong Zhang
  - [**Wei Guo**](mailto:guowei_fy@fjmu.edu.cn)
 
