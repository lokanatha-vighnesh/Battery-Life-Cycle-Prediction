# Team 04 – I3: Battery Early-Life Prediction

## Project overview

This project investigates whether the cycle life of a lithium-ion battery can be predicted from measurements collected during its early charge and discharge cycles.

The project uses the Severson et al. fast-charging battery dataset. The workflow covers data extraction and preparation, classical machine-learning baselines, deep-learning models, and comparison of prediction results at different early-cycle horizons.

The main prediction target is battery cycle life. The project considers three early-life horizons:

- First 10 cycles
- First 30 cycles
- First 100 cycles

## Repository contents

```text
.
├── I3_cell_split.csv
├── I3_cycle_life_targets.csv
├── T04_I3_runlog.csv
│
├── data_processing_src/
│   ├── barcode_channel_merge.py
│   ├── clean_merged_files.py
│   ├── data_cycles_extraction.ipynb
│   ├── data_preprocessing.mlx
│   ├── merge_cycle_files.py
│   ├── train_test_split.py
│   ├── update_barcodes.py
│
├── notebooks/
│   ├── all_models_training.ipynb
│   ├── baselines.ipynb
│   ├── data_cycles_extraction.ipynb
│   └── image_preprocessing.ipynb
│
└── results/
    ├── baseline/
    │   ├── baseline_test_comparison.csv
    │   ├── elastic_net_metrics.csv
    │   └── mean_life_metrics.csv
    │
    └── deep learning/
        ├── all_deep_learning_test_metrics.csv
        └── training_curves/
            └── model training-curve PNG files
```

## Project workflow

### 1. Data processing

The scripts and notebooks in `data_processing_src/` are used for the data preparation workflow:

- Extract cycle-level information from the MATLAB battery dataset.
- Update and map battery barcodes and channel identifiers.
- Merge cycle files from different batches.
- Clean merged data.
- Create train, validation and test partitions using the supplied cell split.

`data_preprocessing.mlx` is a MATLAB Live Script and requires MATLAB to open and run.

### 2. Split and target files

- `I3_cell_split.csv` contains the cell-level partition information.
- `I3_cycle_life_targets.csv` contains the cycle-life targets.
- `T04_I3_runlog.csv` records experiment runs.

Use the supplied split and target files consistently across the 10-, 30- and 100-cycle experiments. The target should be taken from the issued target file rather than inferred from a batch-local field.

### 3. Baseline models

The baseline notebook implements the classical reference methods:

- **Mean-Life Predictor:** predicts the mean cycle-life target calculated from the training cells.
- **Elastic-Net Regression:** uses ΔQ(V) features to predict cycle life.

Baseline results are stored in `results/baseline/`.

### 4. Deep-learning models

The deep-learning notebook contains the model training workflow. The project work includes:

- 1-D CNN for early-cycle sequence/features.
- 2-D CNN for battery cycle-by-voltage representations.
- Temporal-attention model.
- No-attention ablation model.
- Prediction fusion experiments, where implemented in the training notebook.

The saved metric summary and training-curve images are stored in `results/deep learning/`.

### 5. Evaluation

The project uses the following regression metrics:

- **RMSE (Root Mean Squared Error):** measures the typical magnitude of prediction error in cycle-life units; lower is better.
- **MAPE (Mean Absolute Percentage Error):** measures average absolute percentage error; lower is better.

Check the experiment notebooks and CSV result files for the exact evaluation conditions and reported values. Results in this repository should be interpreted according to the run log and the project handout protocol.

## Environment setup

Python 3.11 is recommended for the project environment. MATLAB is needed only for the `.mlx` preprocessing script.

Create and activate a virtual environment from the repository root.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, you can run the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Launch Jupyter

```powershell
jupyter lab
```

Open the notebooks under `notebooks/` and run them in the order required by their data dependencies.

## Data availability and paths

The repository structure shown here contains the processing code, notebooks, split/target protocol files and selected result summaries. The original MATLAB dataset and large extracted or processed datasets are not included in this repository.

Before running the complete pipeline:

1. Obtain the authorized source dataset.
2. Place the source files in your local data directory.
3. Review the input and output paths in the processing scripts and notebooks.
4. Run the extraction and preprocessing stages before model training.

The exact local paths may need to be changed for your machine. Do not commit private data, credentials, large generated arrays or model checkpoints.

## Reproducibility notes

- Keep the supplied cell-level split fixed across experiment horizons.
- Fit preprocessing transformations using training data only, then apply the fitted transformations to validation and test data.
- Do not use test results to select hyperparameters or model settings.
- Record each fit in `T04_I3_runlog.csv` using the required run-log format.
- Report any test evaluation that was performed before the final checkpoint honestly; do not describe an already-scored test set as unscored.
- Record the software environment, random seed, hardware and execution details where required by the project handout.

## Team and course

- **Team:** 04 – I3
- **Course:** 22AIE304 – Deep Learning
- **Project:** Battery Early-Life Prediction

## Notes

This README describes the repository layout supplied for the checkpoint. It does not claim that every experiment has completed successfully. Refer to the notebooks, run log and result files for the actual implementation status and experimental evidence.
