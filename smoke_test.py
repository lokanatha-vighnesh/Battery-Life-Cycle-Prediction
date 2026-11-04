"""
Team 04 – I3: Battery Early-Life Prediction
Smoke-test preflight for the checkpoint repository.

Run from the repository root:
    python smoke_test.py

This script:
1. Checks required repository files and notebook presence.
2. Checks the split and target CSVs for expected columns and basic consistency.
3. Checks that the run log has the required columns.
4. Checks availability of the main Python dependencies.
5. Runs small synthetic sanity checks for:
   - Mean-life baseline
   - Elastic-Net regression
   - 1-D CNN tensor forward/backward pass
   - 2-D CNN tensor forward/backward pass
   - Temporal attention tensor forward/backward pass
   - No-attention branch tensor forward/backward pass
   - Prediction-level fusion

IMPORTANT:
These are framework/component sanity checks using synthetic data. They do NOT
train the actual project notebook models, load the real battery dataset, score
test labels, or establish project performance. Connect the actual notebook
model/training functions for the assignment's full end-to-end smoke-test.
"""

from __future__ import annotations

import csv
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
START_TIME = time.perf_counter()
TOTAL_TIMEOUT_SECONDS = 15 * 60

REQUIRED_NOTEBOOKS = [
    Path("notebooks/all_models_training.ipynb"),
    Path("notebooks/baselines.ipynb"),
]

REQUIRED_FILES = [
    Path("I3_cell_split.csv"),
    Path("I3_cycle_life_targets.csv"),
    Path("T04_I3_runlog.csv"),
    Path("README.md"),
    Path("requirements.txt"),
]

RUNLOG_COLUMNS = [
    "run_id", "date", "commit", "component", "condition", "seed",
    "n_train", "epochs", "updates", "stop_reason", "elapsed_s",
    "cpu_s", "peak_mem_mb", "threads", "workers", "trainable_params",
    "val_metric", "val_value", "best_epoch", "test_scored",
]


def elapsed() -> float:
    return time.perf_counter() - START_TIME


def report(name: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    suffix = f" — {detail}" if detail else ""
    print(f"[{status}] {name}{suffix}")


def check_files() -> bool:
    print("\n=== 1. Repository files ===")
    ok = True
    for relative in REQUIRED_FILES + REQUIRED_NOTEBOOKS:
        exists = (ROOT / relative).is_file()
        report(str(relative), exists)
        ok = ok and exists

    # Check expected project folders.
    for folder in ("data_processing_src", "notebooks", "results"):
        exists = (ROOT / folder).is_dir()
        report(f"Directory: {folder}/", exists)
        ok = ok and exists

    return ok


def read_csv_header(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        return next(reader, [])


def check_protocol_files() -> bool:
    print("\n=== 2. Split and target protocol files ===")
    split_path = ROOT / "I3_cell_split.csv"
    target_path = ROOT / "I3_cycle_life_targets.csv"

    if not split_path.exists() or not target_path.exists():
        report("Protocol CSV availability", False, "Required CSV missing")
        return False

    split_header = read_csv_header(split_path)
    target_header = read_csv_header(target_path)

    # Accept common naming variants, while requiring the key concepts.
    split_text = " ".join(split_header).lower()
    target_text = " ".join(target_header).lower()

    split_has_partition = any(
        term in split_text for term in ("partition", "split", "set")
    )
    split_has_identity = any(
        term in split_text for term in ("sample_id", "unit_id", "cell", "barcode")
    )
    target_has_life = any(
        term in target_text for term in ("cycle_life_target", "cycle_life", "target", "label")
    )

    ok = split_has_partition and split_has_identity and target_has_life
    report(
        "Protocol CSV headers",
        ok,
        f"split columns={split_header}; target columns={target_header}",
    )

    # Count rows without interpreting/scoring any target values.
    try:
        with split_path.open("r", newline="", encoding="utf-8-sig") as f:
            split_rows = sum(1 for _ in csv.DictReader(f))
        with target_path.open("r", newline="", encoding="utf-8-sig") as f:
            target_rows = sum(1 for _ in csv.DictReader(f))
        report(
            "Protocol CSV row counts",
            split_rows > 0 and target_rows > 0,
            f"split rows={split_rows}, target rows={target_rows}",
        )
        ok = ok and split_rows > 0 and target_rows > 0
    except Exception as exc:
        report("Protocol CSV row counts", False, str(exc))
        ok = False

    return ok


def check_runlog() -> bool:
    print("\n=== 3. Run-log format ===")
    path = ROOT / "T04_I3_runlog.csv"
    if not path.exists():
        report("Run log exists", False)
        return False

    try:
        header = read_csv_header(path)
        missing = [column for column in RUNLOG_COLUMNS if column not in header]
        ok = not missing
        report(
            "Required run-log columns",
            ok,
            "All required columns found" if ok else f"Missing: {missing}",
        )
        return ok
    except Exception as exc:
        report("Run-log header", False, str(exc))
        return False


def check_notebook_components() -> bool:
    print("\n=== 4. Notebook component presence (static inspection) ===")
    try:
        import json
    except ImportError:
        report("JSON support", False)
        return False

    checks = {
        "1-D CNN": ("cnn1d", "cnn_1d", "conv1d", "conv1d"),
        "2-D CNN": ("cnn2d", "cnn_2d", "conv2d"),
        "Temporal attention": ("attention", "temporalattention", "temporal_attention"),
        "No-attention ablation": ("no_attention", "noattention", "ablation"),
        "Fusion": ("fusion", "elastic_net", "elasticnet"),
        "Elastic-Net": ("elasticnet", "elastic_net", "enet"),
        "Mean-life baseline": ("mean_life", "meanlife", "training_mean"),
    }

    notebook_text = ""
    found_any = False

    for relative in REQUIRED_NOTEBOOKS:
        path = ROOT / relative
        if not path.exists():
            continue
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            cells = notebook.get("cells", [])
            text_parts = []
            for cell in cells:
                source = cell.get("source", [])
                text_parts.append("".join(source) if isinstance(source, list) else str(source))
            notebook_text += "\n" + "\n".join(text_parts)
            report(f"Notebook readable: {relative}", True, f"{len(cells)} cells")
            found_any = True
        except Exception as exc:
            report(f"Notebook readable: {relative}", False, str(exc))

    lowered = notebook_text.lower().replace("-", "_").replace(" ", "")
    all_ok = found_any
    for component, terms in checks.items():
        present = any(term.replace("-", "_").replace(" ", "") in lowered for term in terms)
        report(f"Notebook mentions {component}", present)
        all_ok = all_ok and present

    return all_ok


def check_dependencies():
    print("\n=== 5. Python dependencies ===")
    modules = [
        ("numpy", "NumPy"),
        ("pandas", "pandas"),
        ("sklearn", "scikit-learn"),
        ("torch", "PyTorch"),
        ("matplotlib", "Matplotlib"),
        ("nbformat", "nbformat"),
    ]
    available = {}
    for module, label in modules:
        try:
            imported = __import__(module)
            version = getattr(imported, "__version__", "installed")
            available[module] = imported
            report(label, True, str(version))
        except Exception as exc:
            available[module] = None
            report(label, False, str(exc))
    return available


def synthetic_baseline_checks(np, sklearn):
    print("\n=== 6. Synthetic baseline checks ===")
    ok = True
    try:
        from sklearn.linear_model import ElasticNet
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.metrics import mean_squared_error

        # Synthetic training/validation arrays only; no project/test data.
        rng = np.random.default_rng(22)
        x_train = rng.normal(size=(24, 12))
        y_train = 800 + 40 * x_train[:, 0] - 15 * x_train[:, 1]
        x_val = rng.normal(size=(8, 12))
        y_val = 800 + 40 * x_val[:, 0] - 15 * x_val[:, 1]

        train_mean = float(np.mean(y_train))
        mean_pred = np.full(shape=len(y_val), fill_value=train_mean)
        mean_rmse = float(np.sqrt(mean_squared_error(y_val, mean_pred)))
        report("Mean-life predictor", np.isfinite(mean_rmse), f"synthetic validation RMSE={mean_rmse:.3f}")
        ok = ok and np.isfinite(mean_rmse)

        model = make_pipeline(
            StandardScaler(),
            ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=2000, random_state=22),
        )
        model.fit(x_train, np.log(y_train))
        pred = np.exp(model.predict(x_val))
        enet_rmse = float(np.sqrt(mean_squared_error(y_val, pred)))
        report("Elastic-Net fit and inference", np.isfinite(enet_rmse), f"synthetic validation RMSE={enet_rmse:.3f}")
        ok = ok and np.isfinite(enet_rmse)
    except Exception as exc:
        report("Synthetic baseline checks", False, str(exc))
        ok = False
    return ok


def synthetic_torch_checks(torch):
    print("\n=== 7. Synthetic deep-learning component checks ===")
    if torch is None:
        report("Deep-learning checks", False, "PyTorch is not installed")
        return False

    import torch.nn as nn

    torch.manual_seed(22)
    device = torch.device("cpu")
    batch, seq_len, features = 4, 10, 6
    ok = True

    class Tiny1DCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(features, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(8, 1),
            )

        def forward(self, x):
            return self.net(x)

    class Tiny2DCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(1, 4, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(4, 1),
            )

        def forward(self, x):
            return self.net(x)

    class TinyAttentionModel(nn.Module):
        def __init__(self, use_attention=True):
            super().__init__()
            self.use_attention = use_attention
            self.projection = nn.Linear(features, 8)
            self.attention = nn.Linear(8, 1)
            self.output = nn.Linear(8, 1)

        def forward(self, x):
            h = torch.relu(self.projection(x))
            if self.use_attention:
                weights = torch.softmax(self.attention(h), dim=1)
                h = torch.sum(weights * h, dim=1)
            else:
                h = h.mean(dim=1)
            return self.output(h)

    def one_update(name, model, x):
        nonlocal ok
        try:
            model.to(device)
            y = torch.rand((x.shape[0], 1), device=device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            model.train()
            optimizer.zero_grad()
            prediction = model(x)
            loss = nn.functional.mse_loss(prediction, y)
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                validation_prediction = model(x)
            passed = (
                torch.isfinite(loss).item()
                and torch.isfinite(validation_prediction).all().item()
                and validation_prediction.shape == (x.shape[0], 1)
            )
            report(name, passed, "one synthetic update and validation forward pass")
            ok = ok and passed
        except Exception as exc:
            report(name, False, str(exc))
            ok = False

    x1d = torch.randn(batch, features, seq_len)
    x2d = torch.randn(batch, 1, seq_len, 16)
    xseq = torch.randn(batch, seq_len, features)

    one_update("1-D CNN component", Tiny1DCNN(), x1d)
    one_update("2-D CNN component", Tiny2DCNN(), x2d)
    one_update("Temporal attention component", TinyAttentionModel(True), xseq)
    one_update("No-attention ablation component", TinyAttentionModel(False), xseq)

    # Prediction-level fusion check: weighted blend only, no additional fit.
    try:
        enet_pred = torch.tensor([[810.0], [790.0], [830.0], [805.0]])
        cnn_pred = torch.tensor([[800.0], [800.0], [820.0], [810.0]])
        alpha = 0.5
        fused = alpha * enet_pred + (1.0 - alpha) * cnn_pred
        passed = fused.shape == (batch, 1) and torch.isfinite(fused).all().item()
        report("Prediction fusion component", passed, "synthetic prediction blend")
        ok = ok and passed
    except Exception as exc:
        report("Prediction fusion component", False, str(exc))
        ok = False

    return ok


def main() -> int:
    print("=" * 68)
    print("TEAM 04 – I3 BATTERY EARLY-LIFE PREDICTION: SMOKE-TEST PREFLIGHT")
    print("=" * 68)
    print(f"Repository: {ROOT}")
    print("Safety: synthetic checks only; no test labels are scored.")

    results = []
    results.append(("Repository files", check_files()))
    results.append(("Protocol files", check_protocol_files()))
    results.append(("Run log", check_runlog()))
    results.append(("Notebook component inspection", check_notebook_components()))

    dependencies = check_dependencies()
    np = dependencies.get("numpy")
    if np is not None and dependencies.get("sklearn") is not None:
        results.append(("Synthetic baselines", synthetic_baseline_checks(np, dependencies["sklearn"])))
    else:
        report("Synthetic baselines", False, "NumPy/scikit-learn unavailable")
        results.append(("Synthetic baselines", False))

    torch = dependencies.get("torch")
    results.append(("Synthetic deep-learning components", synthetic_torch_checks(torch)))

    total = elapsed()
    print("\n" + "=" * 68)
    print("SUMMARY")
    print("=" * 68)
    for name, passed in results:
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
    print(f"Total elapsed: {total:.2f} seconds")
    print(f"Time limit: {TOTAL_TIMEOUT_SECONDS} seconds")

    if total > TOTAL_TIMEOUT_SECONDS:
        print("FAIL  Smoke-test preflight exceeded the 15-minute limit.")
        return 1

    passed_count = sum(1 for _, passed in results if passed)
    print(f"Checks passed: {passed_count}/{len(results)}")

    print("\nLIMITATION:")
    print(
        "This preflight uses synthetic data and small demonstration networks. "
        "It does not execute the actual project model-training pipelines. "
        "A full assignment smoke test must call the real model classes and "
        "training/inference functions from the project implementation."
    )

    return 0 if all(passed for _, passed in results) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        raise SystemExit(130)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
