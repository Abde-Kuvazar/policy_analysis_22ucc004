# src/train_rl.py
"""
Overwrite-safe final train_rl.py — Behavior Cloning (DiscreteBC) focused version.

Usage:
    python src/train_rl.py --data-path data/processed/rl_dataset.csv --n-epochs 50 --batch-size 256

Features:
- Uses Behavior Cloning (DiscreteBC) as the primary offline algorithm (fallback to DiscreteDQN).
- Handles large datasets safely using chunked loading and temporary storage on D: drive.
- Saves preprocessor, checkpoints, best model, and a JSON summary.
"""

from __future__ import annotations
import argparse
import json
import time
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# gym -> gymnasium fallback
try:
    import gymnasium as gym  # type: ignore
except ImportError:
    import gym  # type: ignore

import d3rlpy
from d3rlpy.dataset import MDPDataset

# -----------------------
# Utilities
# -----------------------
def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def parse_args():
    parser = argparse.ArgumentParser(description="Train offline RL (Behavior Cloning) on dataset")
    parser.add_argument("--data-path", type=str, default="data/processed/rl_dataset.csv")
    parser.add_argument("--preprocessor-path", type=str, default="models/preprocessor.joblib")
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--n-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--subset-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()

def clean_numeric_columns(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in numeric_cols:
        if df[col].dtype == object:
            df[col] = (
                df[col].astype(str)
                      .str.replace("%", "", regex=False)
                      .replace("", "0")
                      .astype(float)
            )
    return df

def build_preprocessor_from_chunks(chunks: list[pd.DataFrame], numeric_cols: list[str], categorical_cols: list[str], sample_per_chunk: int = 50_000):
    sampled_chunks = [
        chunk.sample(sample_per_chunk, random_state=42) if len(chunk) > sample_per_chunk else chunk
        for chunk in chunks
    ]
    df_fit = pd.concat(sampled_chunks, ignore_index=True)

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    preprocessor = ColumnTransformer([("num", num_pipe, numeric_cols), ("cat", cat_pipe, categorical_cols)], remainder="drop")
    preprocessor.fit(df_fit[numeric_cols + categorical_cols])
    return preprocessor

def align_chunk_columns(chunk: pd.DataFrame, expected_cols: list[str], numeric_cols: list[str], categorical_cols: list[str]) -> pd.DataFrame:
    for c in expected_cols:
        if c not in chunk.columns:
            chunk[c] = 0 if c in numeric_cols else "Unknown"
    return chunk[expected_cols]

def make_mdpdataset_from_arrays(obs: np.ndarray, acts: np.ndarray, rews: np.ndarray, terms: np.ndarray | None = None) -> MDPDataset:
    if terms is None:
        terms = np.ones(len(obs), dtype=bool)
    return MDPDataset(observations=obs.astype(np.float32), actions=acts.astype(np.int32), rewards=rews.astype(np.float32), terminals=terms.astype(bool))

def evaluate_policy(policy_fn, df: pd.DataFrame, batch_size: int = 2048) -> dict:
    obs_list = df["__obs"].tolist()
    N = len(obs_list)
    realized = []
    arr = np.array(obs_list, dtype=np.float32)
    for i in range(0, N, batch_size):
        sub = arr[i:i + batch_size]
        acts = policy_fn(sub)
        for j, a in enumerate(acts):
            realized.append(float(df["reward"].iloc[i + j]) if int(a) == 1 else 0.0)
    arr_r = np.array(realized, dtype=np.float32)
    return {"avg_reward": float(arr_r.mean()), "total_reward": float(arr_r.sum()), "n": int(len(arr_r))}

def instantiate_bc_algo():
    try:
        from d3rlpy.algos import DiscreteBC
        algo = DiscreteBC(use_gpu=d3rlpy.gpu.get_device_count() > 0)
        print("[algo] Instantiated DiscreteBC")
        return algo
    except Exception as e:
        print("[algo] DiscreteBC failed, fallback to DiscreteDQN:", e)
        from d3rlpy.algos import DiscreteDQN
        algo = DiscreteDQN(use_gpu=d3rlpy.gpu.get_device_count() > 0)
        return algo

# -----------------------
# Main
# -----------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    rl_path = Path(args.data_path)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    if not rl_path.exists():
        raise FileNotFoundError(f"RL dataset not found at {rl_path}")

    preprocessor, numeric_cols, categorical_cols = None, [], []
    preproc_path = Path(args.preprocessor_path)
    if preproc_path.exists():
        try:
            preproc_obj = joblib.load(preproc_path)
            if isinstance(preproc_obj, dict):
                preprocessor = preproc_obj.get("preprocessor")
                numeric_cols = preproc_obj.get("numeric_cols", [])
                categorical_cols = preproc_obj.get("categorical_cols", [])
                print("Loaded preprocessor from dict:", preproc_path)
            else:
                preprocessor = preproc_obj
                print("Loaded ColumnTransformer preprocessor from:", preproc_path)
        except Exception as e:
            print("Warning: failed to load preprocessor:", e)

    # -----------------------
    # Load dataset (temp CSV for large files)
    # -----------------------
    temp_path = Path("D:/kolors-temp/rl_dataset_temp.csv")
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    chunksize = 100_000
    obs_list = []

    if temp_path.exists():
        print(f"Loading temp CSV: {temp_path}")
        reader = pd.read_csv(temp_path, chunksize=chunksize, low_memory=False)
        first_chunk = next(reader)

        if preprocessor is None:
            candidate = [c for c in first_chunk.columns if c not in {"action", "reward", "__obs"}]
            numeric_cols = [c for c in candidate if pd.api.types.is_numeric_dtype(first_chunk[c])]
            categorical_cols = [c for c in candidate if not pd.api.types.is_numeric_dtype(first_chunk[c])]
            expected_cols = numeric_cols + categorical_cols
            fit_chunks = [first_chunk] + [chunk for _, chunk in zip(range(4), reader)]
            preprocessor = build_preprocessor_from_chunks(fit_chunks, numeric_cols, categorical_cols)
            joblib.dump({"preprocessor": preprocessor, "numeric_cols": numeric_cols, "categorical_cols": categorical_cols}, model_dir / "preprocessor.joblib")
            print("Saved new preprocessor")
        else:
            expected_cols = numeric_cols + categorical_cols

        # Transform all chunks
        first_chunk_aligned = align_chunk_columns(first_chunk, expected_cols, numeric_cols, categorical_cols)
        first_chunk_aligned = clean_numeric_columns(first_chunk_aligned, numeric_cols)
        X_transformed = preprocessor.transform(first_chunk_aligned)
        obs_list.extend(X_transformed.tolist())

        for i, chunk in enumerate(reader, start=2):
            chunk_aligned = align_chunk_columns(chunk, expected_cols, numeric_cols, categorical_cols)
            chunk_aligned = clean_numeric_columns(chunk_aligned, numeric_cols)
            X_chunk = preprocessor.transform(chunk_aligned)
            obs_list.extend(X_chunk.tolist())
            if args.verbose:
                print(f"Processed chunk {i}")

        df = pd.read_csv(temp_path, usecols=["action", "reward"])
        df["__obs"] = obs_list

    else:
        print("Temp CSV not found, loading original dataset...")
        reader = pd.read_csv(rl_path, chunksize=chunksize, low_memory=False)
        chunks, rows_loaded, max_rows = [], 0, args.subset_rows
        for i, chunk in enumerate(reader):
            if max_rows and rows_loaded + len(chunk) > max_rows:
                chunk = chunk.iloc[: max_rows - rows_loaded]
            for c in ["action", "reward"]:
                if c not in chunk.columns:
                    raise ValueError(f"Required column '{c}' missing in dataset")
            chunk["action"] = chunk["action"].astype(np.int32)
            chunk["reward"] = chunk["reward"].astype(np.float32)
            chunks.append(chunk)
            rows_loaded += len(chunk)
            print(f"Loaded chunk {i+1}, {rows_loaded} rows")
            if max_rows and rows_loaded >= max_rows:
                break
        df = pd.concat(chunks, ignore_index=True)
        df.to_csv(temp_path, index=False)
        print(f"Saved temp dataset at {temp_path}, shape {df.shape}")

        if preprocessor is None:
            candidate = [c for c in df.columns if c not in {"action", "reward", "__obs"}]
            numeric_cols = [c for c in candidate if pd.api.types.is_numeric_dtype(df[c])]
            categorical_cols = [c for c in candidate if not pd.api.types.is_numeric_dtype(df[c])]
            expected_cols = numeric_cols + categorical_cols
            preprocessor = build_preprocessor_from_chunks([df.iloc[:500_000]], numeric_cols, categorical_cols)
            joblib.dump({"preprocessor": preprocessor, "numeric_cols": numeric_cols, "categorical_cols": categorical_cols}, model_dir / "preprocessor.joblib")
        else:
            expected_cols = numeric_cols + categorical_cols

        obs_list = []
        for chunk in chunks:
            chunk_aligned = align_chunk_columns(chunk, expected_cols, numeric_cols, categorical_cols)
            X_chunk = preprocessor.transform(chunk_aligned)
            obs_list.extend(X_chunk.tolist())
        df["__obs"] = obs_list

    # -----------------------
    # Train/test split
    # -----------------------
    train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=args.seed, shuffle=True)
    obs_train = np.array(train_df["__obs"].tolist(), dtype=np.float32)
    acts_train = train_df["action"].to_numpy(dtype=np.int32)
    rew_train = train_df["reward"].to_numpy(dtype=np.float32)
    obs_test = np.array(test_df["__obs"].tolist(), dtype=np.float32)
    acts_test = test_df["action"].to_numpy(dtype=np.int32)
    rew_test = test_df["reward"].to_numpy(dtype=np.float32)

    train_dataset = make_mdpdataset_from_arrays(obs_train, acts_train, rew_train)
    test_dataset = make_mdpdataset_from_arrays(obs_test, acts_test, rew_test)

    print(f"Train transitions: {train_dataset.transition_count}, Test transitions: {test_dataset.transition_count}")

    # -----------------------
    # Instantiate and train
    # -----------------------
    algo = instantiate_bc_algo()
    model_base = model_dir / f"bc_agent_{int(time.time())}"
    best_avg = -np.inf
    history = {"epoch": [], "test_avg_reward": []}

    epochs_remaining, epoch_counter = args.n_epochs, 0
    step = args.save_interval if args.save_interval > 0 else args.n_epochs

    while epochs_remaining > 0:
        n = min(step, epochs_remaining)
        print(f"Training epochs {epoch_counter + 1}..{epoch_counter + n} ...")
        algo.fit(train_dataset, n_epochs=n, batch_size=args.batch_size, verbose=args.verbose)
        epoch_counter += n
        epochs_remaining -= n

        def policy_fn(states: np.ndarray):
            try:
                return np.array(algo.predict(states)).astype(int).reshape(-1)
            except Exception:
                return np.array([int(algo.predict([s])[0]) for s in states], dtype=int)

        eval_metrics = evaluate_policy(policy_fn, test_df.reset_index(drop=True))
        avg_reward = eval_metrics["avg_reward"]
        total_reward = eval_metrics["total_reward"]
        history["epoch"].append(epoch_counter)
        history["test_avg_reward"].append(avg_reward)
        print(f"[eval] Epoch {epoch_counter}: test avg_reward={avg_reward:.6f}, total_reward={total_reward:.2f}")

        # Checkpoints
        ckpt_path = model_base.with_suffix(f".epoch{epoch_counter}.zip")
        try: algo.save(str(ckpt_path)); print("Saved checkpoint:", ckpt_path)
        except Exception as e: print("Warning: failed to save checkpoint:", e)

        if avg_reward > best_avg:
            best_avg = avg_reward
            best_path = model_dir / "bc_agent_best.zip"
            try: algo.save(str(best_path)); print("Saved new best model:", best_path)
            except Exception as e: print("Warning: failed to save best model:", e)

        with open(model_dir / "training_history.json", "w") as fh:
            json.dump(history, fh)
        print("Saved training history")

    # Final save
    final_path = model_dir / "bc_agent_final.zip"
    try: algo.save(str(final_path)); print("Saved final model:", final_path)
    except Exception as e: print("Warning: failed to save final model:", e)

    summary = {
        "best_test_avg_reward": float(best_avg),
        "n_train_transitions": int(train_dataset.transition_count),
        "n_test_transitions": int(test_dataset.transition_count),
        "preprocessor_path": str(model_dir / "preprocessor.joblib"),
        "final_model": str(final_path)
    }
    with open(model_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print("Training completed. Summary written to:", model_dir / "summary.json")


if __name__ == "__main__":
    main()
