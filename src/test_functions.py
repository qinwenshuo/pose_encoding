import pandas as pd

def df_descriptive(df):
    print("=== DataFrame Description ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nFirst few rows:")
    print(df.head())
    print("\nData types:")
    print(df.dtypes)
    print("\nUnique values:")
    for col in df.select_dtypes(include=['object']).columns:
        print(f"  {col}: {df[col].unique()}")
    if 'score' in df.columns:
        print("\nScore statistics:")
        print(df['score'].describe())
    print()

import psutil
import os

def print_memory_usage(note=""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 2)  # in MB
    print(f"[MEMORY] {note}: {mem:.2f} MB")



import pickle
import os
import numpy as np
import pandas as pd

def inspect_pkl(path, max_items=10, show_data=False):
    """
    Inspect contents of a pickle (.pkl) file safely.

    Parameters
    ----------
    path : str
        Path to the pickle file.
    max_items : int, optional
        Maximum number of top-level items or elements to preview.
    show_data : bool, optional
        If True, print sample data for small objects (arrays, lists, etc.).

    Returns
    -------
    obj : any
        The loaded Python object (for further inspection).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File not found: {path}")

    print(f"--- Inspecting pickle file: {path} ---")
    with open(path, "rb") as f:
        obj = pickle.load(f)

    print(f"Type: {type(obj)}")

    # ----- Dictionary -----
    if isinstance(obj, dict):
        print(f"Dict keys ({len(obj)}): {list(obj.keys())[:max_items]}")
        for k in list(obj.keys())[:max_items]:
            v = obj[k]
            _print_summary(k, v, show_data)

    # ----- List / tuple -----
    elif isinstance(obj, (list, tuple)):
        print(f"{type(obj).__name__} length: {len(obj)}")
        for i, v in enumerate(obj[:max_items]):
            _print_summary(f"[{i}]", v, show_data)

    # ----- Numpy array -----
    elif isinstance(obj, np.ndarray):
        print(f"Array shape: {obj.shape}, dtype: {obj.dtype}")
        if show_data:
            print(obj[:max_items])

    # ----- Pandas -----
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        print(f"{type(obj).__name__} shape: {obj.shape}")
        if show_data:
            print(obj.head(max_items))

    # ----- Fallback -----
    else:
        print(f"Object preview: {repr(obj)[:500]}")

    return obj


def _print_summary(name, v, show_data=False):
    """Helper: short summary for nested entries."""
    t = type(v).__name__
    if isinstance(v, np.ndarray):
        print(f"  - {name}: ndarray shape={v.shape}, dtype={v.dtype}")
        if show_data and v.size < 50:
            print(f"    data: {v}")
    elif isinstance(v, (list, tuple)):
        print(f"  - {name}: {t} len={len(v)}")
        if show_data and len(v) < 10:
            print(f"    data: {v}")
    elif isinstance(v, dict):
        print(f"  - {name}: dict with {len(v)} keys: {list(v.keys())[:5]}")
    elif isinstance(v, (pd.DataFrame, pd.Series)):
        print(f"  - {name}: {t} shape={v.shape}")
    else:
        s = str(v)
        print(f"  - {name}: {t} = {s[:80]}{'...' if len(s)>80 else ''}")

