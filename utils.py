"""A central place for shared helper functions.

This file contains:
 - tensor_to_numpy: safe conversion of torch tensors to numpy arrays
 - load_artifact_from_hub: cloud first artifact loader (with local fallback)
 - load_model_from_hub: cloud first model loader (NO local fallback on failure — prints message)
 - get_db_engine: helper that tries MySQL then optional SQLite fallback

 - load_model_from_hub now attempts to load the model/tokenizer from the Hugging Face repo
   and WILL NOT FALL BACK to a local directory if the cloud load fails. Instead it prints
   a clear message and returns (None, None).
"""

import os
import pandas as pd
import numpy as np
import torch
from typing import Optional, Callable, Any 
from huggingface_hub import hf_hub_download
from transformers import BertForSequenceClassification, BertTokenizer
from urllib.parse import quote_plus
from sqlalchemy import create_engine, text

def tensor_to_numpy(t):
    """
    Safely convert a torch Tensor (possibly on GPU) to a numpy ndarray.
    - If `t` is a torch tensor, call t.cpu().numpy()
    - Otherwise, fall back to np.asarray(t)
    If conversion fails, raises a RuntimeError so the caller can handle it.
    """
    try:
        if hasattr(t, "cpu"):
            return t.cpu().numpy()
        return np.asarray(t)
    except Exception as e:
        # Fail loudly so the problem is obvious during debugging
        raise RuntimeError(f"Could not convert tensor to numpy: {e}")

def load_artifact_from_hub(
    repo_id: str,
    filename: str,
    local_fallback_path: Optional[str] = None, 
    load_fn: Callable[[str], Any], 
):
    """
    Tries to download and load an artifact from Hugging Face Hub. If it fails, falls back to local path.

    Args:
        - repo_id: "<user>/<repo>" or repo id on HF
        - filename: filename inside the dataset repo
        - local_fallback_path: local path to try if cloud fails
        - load_fn: function that takes a path and returns loaded object

    Returns:
        - result of load_fn(path) or None if both cloud and local loading fail
    """
    hf_token = os.getenv("HF_TOKEN", None)
    try:
        # Try to download from the Hub (token used for private repos)
        downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", token=hf_token)
        print(f" Successfully downloaded '{filename}' from Hub.")
        return load_fn(downloaded_path)
    except Exception as e_hf:
        print(f" Could not download '{filename}' from Hub: {e_hf}")
        # Try to load from local fallback
        if local_fallback_path and os.path.exists(local_fallback_path):
            try:
                print(f" Attempting to load local fallback: {local_fallback_path}")
                return load_fn(local_fallback_path)
            except Exception as e_local:
                print(f" Failed to load local fallback '{local_fallback_path}': {e_local}")
                return None
        else:
            print(f" Local fallback '{local_fallback_path}' not found.")
            return None
        
def load_model_from_hub(repo_id: str):
    """
    Cloud-first loader for the entire model (tokenizer + classification head).

    This function will NOT attempt to fallback to a local model directory. If loading from the Hugging Face Hub fails, the
    function prints a clear message and returns (None, None).

    Args:
        - repo_id: Hugging Face model repo id (e.g. "username/model-name")

    Returns:
        - (tokenizer, model) on success
        - (None, None) on failure (and prints an explanation)
    """
    hf_token = os.getenv("HF_TOKEN", None)

    try:
        # Try to load tokenizer and model from HF model hub using the token.
        # use_auth_token is accepted by transformers.from_pretrained for authenticated loads.
        print(f"Attempting to load model/tokenizer from Hugging Face repo: {repo_id}")
        # NOTE: newer versions of transformers accept token arg; use use_auth_token for compatibility
        tokenizer = BertTokenizer.from_pretrained(repo_id, use_auth_token=hf_token)
        model = BertForSequenceClassification.from_pretrained(repo_id, use_auth_token=hf_token)
        print(f" Successfully loaded model and tokenizer from Hub: {repo_id}")
        return tokenizer, model

    except Exception as e_hf:
        # DO NOT try local fallback here. Instead, print and return None.
        print(" Could not load model from Hugging Face Hub.")
        print("   Reason:", e_hf)
        print("   Action: The loader will NOT fallback to a local directory. Please ensure HF_TOKEN and HF_USER are correct and the repo exists or is public.")
        return None, None

def get_db_engine(mysql_user,
                  mysql_password,
                  mysql_host,
                  mysql_port,
                  mysql_db,
                  sqlite_fallback_path=None,
                  test_connection=True):
    """
    robust helper to create a SQLAlchemy engine.

    - Tries to use MySQL (cloud) when mysql_user & mysql_password & mysql_db are provided.
    - If MySQL connection fails OR credentials are not provided, and sqlite_fallback_path
      is given, it will return an engine for that local SQLite file.
    - If test_connection=True the function will run a quick `SELECT 1` test.

    Raises on failure if no fallback path is available.
    """
    # If credentials appear present, prefer MySQL
    if mysql_user and mysql_password and mysql_db:
        encoded_password = quote_plus(str(mysql_password))
        db_url = f"mysql+pymysql://{mysql_user}:{encoded_password}@{mysql_host}:{mysql_port}/{mysql_db}"
        try:
            engine = create_engine(db_url)
            if test_connection:
                # quick connectivity test
                with engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
            # Success -> return MySQL engine
            return engine
        except Exception as e:
            # If MySQL fails, fall back to SQLite if provided
            print(f" Could not connect to MySQL ({e}).")
            if sqlite_fallback_path:
                print(f" Falling back to local SQLite DB at: {sqlite_fallback_path}")
                sqlite_url = f"sqlite:///{sqlite_fallback_path}"
                return create_engine(sqlite_url)
            # No fallback provided -> re-raise the original error to make failure explicit
            raise

    # If no MySQL creds, use sqlite fallback if available
    if sqlite_fallback_path:
        sqlite_url = f"sqlite:///{sqlite_fallback_path}"
        return create_engine(sqlite_url)

    # If nothing is available, raise an informative error
    raise ValueError("No database credentials supplied and no sqlite_fallback_path provided.")
