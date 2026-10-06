"""Hashed n-gram features for the embedded part of the arXiv title pool -> cache/pool.{w,c}.npz"""
import sys, numpy as np, pandas as pd, scipy.sparse as sp; sys.path.insert(0, "research")
from feats import prep, HW, HC
n = int(sys.argv[1]); t = [prep(x) for x in pd.read_parquet("cache/arxiv_pool.parquet").title.iloc[:n]]
sp.save_npz("cache/pool.w.npz", HW.transform(t)); sp.save_npz("cache/pool.c.npz", HC.transform(t)); print(n)
