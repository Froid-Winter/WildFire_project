"""
Quick script: read processed training CSV, fit StandardScaler, save to disk.
"""
import os, sys, glob, joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(__file__))
from config import FEATURE_COLS

data_dir = os.path.join(os.path.dirname(__file__), 'processed_data', 'train')
files = sorted(glob.glob(os.path.join(data_dir, 'part-*.csv')))
train_pdf = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

X_train = train_pdf[FEATURE_COLS].values.astype(np.float32)
scaler = StandardScaler()
scaler.fit(X_train)

out = os.path.join(os.path.dirname(__file__), 'saved_model', 'scaler.joblib')
joblib.dump(scaler, out)
print(f"Scaler saved to {out}")
print(f"Mean shape: {scaler.mean_.shape}")
