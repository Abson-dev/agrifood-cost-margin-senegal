# -*- coding: utf-8 -*-
"""
Created on Sat Jun 14 21:49:18 2025

@author: AHema
"""

import os
import zipfile

structure = [
    "data/raw/",
    "data/processed/",
    "data/geo/",
    "notebooks/",
    "scripts/",
    "dashboard/",
    "outputs/plots/",
    "outputs/models/",
    "outputs/maps/",
    "outputs/reports/",
    "config/",
    "docs/"
]

files = [
    "notebooks/01_eda.ipynb",
    "notebooks/02_nlp_extraction.ipynb",
    "notebooks/03_model_training.ipynb",
    "notebooks/04_geospatial_analysis.ipynb",
    "scripts/agrifood_pipeline.py",
    "scripts/preprocess.py",
    "scripts/model.py",
    "scripts/visualize.py",
    "dashboard/streamlit_app.py",
    "config/config.yaml",
    "docs/value_chain_diagrams.pdf",
    ".gitignore",
    "requirements.txt",
    "README.md",
    "LICENSE"
]

base = "agrifood-cost-margin-senegal"
os.makedirs(base, exist_ok=True)
for folder in structure:
    os.makedirs(os.path.join(base, folder), exist_ok=True)
for f in files:
    with open(os.path.join(base, f), "w") as file:
        file.write("")

with zipfile.ZipFile(base + ".zip", "w") as zipf:
    for root, _, filenames in os.walk(base):
        for filename in filenames:
            filepath = os.path.join(root, filename)
            arcname = os.path.relpath(filepath, start=".")
            zipf.write(filepath, arcname)

print(f"✅ Created: {base}.zip")
