# agrifood-cost-margin-senegal


agrifood-cost-margin-senegal/
│
├── 📁 data/                     # Raw & intermediate data
│   ├── raw/                    # Survey results, bulletins
│   ├── processed/              # Cleaned CSVs, merged data
│   └── geo/                    # GeoJSONs, shapefiles, rasters

├── 📁 notebooks/               # Jupyter notebooks for EDA, modeling
│   ├── 01_eda.ipynb
│   ├── 02_nlp_extraction.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_geospatial_analysis.ipynb

├── 📁 scripts/                 # Python scripts
│   ├── agrifood_pipeline.py   # Main pipeline
│   ├── preprocess.py
│   ├── model.py
│   └── visualize.py

├── 📁 dashboard/               # Streamlit app
│   └── streamlit_app.py

├── 📁 outputs/                 # Model outputs, plots, rasters
│   ├── plots/
│   ├── models/
│   ├── maps/
│   └── reports/

├── 📁 config/                  # Configurations, YAML files
│   └── config.yaml

├── 📁 docs/                    # Documentation, figures, PDF reports
│   └── value_chain_diagrams.pdf

├── .gitignore                 # Ignore data, models, temporary files
├── requirements.txt           # Python dependencies
├── README.md                  # Project overview
└── LICENSE                    # License file (e.g., MIT)


