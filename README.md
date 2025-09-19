# ml-in-a-box
A Dockerized template repo that lets anyone train and test an ML model with zero local setup.

```yaml

ml-in-a-box/
├── data/                 # placeholder with README on where data goes
├── notebooks/            # exploratory work
│   └── example.ipynb     # demo notebook
├── src/                  # actual code
│   ├── data_cleaning.py  # preprocessing pipeline
│   ├── train.py          # training script
│   ├── evaluate.py       # metrics & plots
│   └── utils.py          # helper functions
├── tests/                # unit tests for src
├── requirements.txt      
├── setup.py              # so it can be installed locally as a package
├── README.md             # polished, with gifs & quickstart
└── .gitignore


```
---

## Features
- Python 3 virtual environment
- Jupyter-ready with key ML libraries
- Optional Docker setup

---
## Setup

```bash
    python -m venv .venv
    source .venv/Scripts/activate
    pip install -r requirements.txt
```
---

## Start Jupyter
```bash
    jupyter notebook
```