# ml-in-a-box
A Dockerized template repo that lets anyone train and test an ML model with zero local setup.

```yaml

ml-in-a-box/
├── docker/
│   └── Dockerfile
├── notebooks/
│   └── 00_Intro.ipynb
├── requirements.txt
├── .devcontainer/          # Optional for VSCode users
│   └── devcontainer.json
├── README.md
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