# Project Title: UK Dataiku Data Scientist Technical Assessment
Created from data-science-template
## Overview
The goal is to develop a predictive model that classifies individuals into income brackets (<=50K or >50K) based on demographic and employment attributes extracted from the U.S. Census Bureau database.

---

## Repository Structure
project-name/  
├─ data/          → raw or sample data (not committed if large/sensitive)  
├─ notebooks/     → Jupyter notebooks for EDA & experiments  
├─ src/           → source code (training, evaluation, utilities)  
├─ models/        → trained models (gitignored if large)  
├─ results/       → predictions, metrics, plots  
├─ slides/        → presentation slides  
├─ tests/         → unit tests  
├─ requirements.txt  
├─ .gitignore  
└─ README.md  

---

## Setup Instructions

### 1. Clone the repository
`git clone https://github.com/<username>/<repo-name>.git`  
`cd <repo-name>`

### 2. Create a virtual environment
`python -m venv .venv`  
`source .venv/bin/activate`   # Mac/Linux  
`.venv\Scripts\activate`     # Windows  

### 3. Install dependencies
`pip install -r requirements.txt`  

---

## Usage

### 1. load, preprocess and model
python src/run.py

### 2. open and run EDA notebook (optional)
jupyter notebook notebooks/eda.ipynb

---

## Results
5 Algorithms have been choosen for an inital screening, pipeline automatically trains and tunes:
Logistic Regression
Decision Tree
Random Forest
Gradient Boosting
XGBoost (performs best) 

---

## Notes
- Keep large data files in `data/` but don’t commit them to GitHub.  
- Save outputs (predictions, metrics, plots) in `results/`.  
- `slides/` folder for final presentations.  

---

## License
This project is licensed under the MIT License — see the LICENSE file for details.
