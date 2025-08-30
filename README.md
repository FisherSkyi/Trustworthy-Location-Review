# Filtering the Noise: ML for Trustworthy Location Reviews

**Challenge:** Design and implement an ML-based system to evaluate the quality and relevancy of Google location reviews

**Problem Statement:**
- Gauge review quality: Detect spam, advertisements, irrelevant content, and rants
- Assess relevancy: Determine if review content is genuinely related to the location
- Enforce policies: Automatically flag reviews violating predefined policies

## Run in Colab

The Jupyter Notebook version is available as **jotham.ipynb**. Upload it to Colab and run the notebook.

---

~~In case you want to run on local machine~~

## Set Up
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

To download dataset from kaggle, you need:

1. Create a Kaggle API token on https://www.kaggle.com → Account → Create New API Token (downloads kaggle.json).
2. Put kaggle.json in proper location and set permission
- Linux/macOS: 
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```
- Windows
```bash
mkdir $env:USERPROFILE\.kaggle; 
move .\kaggle.json $env:USERPROFILE\.kaggle\kaggle.json
```
3. run:
```bash
python download_kaggle_dataset.py
```
