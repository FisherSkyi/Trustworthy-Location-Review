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
## Dataset Preprocessing and Statistical Analysis

```bsh
 Dataset Info:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1100 entries, 0 to 1099
Data columns (total 2 columns):
 #   Column       Non-Null Count  Dtype 
---  ------       --------------  ----- 
 0   review_text  1100 non-null   object
 1   rating       1100 non-null   int64 
dtypes: int64(1), object(1)
memory usage: 17.3+ KB
None

 Dataset shape: (1100, 2)

 First 5 reviews:
                                         review_text  rating
0  We went to Marmaris with my wife for a holiday...       5
1  During my holiday in Marmaris we ate here to f...       4
2  Prices are very affordable. The menu in the ph...       3
3  Turkey's cheapest artisan restaurant and its f...       5
4  I don't know what you will look for in terms o...       3

 Data Quality Check:
- Total reviews: 1100
- Average review length: 110.8 characters
```