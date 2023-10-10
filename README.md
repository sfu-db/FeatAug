# FeatAug

## Dataset Download

## How to Run Experiments

#### Step 1: Install Poetry
```
pip install poetry
```

#### Step 2: Using Poetry to Install Running Environment
```
poetry install
```

#### Step 3: Run Experiments for Each Dataset
```
poetry run python exp/instacart.py -m 'lr'
```
The parameter **-m** means the classifier/regressor the users can choose:
'lr' -> Logistic Regression
'rf' -> Random Forest
'xgb' -> XGBoost
'deepfm' -> DeepFM (https://www.ijcai.org/proceedings/2017/0239.pdf)
'dcnv2' -> DCNV2 (https://arxiv.org/pdf/2008.13535.pdf)

