# FeatAug

This is the code repository of the paper "FeatAug: Automatic Feature Augmentation From One-to-Many Relationship Tables"

## Dataset Download

**Tmall Dataset:** https://tianchi.aliyun.com/dataset/dataDetail?dataId=42

**Instacart Dataset:** https://www.kaggle.com/c/instacart-marketbasket-analysis

**Student Dataset:** https://www.kaggle.com/competitions/predict-studentperformance-from-game-play

**Merchant Dataset:** https://www.kaggle.com/competitions/elo-merchant-category-recommendation

**Covtype Dataset:** https://archive.ics.uci.edu/dataset/31/covertype

**Household Dataset:** https://www.kaggle.com/c/costa-rican-household-poverty-prediction/data

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

```

'lr' -> Logistic Regression

'rf' -> Random Forest

'xgb' -> XGBoost

'deepfm' -> DeepFM (https://www.ijcai.org/proceedings/2017/0239.pdf)

'dcnv2' -> DCNV2 (https://arxiv.org/pdf/2008.13535.pdf)

```
