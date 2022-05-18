### Tree모델인 lgbm, xgb와 선형회귀 logistic regression 세개의 모델을 사용해 각각 벡터차원수, 윈도우, 최소단어수를 조절하고,
### averaging을 진행해 예측함

### Imports
import pandas as pd
import numpy as np
import os
from gensim.models import word2vec
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats.mstats import gmean
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import VotingClassifier
#os.environ["PYTHONHASHSEED"] = "123"

### Read data
df_train = pd.read_csv('X_train.csv', encoding='cp949')
df_test = pd.read_csv('X_test.csv', encoding='cp949')
y_train = pd.read_csv('y_train.csv').gender
IDtest = df_test.cust_id.unique()

### Make corpus
p_level = 'gds_grp_nm'  # 상품 분류 수준

# W2V 학습을 하기에는 데이터(즉 corpus)가 부족하여 
# 고객별로 구매한 상품 목록으로부터 n배 oversampling을 수행
def oversample(x, n, seed=0):
    if n == 0:
        return list(x)
    uw = np.unique(x)
    bs = np.array([])
    np.random.seed(seed)
    for j in range(n):
        bs = np.append(bs, np.random.choice(uw, len(uw), replace=False))
    return list(bs)

X_train = list(df_train.groupby('cust_id')[p_level].agg(oversample, 20))
X_test = list(df_test.groupby('cust_id')[p_level].agg(oversample, 20))

### 모델 1
num_features = 200 # 단어 벡터 차원 수
min_word_count = 1 # 최소 단어 수
context = 6 # 학습 윈도우(인접한 단어 리스트) 크기

# 
w2v = word2vec.Word2Vec(X_train, 
                        size=num_features, 
                        min_count=min_word_count,
                        window=context,
                        seed=0, workers=1)
# 필요없는 메모리 unload
w2v.init_sims(replace=True)

### Make features
# 구매상품에 해당하는 벡터의 평균/최소/최대 벡터를 feature로 만드는 전처리기(pipeline에서 사용 가능)
class EmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = num_features
    def fit(self, X, y):
        return self
    def transform(self, X):
        return np.array([
            np.hstack([
                np.max([self.word2vec[w] for w in words if w in self.word2vec] or [np.zeros(self.dim)], axis=0),
                np.min([self.word2vec[w] for w in words if w in self.word2vec] or [np.zeros(self.dim)], axis=0),
                np.mean([self.word2vec[w] for w in words if w in self.word2vec] or [np.zeros(self.dim)], axis=0)                
            ]) 
            for words in X
        ])  
    
model1 = Pipeline([
    ("W2V vectorizer", EmbeddingVectorizer(w2v.wv)),
    ("lightgbm", LGBMClassifier(random_state=0))])

X_tr, X_te, y_tr, y_te = train_test_split(X_train,y_train,test_size=0.2)

model1.fit(X_tr, y_tr)
print(roc_auc_score(y_te,model1.predict_proba(X_te)[:,1]))

model1 = Pipeline([
    ("W2V vectorizer", EmbeddingVectorizer(w2v.wv)),
    ("lightgbm", LGBMClassifier(random_state=0))])

model1.fit(X_train, y_train)

pred1 = model1.predict_proba(X_test)[:,1]

# 모델 2
num_features = 400 # 단어 벡터 차원 수
min_word_count = 2 # 최소 단어 수
context = 4 # 학습 윈도우(인접한 단어 리스트) 크기

# 
w2v = word2vec.Word2Vec(X_train, 
                        size=num_features, 
                        min_count=min_word_count,
                        window=context,
                        seed=0, workers=1)
# 필요없는 메모리 unload
w2v.init_sims(replace=True)

### Make features
# 구매상품에 해당하는 벡터의 평균/최소/최대 벡터를 feature로 만드는 전처리기(pipeline에서 사용 가능)
class EmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = num_features
    def fit(self, X, y):
        return self
    def transform(self, X):
        return np.array([
            np.hstack([
                np.max([self.word2vec[w] for w in words if w in self.word2vec] or [np.zeros(self.dim)], axis=0),
                np.min([self.word2vec[w] for w in words if w in self.word2vec] or [np.zeros(self.dim)], axis=0),
                np.mean([self.word2vec[w] for w in words if w in self.word2vec] or [np.zeros(self.dim)], axis=0)                
            ]) 
            for words in X
        ])  

     
model2 = Pipeline([
    ("W2V vectorizer", EmbeddingVectorizer(w2v.wv)),
    ("xgboost", XGBClassifier(random_state=0))])

model2.fit(X_tr, y_tr)
print(roc_auc_score(y_te,model2.predict_proba(X_te)[:,1]))

model2 = Pipeline([
    ("W2V vectorizer", EmbeddingVectorizer(w2v.wv)),
    ("xgboost", XGBClassifier(random_state=0))])

model2.fit(X_train, y_train)
pred2 = model2.predict_proba(X_test)[:,1]

### 모델 3
num_features = 300 # 단어 벡터 차원 수
min_word_count = 2 # 최소 단어 수
context = 3 # 학습 윈도우(인접한 단어 리스트) 크기

# 
w2v = word2vec.Word2Vec(X_train, 
                        size=num_features, 
                        min_count=min_word_count,
                        window=context,
                        seed=0, workers=1)
# 필요없는 메모리 unload
w2v.init_sims(replace=True)

### Make features
# 구매상품에 해당하는 벡터의 평균/최소/최대 벡터를 feature로 만드는 전처리기(pipeline에서 사용 가능)
class EmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.dim = num_features
    def fit(self, X, y):
        return self
    def transform(self, X):
        return np.array([
            np.hstack([
                np.max([self.word2vec[w] for w in words if w in self.word2vec] or [np.zeros(self.dim)], axis=0),
                np.min([self.word2vec[w] for w in words if w in self.word2vec] or [np.zeros(self.dim)], axis=0),
                np.mean([self.word2vec[w] for w in words if w in self.word2vec] or [np.zeros(self.dim)], axis=0)                
            ]) 
            for words in X
        ])  

model3 = Pipeline([
    ("W2V vectorizer", EmbeddingVectorizer(w2v.wv)),
    ("lr", LogisticRegression(random_state=0))])

model3.fit(X_tr, y_tr)
print(roc_auc_score(y_te,model3.predict_proba(X_te)[:,1]))

model3 = Pipeline([
    ("W2V vectorizer", EmbeddingVectorizer(w2v.wv)),
    ("lr", LogisticRegression(random_state=0))])
model3.fit(X_train,y_train)
pred3 = model3.predict_proba(X_test)[:,1]

models_for_ensemble = [('model1', model1),('model2', model2),('model3', model3)] 
voting = VotingClassifier(
    estimators = [(name,clf) for name, clf in models_for_ensemble],
    voting='soft')
voting_score = voting.fit(X_train, y_train)
pred_voting = voting.predict_proba(X_test)[:,1]

fname = '김보현1.csv'
submissions = pd.concat([pd.Series(IDtest, name="cust_id"), pd.Series(pred_voting, name="gender")] ,axis=1)
submissions.to_csv(fname, index=False)
print("'{}' is ready to submit." .format(fname))
