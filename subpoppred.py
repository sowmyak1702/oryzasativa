
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler,SMOTE
import pickle
d=pd.read_csv("data/dataprediction.csv")
X = d["Sequence"]
le = LabelEncoder()
d["Subpopulation"]=le.fit_transform(d["Subpopulation"])
y = d["Subpopulation"]

# y=pd.DataFrame(y)
nucleotides = ['A', 'C', 'G', 'T']
one_hot = np.zeros((len(X), len(X[0]) * len(nucleotides)))
for i, seq in enumerate(X):
    for j, nuc in enumerate(seq):
        index = nucleotides.index(nuc)
        one_hot[i, j*len(nucleotides) + index] = 1
oversampler = SMOTE(random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(one_hot, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# random forest
rfc = RandomForestClassifier( max_depth =30, max_features= 'sqrt',criterion="gini", min_samples_leaf=1, min_samples_split=5, n_estimators=30)

# Fit the model to the training data
rfc.fit(X_resampled, y_resampled)
pickle.dump(rfc,open("subpoppred.pkl","wb"))

# y_pred = rfc.predict(X_test)
# new_seq= "GCTGTTTCC"
# one_hot1 = np.zeros((1, len(new_seq) * len(nucleotides)))
# for i, nuc in enumerate(new_seq):
#     index = nucleotides.index(nuc)
#     one_hot1[0, i*len(nucleotides) + index] = 1

# subpop=rfc.predict(one_hot1)
# print(subpop)
