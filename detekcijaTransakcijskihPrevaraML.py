# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 13:03:01 2023

@author: korisnik
"""

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import time
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
import warnings

warnings.filterwarnings("ignore")

ccd = pd.read_csv('creditcard.csv');

ccd.isnull().sum().max()

plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=ccd, palette='Set2')
plt.title('Distribucija klasa prije obrade', fontsize=14)
plt.xlabel('Klasa')
plt.ylabel('Brojnost')

amount_val = ccd['Amount'].values
print(amount_val)
time_val = ccd['Time'].values
max_amount = ccd['Amount'].max()
print("Najveća vrijednost u koloni 'Amount' je:", max_amount)

fig, ax = plt.subplots(1, 2, figsize=(18,4))
sns.distplot(amount_val, ax=ax[0], color='g')
ax[0].set_title('Distribucija vrijednosti iz kolone "Amount"', fontsize=14)
ax[0].set_xlim([min(amount_val), max(amount_val)])
ax[0].set_xlabel('Iznos transakcije', fontsize=12)
ax[0].set_ylabel('Gustoća vjerojatnoce', fontsize=12)

sns.distplot(time_val, ax=ax[1], color='y')
ax[1].set_title('Distribucija vrijednosti iz kolone "Time"', fontsize=14)
ax[1].set_xlim([min(time_val), max(time_val)])
ax[1].set_xlabel('Vrijeme transakcije', fontsize=12)
ax[1].set_ylabel('Gustoća vjerojatnoce', fontsize=12)

print("Vrijednosti atributa 'Amount' prije skaliranja")
print(ccd['Amount'].head(5))

print("Vrijednosti atributa 'Time' prije skaliranja")
print(ccd['Time'].tail(5))


from sklearn.preprocessing import StandardScaler, RobustScaler
# StandardScaler i RobustScaler su metode za skaliranje podataka

std_scaler = StandardScaler()
rob_scaler = RobustScaler()

ccd['scaled_amount'] = rob_scaler.fit_transform(ccd['Amount'].values.reshape(-1,1))
ccd['scaled_time'] = rob_scaler.fit_transform(ccd['Time'].values.reshape(-1,1))
# Skalirane vrijednosti se pohranjuju u nove stupce "scaled_amount" i "scaled_time".
#reshape(-1, 1) sluzi da se 1D niz koji ima vrijednosti iz kolone Amount pretvori u 2D niz koji ce se unijeti u dataframe

ccd.drop(['Time','Amount'], axis=1, inplace=True)
# Originalni atributi "Time" i "Amount" se uklanjaju iz DataFrame-a

print("Vrijednosti atributa 'Amount' nakon skaliranja")
print(ccd['scaled_amount'].head())

print("Vrijednosti atributa 'Time' nakon skaliranja")
print(ccd['scaled_time'].tail())

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

print('Legalne transakcije', round(ccd['Class'].value_counts()[0]/len(ccd) * 100,2), '% skupa podataka')
print('Prevare', round(ccd['Class'].value_counts()[1]/len(ccd) * 100,2), '% skupa podataka')

X = ccd.drop('Class', axis=1)
y = ccd['Class']

sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

for train_index, test_index in sss.split(X, y):
    print("Train:", train_index, "Test:", test_index)
    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]


# prevođenje u red
original_Xtrain = original_Xtrain.values
original_Xtest = original_Xtest.values
original_ytrain = original_ytrain.values
original_ytest = original_ytest.values

# Provjeravamo kako su klase distribuirane u trening i test skupu
train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)
test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)
print('-' * 100)

print('Distribucija klasa: \n')
print(train_counts_label/ len(original_ytrain))
print(test_counts_label/ len(original_ytest))

# Obzirom da je raspodjela klasa u ciljnom atributu ("Class") neuravnotežena,
# stvaramo poduzorke (subsamples) podataka kako bi se postigla ravnoteža između klasa
"""
#Random poduzorkovanje
rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)
new_ccd = pd.DataFrame(X_resampled, columns=X.columns)
new_ccd['Class'] = y_resampled

"""
ccd = ccd.sample(frac=1)

# 492 slučaja prevara
fraud_ccd = ccd.loc[ccd['Class'] == 1]
non_fraud_ccd = ccd.loc[ccd['Class'] == 0][:492]

normal_distributed_ccd = pd.concat([fraud_ccd, non_fraud_ccd])

# nasumično premiještanje redova u DataFrame-u 
new_ccd = normal_distributed_ccd.sample(frac=1, random_state=42)

print(new_ccd.head())

print('Distribucija klasa unutar dobijenog podskupa podataka')
print(new_ccd['Class'].value_counts()/len(new_ccd))

print('----' * 63)
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=new_ccd, palette='Set2')
plt.title('Distribucija klasa nakon obrade', fontsize=14)
plt.xlabel('Klasa')
plt.ylabel('Brojnost')

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24,20))

# Cijeli skup podataka
corr = ccd.corr()
sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax1)
ax1.set_title("Korelacijska matrica za nebalansirani skup podataka", fontsize=14)

sub_sample_corr = new_ccd.corr()
sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax2)
ax2.set_title('Korelacijska matrica za balansirani skup podataka', fontsize=14)

from scipy.stats import norm
plt.figure()
f, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(20, 6))

v14_fraud_dist = new_ccd['V14'].loc[new_ccd['Class'] == 1].values
sns.distplot(v14_fraud_dist,ax=ax1, fit=norm, color='#FB8861', kde=False)
ax1.set_title('Distribucija vrijednosti atributa V14', fontsize=14)

v12_fraud_dist = new_ccd['V12'].loc[new_ccd['Class'] == 1].values
sns.distplot(v12_fraud_dist,ax=ax2, fit=norm, color='#56F9BB', kde=False)
ax2.set_title('Distribucija vrijednosti atributa V12', fontsize=14)


v10_fraud_dist = new_ccd['V10'].loc[new_ccd['Class'] == 1].values
sns.distplot(v10_fraud_dist,ax=ax3, fit=norm, color='#C5B3F9', kde=False)
ax3.set_title('Distribucija vrijednosti atributa V10', fontsize=14)

v17_fraud_dist = new_ccd['V17'].loc[new_ccd['Class'] == 1].values
sns.distplot(v17_fraud_dist,ax=ax4, fit=norm, color='#C5B3F9', kde=False)
ax4.set_title('Distribucija vrijednosti atributa V17', fontsize=14)
print('----' * 44)

total_instances_before = len(new_ccd)
print("Ukupan broj instanci prije uklanjanja outlier-a:", total_instances_before)

# Prikazuje ukupan broj prevara prije uklanjanja outlier-a
total_frauds_before = len(new_ccd[new_ccd['Class'] == 1])
print("Ukupan broj prevara prije uklanjanja outlier-a:", total_frauds_before)

# Prikazuje ukupan broj legalnih transakcija prije uklanjanja outlier-a
total_legs_before = len(new_ccd[new_ccd['Class'] == 0])
print("Ukupan broj legalnih transakcija prije uklanjanja outlier-a:", total_legs_before)

v14_fraud = new_ccd['V14'].loc[new_ccd['Class'] == 1].values

print("V14")


q25, q75 = np.percentile(v14_fraud, 25), np.percentile(v14_fraud, 75)
print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))
v14_iqr = q75 - q25
print('iqr: {}'.format(v14_iqr))
v14_cut_off = v14_iqr * 1.5
v14_lower, v14_upper = q25 - v14_cut_off, q75 + v14_cut_off
print('Odsječak: {}'.format(v14_cut_off))
print('Donja granica za V14: {}'.format(v14_lower))
print('Gornja granica za V14: {}'.format(v14_upper))
outliers = [x for x in v14_fraud if x < v14_lower or x > v14_upper]
print('Ukupan broj outlier-a za atribut V14 u slučaju prevare : {}'.format(len(outliers)))
print('V14 vrijednosti outliers-a:{}'.format(outliers))

new_ccd = new_ccd.drop(new_ccd[(new_ccd['V14'] > v14_upper) | (new_ccd['V14'] < v14_lower)].index)
print('Broj instanci nakon uklanjanja outliers-a: {}'.format(len(new_ccd)))

print('----' * 63)

total_instances_after = len(new_ccd)
print("Ukupan broj instanci nakon uklanjanja outlier-a:", total_instances_after)

# Prikazuje ukupan broj prevara prije uklanjanja outlier-a
total_frauds_after = len(new_ccd[new_ccd['Class'] == 1])
print("Ukupan broj prevara nakon uklanjanja outlier-a:", total_frauds_after)

# Prikazuje ukupan broj prevara prije uklanjanja outlier-a
total_legs_after = len(new_ccd[new_ccd['Class'] == 0])
print("Ukupan broj legalnih transakcija nakon uklanjanja outlier-a:", total_legs_after)

v12_fraud = new_ccd['V12'].loc[new_ccd['Class'] == 1].values
q25, q75 = np.percentile(v12_fraud, 25), np.percentile(v12_fraud, 75)
v12_iqr = q75 - q25

v12_cut_off = v12_iqr * 1.5
v12_lower, v12_upper = q25 - v12_cut_off, q75 + v12_cut_off
print('Donja granica za V12: {}'.format(v12_lower))
print('Gornja granica za V12: {}'.format(v12_upper))
outliers = [x for x in v12_fraud if x < v12_lower or x > v12_upper]
print('V12 outliers: {}'.format(outliers))
print('Ukupan broj outlier-a za atribut V12 u slučaju prevare: {}'.format(len(outliers)))
new_ccd = new_ccd.drop(new_ccd[(new_ccd['V12'] > v12_upper) | (new_ccd['V12'] < v12_lower)].index)

print('Broj instanci nakon uklanjanja outliers-a: {}'.format(len(new_ccd)))
print('----' * 44)

# Uklanjanje outliers-a iz atributa V10
v10_fraud = new_ccd['V10'].loc[new_ccd['Class'] == 1].values
q25, q75 = np.percentile(v10_fraud, 25), np.percentile(v10_fraud, 75)
v10_iqr = q75 - q25

v10_cut_off = v10_iqr * 1.5
v10_lower, v10_upper = q25 - v10_cut_off, q75 + v10_cut_off
print('Donja granica za V10: {}'.format(v10_lower))
print('Gornja granica za V10: {}'.format(v10_upper))
outliers = [x for x in v10_fraud if x < v10_lower or x > v10_upper]
print('V10 outliers: {}'.format(outliers))
print('Ukupan broj outlier-a za atribut V10 u slučaju prevare: {}'.format(len(outliers)))
new_ccd = new_ccd.drop(new_ccd[(new_ccd['V10'] > v10_upper) | (new_ccd['V10'] < v10_lower)].index)
print('Broj instanci nakon uklanjanja outliers-a: {}'.format(len(new_ccd)))


count_v17_fraud_before = len(new_ccd[(new_ccd['Class'] == 1) & (new_ccd['V17'])])
print("Broj instanci klase 'V17' u slučajevima prijevare prije uklanjanja outlier-a:", count_v17_fraud_before)

# Uklanjanje outliers-a iz atributa V17
v17_fraud = new_ccd['V17'].loc[new_ccd['Class'] == 1].values
q25, q75 = np.percentile(v17_fraud, 25), np.percentile(v17_fraud, 75)
v17_iqr = q75 - q25

v17_cut_off = v17_iqr * 1.5
v17_lower, v17_upper = q25 - v17_cut_off, q75 + v17_cut_off
print('Donja granica za V17: {}'.format(v17_lower))
print('Gornja granica za V17: {}'.format(v17_upper))
outliers = [x for x in v17_fraud if x < v17_lower or x > v17_upper]
print('V17 outliers: {}'.format(outliers))
print('Ukupan broj outlier-a za atribut V17 u slučaju prevare: {}'.format(len(outliers)))
new_ccd = new_ccd.drop(new_ccd[(new_ccd['V17'] > v17_upper) | (new_ccd['V17'] < v17_lower)].index)
print('Broj instanci nakon uklanjanja outliers-a: {}'.format(len(new_ccd)))
count_v17_fraud_after = len(new_ccd[(new_ccd['Class'] == 1) & (new_ccd['V17'])])
print("Broj instanci klase 'V17' u slučajevima prijevare poslije uklanjanja outlier-a:", count_v17_fraud_after)
missing_v17_for_fraud = new_ccd[(new_ccd['Class'] == 1) & (new_ccd['V17'].isnull())]
v17_counts = ccd['V17'].value_counts()
print(v17_counts)

# Provjerava vrijednosti koje se pojavljuju više puta (npr., više od jednom)
repeated_v17_values = v17_counts[v17_counts > 1]

# Prikazuje vrijednosti koje se pojavljuju više puta i broj ponavljanja
print("Vrijednosti atributa 'V17' koje se ponavljaju više puta:")
print(repeated_v17_values)
plt.figure()
f, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, figsize=(20, 6))
print("V14")

v14_fraud_dist = new_ccd['V14'].loc[new_ccd['Class'] == 1].values
sns.distplot(v14_fraud_dist,ax=ax1, fit=norm, color='#FB8861', kde=False)
ax1.set_title('Distribucija vrijednosti\nV14 nakon uklanjanja outliera-a', fontsize=14)

v12_fraud_dist = new_ccd['V12'].loc[new_ccd['Class'] == 1].values
sns.distplot(v12_fraud_dist,ax=ax2, fit=norm, color='#56F9BB', kde=False)
ax2.set_title('Distribucija vrijednosti\nV12 nakon uklanjanja outliera-a', fontsize=14)

v10_fraud_dist = new_ccd['V10'].loc[new_ccd['Class'] == 1].values
sns.distplot(v10_fraud_dist,ax=ax3, fit=norm, color='#C5B3F9', kde=False)
ax3.set_title('Distribucija vrijednosti\nV10 nakon uklanjanja outliera-a', fontsize=14)

v17_fraud_dist = new_ccd['V17'].loc[new_ccd['Class'] == 1].values
sns.distplot(v17_fraud_dist,ax=ax4, fit=norm, color='#C5B3F9', kde=False)
ax4.set_title('Distribucija vrijednosti\nV17 nakon uklanjanja outliera-a', fontsize=14)

total_instances_after = len(new_ccd)
print("Ukupan broj instanci nakon uklanjanja outlier-a:", total_instances_after)

# Prikazuje ukupan broj prevara prije uklanjanja outlier-a
total_frauds_after = len(new_ccd[new_ccd['Class'] == 1])
print("Ukupan broj prevara nakon uklanjanja outlier-a:", total_frauds_after)

total_leg_after = len(new_ccd[new_ccd['Class'] == 0])
print("Ukupan broj legalnih transakcija nakon uklanjanja outlier-a:", total_leg_after)
"""
#  novi skup podataka iz podataka dobivenih metodom random undersampling-a
X = new_ccd.drop('Class', axis=1)
y = new_ccd['Class']

# T-SNE Implementation
t0 = time.time()
X_reduced_tsne = TSNE(n_components=2, random_state=42).fit_transform(X.values)
t1 = time.time()
print("T-SNE took {:.2} s".format(t1 - t0))

# PCA Implementation
t0 = time.time()
X_reduced_pca = PCA(n_components=2, random_state=42).fit_transform(X.values)
t1 = time.time()
print("PCA took {:.2} s".format(t1 - t0))

# TruncatedSVD
t0 = time.time()
X_reduced_svd = TruncatedSVD(n_components=2, algorithm='randomized', random_state=42).fit_transform(X.values)
t1 = time.time()
print("Truncated SVD took {:.2} s".format(t1 - t0))
"""
# Poduzorkovanje prije kros validacije vodi ka prenaučenosti modela
X = new_ccd.drop('Class', axis=1)
y = new_ccd['Class']

from sklearn.model_selection import train_test_split

# Ovo se eksplicitno koristi za poduzorkovanje
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Pretvaranje vrijednosti u red da se prilagode klasifikatorima
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

# Implementacija klasifikatora

classifiers = {
    "LogisiticRegression": LogisticRegression(),
    "KNearest": KNeighborsClassifier(),
    "DecisionTreeClassifier": DecisionTreeClassifier()
}

from sklearn.model_selection import cross_val_score


for key, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    training_score = cross_val_score(classifier, X_train, y_train, cv=5)
    print("Klasifikator: ", classifier.__class__.__name__, "ima rezultat treniranja od", round(training_score.mean(), 2) * 100, "% tačnosti")
    
# Korištenje GridSearchCV za pronalazak najboljih parametara
from sklearn.model_selection import GridSearchCV

# Logistička regresija
log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}

grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)
grid_log_reg.fit(X_train, y_train)
# Automatski dobijamo najbolje estimatore za logreg
log_reg = grid_log_reg.best_estimator_

knears_params = {"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)
grid_knears.fit(X_train, y_train)
# KNN
knears_neighbors = grid_knears.best_estimator_

# Stablo odluke
tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), 
              "min_samples_leaf": list(range(5,7,1))}
grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)
grid_tree.fit(X_train, y_train)
tree_clf = grid_tree.best_estimator_



#  Evaluacija performansi logističke regresije korištenjem kros validacije
log_reg_score = cross_val_score(log_reg, X_train, y_train, cv=5)
print('Rezultat logističke regresije nakon primjene najboljih estimatora: ', round(log_reg_score.mean() * 100, 2).astype(str) + '%')


knears_score = cross_val_score(knears_neighbors, X_train, y_train, cv=5)
print('Rezultat KNN algoritma nakon primjene najboljih estimatora: ', round(knears_score.mean() * 100, 2).astype(str) + '%')

tree_score = cross_val_score(tree_clf, X_train, y_train, cv=5)
print('Rezultat Stabla odluke nakon primjene najboljih estimatora: ', round(tree_score.mean() * 100, 2).astype(str) + '%')


# Poduzorkovanje tokom svake iteracije kros validacije
undersample_X = ccd.drop('Class', axis=1)
undersample_y = ccd['Class']

for train_index, test_index in sss.split(undersample_X, undersample_y):
    print("Train:", train_index, "Test:", test_index)
    undersample_Xtrain, undersample_Xtest = undersample_X.iloc[train_index], undersample_X.iloc[test_index]
    undersample_ytrain, undersample_ytest = undersample_y.iloc[train_index], undersample_y.iloc[test_index]
    
undersample_Xtrain = undersample_Xtrain.values
undersample_Xtest = undersample_Xtest.values
undersample_ytrain = undersample_ytrain.values
undersample_ytest = undersample_ytest.values 

undersample_accuracy = []
undersample_precision = []
undersample_recall = []
undersample_f1 = []
undersample_auc = []
from imblearn.pipeline import make_pipeline

# Implementacija NearMiss tehnike 
X_nearmiss, y_nearmiss = NearMiss().fit_resample(undersample_X.values, undersample_y.values)
print('NearMiss distribucija klasa: {}'.format(Counter(y_nearmiss)))


for train, test in sss.split(undersample_Xtrain, undersample_ytrain):
    undersample_pipeline = make_pipeline(NearMiss(sampling_strategy='majority'), log_reg) # SMOTE happens during Cross Validation not before..
    undersample_model = undersample_pipeline.fit(undersample_Xtrain[train], undersample_ytrain[train])
    undersample_prediction = undersample_model.predict(undersample_Xtrain[test])
    
    undersample_accuracy.append(undersample_pipeline.score(original_Xtrain[test], original_ytrain[test]))
    undersample_precision.append(precision_score(original_ytrain[test], undersample_prediction))
    undersample_recall.append(recall_score(original_ytrain[test], undersample_prediction))
    undersample_f1.append(f1_score(original_ytrain[test], undersample_prediction))
    undersample_auc.append(roc_auc_score(original_ytrain[test], undersample_prediction))
    
def graph_roc_curve_multiple(log_fpr, log_tpr, knear_fpr, knear_tpr, tree_fpr, tree_tpr):
        plt.figure(figsize=(16,8))
        plt.title('ROC kriva \n za 3 primjenjena klasifikatora', fontsize=18)
        plt.plot(log_fpr, log_tpr, label='Rezultat primjene logističke regresije: {:.4f}'.format(roc_auc_score(y_train, log_reg_pred)))
        plt.plot(knear_fpr, knear_tpr, label='Rezultat primjene KNN algoritma: {:.4f}'.format(roc_auc_score(y_train, knears_pred)))
        plt.plot(tree_fpr, tree_tpr, label='Rezultat primjene stabla odluke: {:.4f}'.format(roc_auc_score(y_train, tree_pred)))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.axis([-0.01, 1, 0, 1])
        plt.xlabel('Stopa lažno pozitivnih rezultata', fontsize=16)
        plt.ylabel('Stopa stvarno pozitivnih rezultata', fontsize=16)
        plt.annotate('Minimalni ROC rezultat 50% \n (Najmanji postignuti rezultat)', xy=(0.5, 0.5), xytext=(0.6, 0.3),
                    arrowprops=dict(facecolor='#6E726D', shrink=0.05),
                    )
        plt.legend()
        
from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_predict
# Kreiranje DataFrame-a sa svim rezultatima i imenima klasifikatora

log_reg_pred = cross_val_predict(log_reg, X_train, y_train, cv=5,
                             method="decision_function")

knears_pred = cross_val_predict(knears_neighbors, X_train, y_train, cv=5)

tree_pred = cross_val_predict(tree_clf, X_train, y_train, cv=5)

log_fpr, log_tpr, log_thresold = roc_curve(y_train, log_reg_pred)
knear_fpr, knear_tpr, knear_threshold = roc_curve(y_train, knears_pred)
tree_fpr, tree_tpr, tree_threshold = roc_curve(y_train, tree_pred)


def graph_roc_curve_multiple(log_fpr, log_tpr, knear_fpr, knear_tpr, tree_fpr, tree_tpr):
    plt.figure(figsize=(16,8))
    plt.title('ROC kriva \n za 3 primjenjena klasifikatora', fontsize=18)
    plt.plot(log_fpr, log_tpr, label='Rezultat primjene logističke regresije: {:.4f}'.format(roc_auc_score(y_train, log_reg_pred)))
    plt.plot(knear_fpr, knear_tpr, label='Rezultat primjene KNN algoritma: {:.4f}'.format(roc_auc_score(y_train, knears_pred)))
    plt.plot(tree_fpr, tree_tpr, label='Rezultat primjene stabla odluke: {:.4f}'.format(roc_auc_score(y_train, tree_pred)))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([-0.01, 1, 0, 1])
    plt.xlabel('Stopa lažno pozitivnih rezultata', fontsize=16)
    plt.ylabel('Stopa stvarno pozitivnih rezultata', fontsize=16)
    plt.annotate('Minimalni ROC rezultat 50% \n (Najmanji postignuti rezultat)', xy=(0.5, 0.5), xytext=(0.6, 0.3),
                arrowprops=dict(facecolor='#6E726D', shrink=0.05),
                )
    plt.legend()
    
graph_roc_curve_multiple(log_fpr, log_tpr, knear_fpr, knear_tpr, tree_fpr, tree_tpr)

#AUC za svaki model
auc_log_reg = roc_auc_score(y_train, log_reg_pred)
auc_knears = roc_auc_score(y_train, knears_pred)
auc_tree = roc_auc_score(y_train, tree_pred)

# Ispis rezultata
print(f'AUC za logističku regresiju: {auc_log_reg:.4f}')
print(f'AUC za KNN algoritam: {auc_knears:.4f}')
print(f'AUC za stablo odluke: {auc_tree:.4f}')

def logistic_roc_curve(log_fpr, log_tpr):
    plt.figure(figsize=(12,8))
    plt.title('ROC kriva za logističku regresiju', fontsize=16)
    plt.plot(log_fpr, log_tpr, 'b-', linewidth=2)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('Stopa lažno pozitivnih rezultata', fontsize=16)
    plt.ylabel('Stopa stvarno pozitivnih rezultata', fontsize=16)
    plt.axis([-0.01,1,0,1])
    
    
logistic_roc_curve(log_fpr, log_tpr)
from sklearn.metrics import precision_recall_curve

precision, recall, threshold = precision_recall_curve(y_train, log_reg_pred)

from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
y_pred = log_reg.predict(X_train)

# Prenaučenost
print('---' * 45)
print('Overfitting: \n')
print('Odziv: {:.2f}'.format(recall_score(y_train, y_pred)))
print('Preciznost: {:.2f}'.format(precision_score(y_train, y_pred)))
print('F1: {:.2f}'.format(f1_score(y_train, y_pred)))
print('Tačnost: {:.2f}'.format(accuracy_score(y_train, y_pred)))
print('---' * 45)

# Kako bi trebalo izgledati
print('---' * 45)
print('How it should be:\n')
print("Tačnost: {:.2f}".format(np.mean(undersample_accuracy)))
print("Preciznost: {:.2f}".format(np.mean(undersample_precision)))
print("Odziv: {:.2f}".format(np.mean(undersample_recall)))
print("F1 Score: {:.2f}".format(np.mean(undersample_f1)))
print('---' * 45)

undersample_y_score = log_reg.decision_function(original_Xtest)
from sklearn.metrics import average_precision_score

undersample_average_precision = average_precision_score(original_ytest, undersample_y_score)

print('Srednja vrijednost odnosa preciznost-odziv: {0:0.2f}'.format(
      undersample_average_precision))

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12,6))

precision, recall, _ = precision_recall_curve(original_ytest, undersample_y_score)

plt.step(recall, precision, color='#004a93', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='#48a6ff')
plt.scatter(1.0, 1.0, color='red', marker='o', label='Idealna tačka (1,1)')
plt.text(1.0, 1.0, ' (1,1)', color='red', verticalalignment='bottom', horizontalalignment='right')

plt.scatter(recall, precision, marker='o', color='red',s=5, label='Prag odlučivanja')
plt.xlabel('Odziv')
plt.ylabel('Preciznost')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Preciznost-Odziv kriva za logističku regresiju: \n Srednja vrijednost Precision-Recall rezultata ={0:0.2f}'.format(
          undersample_average_precision), fontsize=16)

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, RandomizedSearchCV


print('Dužina X (trening): {} | Dužina y (trening): {}'.format(len(original_Xtrain), len(original_ytrain)))
print('Dužina X (test): {} | Dužina y (test): {}'.format(len(original_Xtest), len(original_ytest)))


accuracy_lst = []
precision_lst = []
recall_lst = []
f1_lst = []
auc_lst = []

log_reg_sm = LogisticRegression()
knn_sm=KNeighborsClassifier()
dec_tree_sm=DecisionTreeClassifier()

rand_log_reg = RandomizedSearchCV(LogisticRegression(), log_reg_params, n_iter=4)

#smote za logreg
log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
for train, test in sss.split(original_Xtrain, original_ytrain):
    pipeline = imbalanced_make_pipeline(SMOTE(sampling_strategy='minority'), rand_log_reg) # SMOTE happens during Cross Validation not before..
    model = pipeline.fit(original_Xtrain[train], original_ytrain[train])
    best_est = rand_log_reg.best_estimator_
    prediction = best_est.predict(original_Xtrain[test])
    
    accuracy_lst.append(pipeline.score(original_Xtrain[test], original_ytrain[test]))
    precision_lst.append(precision_score(original_ytrain[test], prediction))
    recall_lst.append(recall_score(original_ytrain[test], prediction))
    f1_lst.append(f1_score(original_ytrain[test], prediction))
    auc_lst.append(roc_auc_score(original_ytrain[test], prediction))

print('---' * 45)
print('Logistička regresija sa SMOTE')
print("Tačnost: {}".format(np.mean(accuracy_lst)))
print("Preciznost: {}".format(np.mean(precision_lst)))
print("Odziv: {}".format(np.mean(recall_lst)))
print("F1: {}".format(np.mean(f1_lst)))
print('---' * 45)




labels = ['Ne prevara', 'Prevara']
smote_prediction = best_est.predict(original_Xtest)
print(classification_report(original_ytest, smote_prediction, target_names=labels))
y_score = best_est.decision_function(original_Xtest)

fig = plt.figure(figsize=(12,6))

precision, recall, _ = precision_recall_curve(original_ytest, y_score)

plt.step(recall, precision, color='r', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='#F59B00')
average_precision = average_precision_score(original_ytest, y_score)

print('Srednja vrijednost odnosa preciznost-odziv: {0:0.2f}'.format(
      average_precision))
original_ytest_value_counts = pd.Series(original_ytest).value_counts()
print(original_ytest_value_counts)
plt.scatter(1.0, 1.0, color='red', marker='o', label='Idealna tačka (1,1)')
plt.text(1.0, 1.0, ' (1,1)', color='red', verticalalignment='bottom', horizontalalignment='right')

plt.scatter(recall, precision, marker='o', color='red',s=5, label='Prag odlučivanja')

plt.xlabel('Odziv')
plt.ylabel('Preciznost')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Preciznost-Odziv kriva nakon preuzorkovanja: \n Srednja vrijednost Precision-Recall rezultata ={0:0.2f}'.format(
          average_precision), fontsize=16)

# SMOTE nakon splittinga i kros validacije
sm = SMOTE(sampling_strategy='minority', random_state=42)

Xsm_train, ysm_train = sm.fit_resample(original_Xtrain, original_ytrain)



# Logistic Regression
t0 = time.time()
log_reg_sm = grid_log_reg.best_estimator_
log_reg_sm.fit(Xsm_train, ysm_train)
t1 = time.time()
#print("Fitting oversample data took :{} sec".format(t1 - t0))

from sklearn.metrics import confusion_matrix

# Logistic Regression koristenjem SMOTE tehnike
y_pred_log_reg = log_reg_sm.predict(X_test)

# Ostali modeli koristenjem poduzorkovanja
y_pred_knear = knears_neighbors.predict(X_test)

y_pred_tree = tree_clf.predict(X_test)


log_reg_cf = confusion_matrix(y_test, y_pred_log_reg)
kneighbors_cf = confusion_matrix(y_test, y_pred_knear)
tree_cf = confusion_matrix(y_test, y_pred_tree)

from sklearn.metrics import confusion_matrix


y_pred_log_reg = log_reg_sm.predict(X_test)
y_pred_knear = knears_neighbors.predict(X_test)
y_pred_tree = tree_clf.predict(X_test)


log_reg_cf = confusion_matrix(y_test, y_pred_log_reg)
kneighbors_cf = confusion_matrix(y_test, y_pred_knear)
tree_cf = confusion_matrix(y_test, y_pred_tree)

fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# Postavite aspect parametar za odgovarajuće proporcije
sns.heatmap(log_reg_cf, ax=ax[0], annot=True, cmap=plt.cm.copper, fmt='g', cbar=False, annot_kws={"size": 16}, square=True)
ax[0].set_title("Logistička regresija \n Matrica zabune", fontsize=14)
ax[0].set_xticklabels(['Legalna transakcija', 'Prevara'], fontsize=12)
ax[0].set_yticklabels(['Legalna transakcija', 'Prevara'], fontsize=12)

sns.heatmap(kneighbors_cf, ax=ax[1], annot=True, cmap=plt.cm.copper, fmt='g', cbar=False, annot_kws={"size": 16}, square=True)
ax[1].set_title("KNN metod \n Matrica zabune", fontsize=14)
ax[1].set_xticklabels(['Legalna transakcija', 'Prevara'], fontsize=12)
ax[1].set_yticklabels(['Legalna transakcija', 'Prevara'], fontsize=12)

sns.heatmap(tree_cf, ax=ax[2], annot=True, cmap=plt.cm.copper, fmt='g', cbar=False, annot_kws={"size": 16}, square=True)
ax[2].set_title("Stablo odluke \n Matrica zabune", fontsize=14)
ax[2].set_xticklabels(['Legalna transakcija', 'Prevara'], fontsize=12)
ax[2].set_yticklabels(['Legalna transakcija', 'Prevara'], fontsize=12)

plt.tight_layout()  # Osigurava da se podaci ne sijeku
plt.show()

from sklearn.metrics import classification_report

print('Logistička regresija:')
print(classification_report(y_test, y_pred_log_reg))

print('KNN:')
print(classification_report(y_test, y_pred_knear))

print('Stablo odluke:')
print(classification_report(y_test, y_pred_tree))

# Konačni rezultat na testnom skupu podataka za logističku regresiju
from sklearn.metrics import accuracy_score

# Logistička regresija sa poduzorkovanjem
y_pred = log_reg.predict(X_test)
undersample_score_log = accuracy_score(y_test, y_pred)



# Logistička regresija sa SMOTE tehnikom
y_pred_sm = best_est.predict(original_Xtest)
oversample_score_log = accuracy_score(original_ytest, y_pred_sm)



from IPython.display import display


d = {'Tehnika': ['Random poduzorkovanje', 'Preuzorkovanje (SMOTE)'], 'Rezultat': [undersample_score_log, oversample_score_log]}
final_df = pd.DataFrame(data=d)



score = final_df['Rezultat']
final_df.drop('Rezultat', axis=1, inplace=True)
final_df.insert(1, 'Rezultat', score)


display(final_df)

# Odaberemo indeks jedne transakcije iz testnog skupa
index_to_check = 3 

# Dobivanje pojedinačne transakcije i stvarnih oznaka iz testnog skupa
transaction_to_check = X_test[index_to_check].reshape(1, -1)

actual_label = y_test[index_to_check]

# Predikcija pomoću KNN klasifikatora
prediction = knears_neighbors.predict(transaction_to_check)

# Ispis detalja o transakciji i njenom predikcijom
print("Detalji transakcije:")
print(transaction_to_check)
print("\nStvarna oznaka:", actual_label)
print("Predikcija:", prediction[0])

# Pronalaženje indeksa najbližih susjeda iz trening skupa
_, indices = knears_neighbors.kneighbors(transaction_to_check)

# Ispisivanje najbližih susjeda iz trening skupa
print("Vrijednost atributa 'Class' najbližih susjeda iz trening skupa:")
for index in indices[0]:
    print(y_train[index])


coefficients = pd.DataFrame(log_reg.coef_.flatten(), index=X.columns, columns=['Coefficient'])
coefficients = coefficients.sort_values(by='Coefficient', ascending=False)
print("Važnost atributa:\n", coefficients)



decision_function_values = log_reg.decision_function(X_test)
plt.figure(figsize=(10, 6))
plt.scatter(range(len(X_test)), decision_function_values, c=y_test, cmap='viridis')
plt.colorbar(label='Klasa')
plt.title('Vrijednosti odlučujuće funkcije za test skup podataka')
plt.xlabel('Instanca')
plt.ylabel('Vrijednost odlučujuće funkcije')
plt.show()

from mlxtend.plotting import plot_decision_regions

# Prikazuje samo prva dva atributa
X_two_features = X_train[:, :2]
log_reg.fit(X_two_features, y_train)

plt.figure(figsize=(10, 6))
plot_decision_regions(X_two_features, y_train, clf=log_reg, legend=2)
plt.title('Granica odlučivanja logističke regresije za prva dva atributa')
plt.xlabel('Prvi atribut')
plt.ylabel('Drugi atribut')
plt.show()

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Vizualiziraj stablo odluke
plt.figure(figsize=(20, 10))
plot_tree(tree_clf, filled=True, feature_names=list(X.columns), class_names=['0', '1'])
plt.show()

# Prolazak kroz prvih 3 transakcije iz testnog skupa
for index, transaction_to_check in enumerate(X_test[:3]):
    # Ispis indeksa transakcije
    print(f"\nInformacije o transakciji s indeksom {index}:")

    # Ispis pojedinačnih informacija o transakciji
    for i, column_value in enumerate(transaction_to_check):
        column_name = f"V{i + 1}" if i < 28 else "Amount" if i == 28 else "Time"
        print(f"{column_name}: {column_value}")

    # Prati put transakcije kroz stablo odluke
    path = tree_clf.decision_path(transaction_to_check.reshape(1, -1))

    # Ispisuj uslove čvorova i rezultate na putu
    node_indicator = tree_clf.decision_path(transaction_to_check.reshape(1, -1)).toarray()

    for node_index, node_value in enumerate(node_indicator[0]):
        if node_value == 1:  # Ako je transakcija prošla kroz čvor
            if tree_clf.tree_.feature[node_index] != -2:  # Ako čvor nije list
                feature_index = tree_clf.tree_.feature[node_index]
                threshold = tree_clf.tree_.threshold[node_index]

                print(f"\nČvor {node_index}: Ako vrijednost na indeksu {feature_index} <= {threshold}")

    # Predviđanje klase za transakciju
    predicted_class = tree_clf.predict(transaction_to_check.reshape(1, -1))
    print("\nPredviđena klasa:", predicted_class[0])

    actual_class = y_test[index]
    print("Stvarna klasa:", actual_class)
    
