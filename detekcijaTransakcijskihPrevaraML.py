# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np 
import pandas as pd 
#import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings
from imblearn.under_sampling import NearMiss
from collections import Counter
from imblearn.pipeline import make_pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


warnings.filterwarnings("ignore")

ccd = pd.read_csv('creditcard.csv');
"print(ccd.head());"
"print(ccd.tail());"

"print(ccd.info())"
"print(ccd.isnull().sum ())"


plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=ccd, palette='Set2')
plt.title('Distribucija klasa prije obrade', fontsize=14)
plt.xlabel('Klasa')
plt.ylabel('Brojnost')

#
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
print(ccd['scaled_amount'].head(5))


print("Vrijednosti atributa 'Time' nakon skaliranja")
print(ccd['scaled_time'].tail(5))

from sklearn.model_selection import train_test_split
# train_test_split se koristi za podjelu podataka na skupove za treniranje i testiranje
from sklearn.model_selection import StratifiedShuffleSplit
# StratifiedShuffleSplit koristi za k-stratifikaciju, odnosno podjelu podataka na k podskupova, očuvajući omjer klasa

print('Legalne transakcije čine ', round(ccd['Class'].value_counts()[0]/len(ccd) * 100,2), '% skupa podataka')
print('Prevare čine ', round(ccd['Class'].value_counts()[1]/len(ccd) * 100,2), '% skupa podataka')


X = ccd.drop('Class', axis=1)
# uklanja stupac "Class" iz skupa podataka i rezultat je DataFrame koji sadrži samo značajke koje će se koristiti za treniranje modela.
y = ccd['Class']

sss = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)

# Ovaj dio koda izvodi postupak kros-validacije i razdvaja podatke na trening i test skupove.
for train_index, test_index in sss.split(X, y):
    print("Train:", train_index, "Test:", test_index)
    
    original_Xtrain, original_Xtest = X.iloc[train_index], X.iloc[test_index]
    original_ytrain, original_ytest = y.iloc[train_index], y.iloc[test_index]
# Slično kao i za ulazne podatke, ovdje se odabiru odgovarajući ciljni atributi (u ovom slučaju, stupac "Class") za trening i test skupove.

original_Xtrain = original_Xtrain.values
original_Xtest = original_Xtest.values
original_ytrain = original_ytrain.values
original_ytest = original_ytest.values 
# Pretvaraju se odabrani podaci iz DataFrame oblika u oblik pogodan za treniranje i testiranje modela, tj. u oblik NumPy nizova.

train_unique_label, train_counts_label = np.unique(original_ytrain, return_counts=True)
test_unique_label, test_counts_label = np.unique(original_ytest, return_counts=True)
#  Ove linije koda računaju broj jedinstvenih vrijednosti i broj ponavljanja tih vrijednosti u ciljnom atributu "Class" za trening i test skupove.

print('-' * 100)

print('Label Distributions: \n')
print(train_counts_label/ len(original_ytrain))
print(test_counts_label/ len(original_ytest))

#RandomUnderSampling

from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)
new_ccd = pd.DataFrame(X_resampled, columns=X.columns)
new_ccd['Class'] = y_resampled

print(new_ccd.head())

print('Distribucija klasa unutar dobijenog podskupa podataka')
print(new_ccd['Class'].value_counts()/len(new_ccd))

print('----' * 63)
plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=new_ccd, palette='Set2')
plt.title('Distribucija klasa nakon obrade', fontsize=14)
plt.xlabel('Klasa')
plt.ylabel('Brojnost')

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


# # -----> V14 Removing Outliers (Highest Negative Correlated with Labels)
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

from imblearn.under_sampling import RandomUnderSampler
X1 = new_ccd.drop('Class', axis=1)
# uklanja stupac "Class" iz skupa podataka i rezultat je DataFrame koji sadrži samo značajke koje će se koristiti za treniranje modela.
y1 = new_ccd['Class']
rus = RandomUnderSampler(sampling_strategy='auto', random_state=42)
X_resampled1, y_resampled1 = rus.fit_resample(X1, y1)
new_ccd = pd.DataFrame(X_resampled1, columns=X1.columns)
new_ccd['Class'] = y_resampled1


plt.figure(figsize=(6, 4))
sns.countplot(x='Class', data=new_ccd, palette='Set2')
plt.title('Distribucija klasa nakon obrade', fontsize=14)
plt.xlabel('Klasa')
plt.ylabel('Brojnost')

total_instances_afterRUS = len(new_ccd)
print("Ukupan broj instanci nakon uklanjanja outlier-a  i undersamplinga:", total_instances_afterRUS)

# Prikazuje ukupan broj prevara prije uklanjanja outlier-a
total_frauds_afterRUS = len(new_ccd[new_ccd['Class'] == 1])
print("Ukupan broj prevara nakon uklanjanja outlier-a  i undersamplinga:", total_frauds_afterRUS)

total_leg_afterRUS = len(new_ccd[new_ccd['Class'] == 0])
print("Ukupan broj legalnih transakcija nakon uklanjanja outlier-a  i undersamplinga:", total_leg_afterRUS)

#Skalirane podatke dijelimo na trening i test skupove
from sklearn.model_selection import train_test_split

#Eksplicitno se koristi za poduzorkovanje
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values


classifiers = {
    "Logistička regresija": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "Stablo odluke": DecisionTreeClassifier()
}

from sklearn.model_selection import cross_val_score


for key, classifier in classifiers.items():
    classifier.fit(X_train, y_train)
    training_score = cross_val_score(classifier, X_train, y_train, cv=5)
    print("Klasifikator: ", classifier.__class__.__name__, "ima rezultat treniranja od: ", round(training_score.mean(), 2) * 100, "% preciznosti")


#Korištenje GridSearchCV za pronalazak najboljih parametara
from sklearn.model_selection import GridSearchCV


# Logistička regresija
log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'max_iter': [100, 500, 1000]}



grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)
grid_log_reg.fit(X_train, y_train)

log_reg = grid_log_reg.best_estimator_
knears_params = {"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)
grid_knears.fit(X_train, y_train)
# KNN klasifikator
knears_neighbors = grid_knears.best_estimator_
#Stablo odluke klasifikator
tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), 
              "min_samples_leaf": list(range(5,7,1))}
grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)
grid_tree.fit(X_train, y_train)


tree_clf = grid_tree.best_estimator_

#Slučaj prenaučenosti modela

print("Rezultati dobijeni nakon križne validacije su sljedeći: ")
log_reg_score = cross_val_score(log_reg, X_train, y_train, cv=5)
print('Klasifikator LogisticRegression ima rezultat treniranja od: ', round(log_reg_score.mean() * 100, 2).astype(str) + '% preciznosti')

knears_score = cross_val_score(knears_neighbors, X_train, y_train, cv=5)
print('Klasifikator KNN ima rezultat treniranja', round(knears_score.mean() * 100, 2).astype(str) + '% preciznosti')

tree_score = cross_val_score(tree_clf, X_train, y_train, cv=5)
print('Klasifikator Stablo odluke ima rezultat treniranja', round(tree_score.mean() * 100, 2).astype(str) + '% preciznosti')

#Poduzorkovanje tokom kros validacije
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
print('NearMiss Label Distribution: {}'.format(Counter(y_nearmiss)))
# Cross Validating the right way

for train, test in sss.split(undersample_Xtrain, undersample_ytrain):
    undersample_pipeline = make_pipeline(NearMiss(sampling_strategy='majority'), log_reg) # SMOTE happens during Cross Validation not before..
    undersample_model = undersample_pipeline.fit(undersample_Xtrain[train], undersample_ytrain[train])
    undersample_prediction = undersample_model.predict(undersample_Xtrain[test])
    
    undersample_accuracy.append(undersample_pipeline.score(original_Xtrain[test], original_ytrain[test]))
    undersample_precision.append(precision_score(original_ytrain[test], undersample_prediction))
    undersample_recall.append(recall_score(original_ytrain[test], undersample_prediction))
    undersample_f1.append(f1_score(original_ytrain[test], undersample_prediction))
    undersample_auc.append(roc_auc_score(original_ytrain[test], undersample_prediction))



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

from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
y_pred = log_reg.predict(X_train) #predvidjene izlazne vrijednosti

#Slučaj overfittinga
print('---' * 45)
print('Slučaj prenaučenosti modela: \n')
print('Tačnost: {:.2f}'.format(accuracy_score(y_train, y_pred)))
print('Preciznost: {:.2f}'.format(precision_score(y_train, y_pred)))
print('Recall rezultat: {:.2f}'.format(recall_score(y_train, y_pred)))
print('F1 rezultat: {:.2f}'.format(f1_score(y_train, y_pred)))
print('---' * 30)

#Kako bi trebalo izgledati
print('---' * 45)
print('Kako bi trebalo biti:\n')
print("Tačnost: {:.2f}".format(np.mean(undersample_accuracy)))
print("Preciznost: {:.2f}".format(np.mean(undersample_precision)))
print("Recall rezultat: {:.2f}".format(np.mean(undersample_recall)))
print("F1 rezultat: {:.2f}".format(np.mean(undersample_f1)))
print('---' * 30)

undersample_y_score = log_reg.decision_function(original_Xtest)
from sklearn.metrics import average_precision_score

undersample_average_precision = average_precision_score(original_ytest, undersample_y_score)

print('Srednja vrijednost precision-recall rezultata: {0:0.2f}'.format(
      undersample_average_precision))

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12,6))

precision, recall, _ = precision_recall_curve(original_ytest, undersample_y_score)

plt.step(recall, precision, color='#004a93', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='#48a6ff')

plt.xlabel('Recall')
plt.ylabel('Preciznost')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall kriva nakon poduzorkovanja: \n Srednja vrijednost Precision-Recall rezultata ={0:0.2f}'.format(
          undersample_average_precision), fontsize=16)

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, RandomizedSearchCV

print('Veličina ulaznih trening podataka: {} | Veličina izlaznih trening podataka: {}'
      .format(len(original_Xtrain), len(original_ytrain)))
print('Veličina ulaznih test podataka: {} | Veličina izlaznih test podataka: {}'
      .format(len(original_Xtest), len(original_ytest)))

#Kreiraju se prazne liste koje će se koristiti za pohranu rezultata klasifikacije
accuracy_lst = []
precision_lst = []
recall_lst = []
f1_lst = []
auc_lst = []

#Definiranje i podešavanje logističke regresije kao klasifikatora i korištenje RandomizedSearchCV za traženje optimalnih parametara 
log_reg_sm = LogisticRegression()
rand_log_reg = RandomizedSearchCV(LogisticRegression(), log_reg_params, n_iter=4)

# Implementacija SMOTE tehnike koja se primjenjuje tokom kros validacije
log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
for train, test in sss.split(original_Xtrain, original_ytrain):
    pipeline = make_pipeline(SMOTE(sampling_strategy='minority'), rand_log_reg) # SMOTE tokom Cross Validation a ne prije
    model = pipeline.fit(original_Xtrain[train], original_ytrain[train])
    best_est = rand_log_reg.best_estimator_
    prediction = best_est.predict(original_Xtrain[test])
    accuracy_lst.append(pipeline.score(original_Xtrain[test], original_ytrain[test]))
    precision_lst.append(precision_score(original_ytrain[test], prediction))
    recall_lst.append(recall_score(original_ytrain[test], prediction))
    f1_lst.append(f1_score(original_ytrain[test], prediction))
    auc_lst.append(roc_auc_score(original_ytrain[test], prediction))
    
print('---' * 25)
print("tačnost: {}".format(np.mean(accuracy_lst)))
print("preciznost {}".format(np.mean(precision_lst)))
print("recall: {}".format(np.mean(recall_lst)))
print("f1: {}".format(np.mean(f1_lst)))
print('---' * 25)

from sklearn.metrics import classification_report


labels = ['No Fraud', 'Fraud']
smote_prediction = best_est.predict(original_Xtest)
print(classification_report(original_ytest, smote_prediction, target_names=labels))

y_score = best_est.decision_function(original_Xtest)

average_precision = average_precision_score(original_ytest, y_score)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))

fig = plt.figure(figsize=(12,6))

precision, recall, _ = precision_recall_curve(original_ytest, y_score)

plt.step(recall, precision, color='r', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='#48a6ff')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall kriva nakon preduzorkovanja: \n Srednja vrijednost Precision-Recall rezultata ={0:0.2f}'.format(
          average_precision), fontsize=16)

# SMOTE nakon splittinga i kros validacije
sm = SMOTE(sampling_strategy='minority', random_state=42)
# Xsm_train, ysm_train = sm.fit_sample(X_train, y_train)



Xsm_train, ysm_train = sm.fit_resample(original_Xtrain, original_ytrain)

# Rezultat je popravljen za 2%
# Implementacija GridSearchCV i ostalih evaluacijskih modela

# Logistička regresija
t0 = time.time()
log_reg_sm = grid_log_reg.best_estimator_
log_reg_sm.fit(Xsm_train, ysm_train)
t1 = time.time()
print("Vrijeme potrebno za treniranje modela logističke regresije na prethodno generisanom oversample skupu podataka :{} sec".format(t1 - t0))

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
undersample_score = accuracy_score(y_test, y_pred)



# Logistička regresija sa SMOTE tehnikom
y_pred_sm = best_est.predict(original_Xtest)
oversample_score = accuracy_score(original_ytest, y_pred_sm)
from IPython.display import display


d = {'Tehnika': ['Random poduzorkovanje', 'Preduzorkovanje (SMOTE)'], 'Rezultat': [undersample_score, oversample_score]}
final_df = pd.DataFrame(data=d)


score = final_df['Rezultat']
final_df.drop('Rezultat', axis=1, inplace=True)
final_df.insert(1, 'Rezultat', score)


display(final_df)