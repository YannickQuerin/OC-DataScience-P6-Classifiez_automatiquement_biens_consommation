

# Chargement des librairies
import datetime
import time
import sys
import os
import string
import json
import disarray
from datetime import datetime
from termcolor import colored, cprint
#import jyquickhelper
import numpy as np
import pandas as pd
import fonctions_data
import matplotlib as mpl
import matplotlib.pyplot as plt
# Feature extraction
import seaborn as sns

# BERT
#import transformers
from sentence_transformers import SentenceTransformer, models

# USE
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.manifold import TSNE

# Algorithmes de classification supervisée
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier

import sklearn
from sklearn.metrics import adjusted_rand_score, rand_score, accuracy_score, silhouette_score, accuracy_score, confusion_matrix, \
homogeneity_score, completeness_score, v_measure_score, precision_score, recall_score, f1_score
from sklearn import decomposition, preprocessing
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, \
    HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Traitement de text
import texthero as hero
from texthero import preprocessing
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, FreqDist
from stop_words import get_stop_words
from wordcloud import WordCloud
from gensim.models import Word2Vec
import multiprocessing

# Réduction dimension
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE

# Clustering
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report

# Warnings
import warnings
warnings.filterwarnings('ignore')


# Versions
print('Version des librairies utilisées :')
#print('jyquickhelper         : ' + jyquickhelper.__version__)
print('Python                : ' + sys.version)
print('NumPy                 : ' + np.version.full_version)
print('Pandas                : ' + pd.__version__)
print('Matplotlib            : ' + mpl.__version__)
print('Seaborn               : ' + sns.__version__)
print('Sklearn               : ' + sklearn.__version__)
print('Outils dataframe      : ' + fonctions_data.__version__)
now = datetime.now().isoformat()
print('Lancé le           : ' + now)

#pip install torch --upgrade



seed = 27 

#data = pd.read_csv('data_clean.csv')
data = 

# Taille : nombre de lignes/colonnes
print(f'Le jeu de données contient {data.shape[0]} lignes et {data.shape[1]} variables.')

data.drop(columns='Unnamed: 0', inplace =True)
data.head(5)




models_performance = {}
#models_performance = []


def metrics_report(model_name, test_labels, predictions, performances):
    """
    Compute performance metrics of a model and store them in a dictionary
    
    Args:
        model_name(string): name of the evaluated model
        test_labels(array): labels related to predictors
        preductions(array): predicted results
        performances(dict): used dictionary to store metrics
    Returns:
        performances(dict): used dictionary to store metrics filed with models ones
    """    
    accuracy = accuracy_score(test_labels, predictions)
    macro_precision = precision_score(test_labels, predictions, average='macro')
    macro_recall = recall_score(test_labels, predictions, average='macro')
    macro_f1 = f1_score(test_labels, predictions, average='macro')
    micro_precision = precision_score(test_labels, predictions, average='micro')
    micro_recall = recall_score(test_labels, predictions, average='micro')
    micro_f1 = f1_score(test_labels, predictions, average='micro')
    print("------" + model_name + " Model Metrics-----")
    print("Accuracy: {:.4f}\nPrecision:\n  - Macro: {:.4f}\n  - Micro: {:.4f}\nRecall:\n  - Macro: {:.4f}\n  - Micro: {:.4f}\nF1-measure:\n  - Macro: {:.4f}\n  - Micro: {:.4f}"\
          .format(accuracy, macro_precision, micro_precision, macro_recall, micro_recall, macro_f1, micro_f1))
    
    performances[model_name] = {}
    performances[model_name]["micro_precision"] =  micro_precision
    performances[model_name]["micro_recall"] = micro_recall
    performances[model_name]["micro_f1"] = micro_f1
    
    return performances





# Variable dont on veut extraire les features
data['description'] = \
    data['description'].fillna('').astype(str)
data_desc_orig = data['description']

type_donnee = 'TSNE_ROBERTA_DESC_ORIG_LEM'

cols = ['VAR1', 'VAR2']

# creation d'un variable numerique pour la categorie

le = LabelEncoder()
le.fit(data["category"])
data["category_label"] = le.transform(data["category"])

# Instanciation du modèle
bert_desc_orig = SentenceTransformer('roberta-large-nli-mean-tokens')

# Encodage
vector_roberta_desc_orig = \
    bert_desc_orig.encode(data_desc_orig, show_progress_bar=False)

# Réduction de dimension tsne
tsne_roberta_desc_orig = TSNE(verbose=1, perplexity=50, n_iter=5000)
X_proj_tsne_roberta_desc_orig = \
    tsne_roberta_desc_orig.fit_transform(vector_roberta_desc_orig)
# Dataframe pour clustering
df_tsne_roberta_desc_orig = pd.DataFrame({'VAR1' : X_proj_tsne_roberta_desc_orig[:, 0],
                                          'VAR2' : X_proj_tsne_roberta_desc_orig[:, 1],
                                          'CATEGORIE' : data['category']})

Xtrain, Xtest, ytrain, ytest = train_test_split(df_tsne_roberta_desc_orig[cols], data.category_label, test_size=0.15, stratify=data.category_label, random_state=1944)

knn_clf_sup = KNeighborsClassifier()
knn_clf_sup.fit(Xtrain, ytrain)
knn_predictions = knn_clf_sup.predict(Xtrain)
metrics_report("knn", ytrain, knn_predictions, models_performance)

svm_clf_sup = OneVsRestClassifier(LinearSVC(), n_jobs=-1)
svm_clf_sup.fit(Xtrain, ytrain)
svm_predictions = svm_clf_sup.predict(Xtrain)
metrics_report("svm", ytrain, svm_predictions, models_performance)

# rf_clf_sup = RandomForestClassifier(n_jobs=-1)
# rf_clf_sup.fit(Xtrain, ytrain)
# rf_predictions = rf_clf_sup.predict(Xtrain)
# metrics_report("Random Forest", ytrain, rf_predictions, models_performance)

gb_clf_sup = OneVsRestClassifier(GradientBoostingClassifier())
gb_clf_sup.fit(Xtrain, ytrain)
gb_predictions = gb_clf_sup.predict(Xtrain)
metrics_report("Gradient Boosting", ytrain, gb_predictions, models_performance)

gnb_clf_sup = OneVsRestClassifier(GaussianNB())
gnb_clf_sup.fit(Xtrain, ytrain)
gnb_predictions = gnb_clf_sup.predict(Xtrain)
metrics_report("Naive Bayes", ytrain, gnb_predictions, models_performance)    

etc_clf_sup = ExtraTreesClassifier(random_state=0)
etc_clf_sup.fit(Xtrain, ytrain)
etc_predictions = etc_clf_sup.predict(Xtrain)
metrics_report("extra tree classifier", ytrain, etc_predictions, models_performance)

result_df = pd.DataFrame.from_dict(models_performance, orient="index")
result_df

from sklearn.ensemble import ExtraTreesClassifier

etc_model = ExtraTreesClassifier(random_state=0).fit(Xtrain, ytrain) 
etc_predictions = etc_model.predict(Xtest) 

# model accuracy for X_test   
accuracy_etc = etc_model.score(Xtest, ytest) 
print("Accuracy de l'algorithme etc :", round(accuracy_etc, 3))

def conf_mat_transform(y_true,y_pred) :
    conf_mat = confusion_matrix(y_true,y_pred)
    
    corresp = np.argmax(conf_mat, axis=0)
    print ("Correspondance des clusters : ", corresp)
    labels = pd.Series(y_true, name="y_true").to_frame()
    labels['y_pred'] = y_pred
    labels['y_pred_transform'] = labels['y_pred'].apply(lambda x : corresp[x]) 
    
    return labels['y_pred_transform']

conf_mat = confusion_matrix(le.inverse_transform(ytest), le.inverse_transform(etc_model.predict(Xtest)))
print(conf_mat)
#test_labels = le.inverse_transform(y_test)
labels_transform_train = conf_mat_transform(ytrain, etc_model.predict(Xtrain))
labels_transform_test = conf_mat_transform(ytest, etc_model.predict(Xtest))

accuracy_tsne_roberta_desc_lem_test = accuracy_score(ytest, labels_transform_test)
accuracy_tsne_roberta_desc_lem_train = accuracy_score(ytrain, labels_transform_train)

print()
print("ACCURACY TSNE_ROBERTA_DESC_ORIG_LEM (Test): {:.3}".format(accuracy_tsne_roberta_desc_lem_test))
print("ACCURACY TSNE_ROBERTA_DESC_ORIG_LEM (Train): {:.3}".format(accuracy_tsne_roberta_desc_lem_train))
print()

print()
print(classification_report(ytest, labels_transform_test))

list_labels = ["Furnishing", "Baby", "Watches", "Decor", "Kitchen", \
               "Beauty", "Computers"]

df_cm = pd.DataFrame(conf_mat, index = [label for label in list_labels],
                  columns = [i for i in "0123456"])
plt.figure(figsize = (6,4))
sns.heatmap(df_cm, annot=True, cmap="Blues")
plt.xlabel('numero Cluster')
plt.ylabel('categorie Depart')
plt.title(f'matrice confusion {type_donnee}')
plt.show()



# Variable dont on veut extraire les features
data['desc_lem'] = \
    data['desc_lem'].fillna('').astype(str)
data_desc_lem = data['desc_lem']

type_donnee = 'TSNE_ROBERTA_DESC_LEM'

# Instanciation du modèle
bert_desc_lem = SentenceTransformer('roberta-large-nli-mean-tokens')

# Encodage
vector_roberta_desc_lem = \
    bert_desc_lem.encode(data_desc_lem, show_progress_bar=False)

# Réduction de dimension tsne
tsne_roberta_desc_lem = TSNE(verbose=1, perplexity=50, n_iter=5000)
X_proj_tsne_roberta_desc_lem = \
    tsne_roberta_desc_lem.fit_transform(vector_roberta_desc_lem)
# Dataframe pour clustering
df_tsne_roberta_desc_lem = pd.DataFrame({'VAR1' : X_proj_tsne_roberta_desc_lem[:, 0],
                                           'VAR2' : X_proj_tsne_roberta_desc_lem[:, 1],
                                           'CATEGORIE' : data['category']})

Xtrain, Xtest, ytrain, ytest = train_test_split(df_tsne_roberta_desc_lem[cols], data.category_label, test_size=0.15, stratify=data.category_label, random_state=1944)

knn_clf_sup = KNeighborsClassifier()
knn_clf_sup.fit(Xtrain, ytrain)
knn_predictions = knn_clf_sup.predict(Xtrain)
metrics_report("knn", ytrain, knn_predictions, models_performance)

svm_clf_sup = OneVsRestClassifier(LinearSVC(), n_jobs=-1)
svm_clf_sup.fit(Xtrain, ytrain)
svm_predictions = svm_clf_sup.predict(Xtrain)
metrics_report("svm", ytrain, svm_predictions, models_performance)

rf_clf_sup = RandomForestClassifier(n_jobs=-1)
rf_clf_sup.fit(Xtrain, ytrain)
rf_predictions = rf_clf_sup.predict(Xtrain)
metrics_report("Random Forest", ytrain, rf_predictions, models_performance)

gnb_clf_sup = OneVsRestClassifier(GaussianNB())
gnb_clf_sup.fit(Xtrain, ytrain)
gnb_predictions = gnb_clf_sup.predict(Xtrain)
metrics_report("Naive Bayes", ytrain, gnb_predictions, models_performance)    

gb_clf_sup = OneVsRestClassifier(GradientBoostingClassifier())
gb_clf_sup.fit(Xtrain, ytrain)
gb_predictions = gb_clf_sup.predict(Xtrain)
metrics_report("Gradient Boosting", ytrain, gb_predictions, models_performance)

etc_clf_sup = ExtraTreesClassifier(random_state=0)
etc_clf_sup.fit(Xtrain, ytrain)
etc_predictions = etc_clf_sup.predict(Xtrain)
metrics_report("extra tree classifier", ytrain, etc_predictions, models_performance)

result_df = pd.DataFrame.from_dict(models_performance, orient="index")
result_df

from sklearn.ensemble import ExtraTreesClassifier

etc_model = ExtraTreesClassifier(random_state=0).fit(Xtrain, ytrain) 
etc_predictions = etc_model.predict(Xtest) 

# model accuracy for X_test   
accuracy_etc = etc_model.score(Xtest, ytest) 
print("Accuracy de l'algorithme etc :", round(accuracy_etc, 3))

conf_mat = confusion_matrix(le.inverse_transform(ytest), le.inverse_transform(etc_model.predict(Xtest)))
print(conf_mat)
#test_labels = le.inverse_transform(y_test)
labels_transform_train = conf_mat_transform(ytrain, etc_model.predict(Xtrain))
labels_transform_test = conf_mat_transform(ytest, etc_model.predict(Xtest))

accuracy_tsne_roberta_desc_lem_test = accuracy_score(ytest, labels_transform_test)
accuracy_tsne_roberta_desc_lem_train = accuracy_score(ytrain, labels_transform_train)

print()
print("ACCURACY TSNE_ROBERTA_DESC_LEM (Test): {:.3}".format(accuracy_tsne_roberta_desc_lem_test))
print("ACCURACY TSNE_ROBERTA_DESC_LEM (Train): {:.3}".format(accuracy_tsne_roberta_desc_lem_train))
print()

print()
print(classification_report(ytest, labels_transform_test))

list_labels = ["Furnishing", "Baby", "Watches", "Decor", "Kitchen", \
               "Beauty", "Computers"]

df_cm = pd.DataFrame(conf_mat, index = [label for label in list_labels],
                  columns = [i for i in "0123456"])
plt.figure(figsize = (6,4))
sns.heatmap(df_cm, annot=True, cmap="Blues")
plt.xlabel('numero Cluster')
plt.ylabel('categorie Depart')
plt.title(f'matrice confusion {type_donnee}')
plt.show()



# Variable dont on veut extraire les features
data['desc_stem'] = \
    data['desc_stem'].fillna('').astype(str)
data_desc_stem = data['desc_stem']

type_donnee = 'TSNE_ROBERTA_DESC_STEM'

# Instanciation du modèle
bert_desc_stem = SentenceTransformer('roberta-large-nli-mean-tokens')

# Encodage
vector_roberta_desc_stem = \
    bert_desc_orig.encode(data_desc_stem, show_progress_bar=False)

# Réduction de dimension tsne
tsne_roberta_desc_stem = TSNE(verbose=1, perplexity=50, n_iter=5000)
X_proj_tsne_roberta_desc_stem = \
    tsne_roberta_desc_stem.fit_transform(vector_roberta_desc_stem)
# Dataframe pour clustering
df_tsne_roberta_desc_stem = pd.DataFrame({'VAR1' : X_proj_tsne_roberta_desc_stem[:, 0],
                                           'VAR2' : X_proj_tsne_roberta_desc_stem[:, 1],
                                           'CATEGORIE' : data['category']})

Xtrain, Xtest, ytrain, ytest = train_test_split(df_tsne_roberta_desc_stem[cols], data.category_label, test_size=0.15, stratify=data.category_label, random_state=1944)

knn_clf_sup = KNeighborsClassifier()
knn_clf_sup.fit(Xtrain, ytrain)
knn_predictions = knn_clf_sup.predict(Xtrain)
metrics_report("knn", ytrain, knn_predictions, models_performance)

svm_clf_sup = OneVsRestClassifier(LinearSVC(), n_jobs=-1)
svm_clf_sup.fit(Xtrain, ytrain)
svm_predictions = svm_clf_sup.predict(Xtrain)
metrics_report("svm", ytrain, svm_predictions, models_performance)

rf_clf_sup = RandomForestClassifier(n_jobs=-1)
rf_clf_sup.fit(Xtrain, ytrain)
rf_predictions = rf_clf_sup.predict(Xtrain)
metrics_report("Random Forest", ytrain, rf_predictions, models_performance)

gnb_clf_sup = OneVsRestClassifier(GaussianNB())
gnb_clf_sup.fit(Xtrain, ytrain)
gnb_predictions = gnb_clf_sup.predict(Xtrain)
metrics_report("Naive Bayes", ytrain, gnb_predictions, models_performance)    

gb_clf_sup = OneVsRestClassifier(GradientBoostingClassifier())
gb_clf_sup.fit(Xtrain, ytrain)
gb_predictions = gb_clf_sup.predict(Xtrain)
metrics_report("Gradient Boosting", ytrain, gb_predictions, models_performance)

etc_clf_sup = ExtraTreesClassifier(random_state=0)
etc_clf_sup.fit(Xtrain, ytrain)
etc_predictions = etc_clf_sup.predict(Xtrain)
metrics_report("extra tree classifier", ytrain, etc_predictions, models_performance)

result_df = pd.DataFrame.from_dict(models_performance, orient="index")
result_df

from sklearn.ensemble import ExtraTreesClassifier

etc_model = ExtraTreesClassifier(random_state=0).fit(Xtrain, ytrain) 
etc_predictions = etc_model.predict(Xtest) 

# model accuracy for X_test   
accuracy_etc = etc_model.score(Xtest, ytest) 
print("Accuracy de l'algorithme etc :", round(accuracy_etc, 3))

conf_mat = confusion_matrix(le.inverse_transform(ytest), le.inverse_transform(etc_model.predict(Xtest)))
print(conf_mat)
#test_labels = le.inverse_transform(y_test)
labels_transform_train = conf_mat_transform(ytrain, etc_model.predict(Xtrain))
labels_transform_test = conf_mat_transform(ytest, etc_model.predict(Xtest))

accuracy_tsne_roberta_desc_lem_test = accuracy_score(ytest, labels_transform_test)
accuracy_tsne_roberta_desc_lem_train = accuracy_score(ytrain, labels_transform_train)

print()
print("ACCURACY TSNE_ROBERTA_DESC_STEM (Test): {:.3}".format(accuracy_tsne_roberta_desc_lem_test))
print("ACCURACY TSNE_ROBERTA_DESC_STEM (Train): {:.3}".format(accuracy_tsne_roberta_desc_lem_train))
print()

print()
print(classification_report(ytest, labels_transform_test))

df_cm = pd.DataFrame(conf_mat, index = [label for label in list_labels],
                  columns = [i for i in "0123456"])
plt.figure(figsize = (6,4))
sns.heatmap(df_cm, annot=True, cmap="Blues")
plt.xlabel('numero Cluster')
plt.ylabel('categorie Depart')
plt.title(f'matrice confusion {type_donnee}')
plt.show()



# Variable dont on veut extraire les features
data['product_specifications'] = \
    data['product_specifications'].fillna('').astype(str)
data_prods_orig = data['product_specifications']

type_donnee = 'TSNE_ROBERTA_PRODS_ORIG'

# Instanciation du modèle
bert_prods_orig = SentenceTransformer('roberta-large-nli-mean-tokens')

# Encodage
vector_roberta_prods_orig = \
    bert_prods_orig.encode(data_prods_orig, show_progress_bar=False)

# Réduction de dimension tsne
tsne_roberta_prods_orig = TSNE(verbose=1, perplexity=50, n_iter=5000)
X_proj_tsne_roberta_prods_orig = \
    tsne_roberta_prods_orig.fit_transform(vector_roberta_prods_orig)
# Dataframe pour clustering
df_tsne_roberta_prods_orig = pd.DataFrame({'VAR1' : X_proj_tsne_roberta_prods_orig[:, 0],
                                           'VAR2' : X_proj_tsne_roberta_prods_orig[:, 1],
                                           'CATEGORIE' : data['category']})

Xtrain, Xtest, ytrain, ytest = train_test_split(df_tsne_roberta_prods_orig[cols], data.category_label, test_size=0.2, stratify=data.category_label, random_state=1944)

knn_clf_sup = KNeighborsClassifier()
knn_clf_sup.fit(Xtrain, ytrain)
knn_predictions = knn_clf_sup.predict(Xtrain)
metrics_report("knn", ytrain, knn_predictions, models_performance)

svm_clf_sup = OneVsRestClassifier(LinearSVC(), n_jobs=-1)
svm_clf_sup.fit(Xtrain, ytrain)
svm_predictions = svm_clf_sup.predict(Xtrain)
metrics_report("svm", ytrain, svm_predictions, models_performance)

rf_clf_sup = RandomForestClassifier(n_jobs=-1)
rf_clf_sup.fit(Xtrain, ytrain)
rf_predictions = rf_clf_sup.predict(Xtrain)
metrics_report("Random Forest", ytrain, rf_predictions, models_performance)

gnb_clf_sup = OneVsRestClassifier(GaussianNB())
gnb_clf_sup.fit(Xtrain, ytrain)
gnb_predictions = gnb_clf_sup.predict(Xtrain)
metrics_report("Naive Bayes", ytrain, gnb_predictions, models_performance)    

gb_clf_sup = OneVsRestClassifier(GradientBoostingClassifier())
gb_clf_sup.fit(Xtrain, ytrain)
gb_predictions = gb_clf_sup.predict(Xtrain)
metrics_report("Gradient Boosting", ytrain, gb_predictions, models_performance)

etc_clf_sup = ExtraTreesClassifier(random_state=0)
etc_clf_sup.fit(Xtrain, ytrain)
etc_predictions = etc_clf_sup.predict(Xtrain)
metrics_report("extra tree classifier", ytrain, etc_predictions, models_performance)

result_df = pd.DataFrame.from_dict(models_performance, orient="index")
result_df

from sklearn.ensemble import ExtraTreesClassifier

etc_model = ExtraTreesClassifier(random_state=0).fit(Xtrain, ytrain) 
etc_predictions = etc_model.predict(Xtest) 

# model accuracy for X_test   
accuracy_etc = etc_model.score(Xtest, ytest) 
print("Accuracy de l'algorithme etc :", round(accuracy_etc, 3))

conf_mat = confusion_matrix(le.inverse_transform(ytest), le.inverse_transform(etc_model.predict(Xtest)))
print(conf_mat)
#test_labels = le.inverse_transform(y_test)
labels_transform_train = conf_mat_transform(ytrain, etc_model.predict(Xtrain))
labels_transform_test = conf_mat_transform(ytest, etc_model.predict(Xtest))

accuracy_tsne_roberta_desc_lem_test = accuracy_score(ytest, labels_transform_test)
accuracy_tsne_roberta_desc_lem_train = accuracy_score(ytrain, labels_transform_train)

print()
print("ACCURACY {} (Test): {:.3}".format(type_donnee, accuracy_tsne_roberta_desc_lem_test))
print("ACCURACY {} (Train): {:.3}".format(type_donnee, accuracy_tsne_roberta_desc_lem_train))
print()

print()
print(classification_report(ytest, labels_transform_test))

df_cm = pd.DataFrame(conf_mat, index = [label for label in list_labels],
                  columns = [i for i in "0123456"])
plt.figure(figsize = (6,4))
sns.heatmap(df_cm, annot=True, cmap="Blues")
plt.xlabel('numero Cluster')
plt.ylabel('categorie Depart')
plt.title(f'matrice confusion {type_donnee}')
plt.show()



# Variable dont on veut extraire les features
data['product_specifications_clean'] = \
    data['product_specifications_clean'].fillna('').astype(str)
data_prods_clean = data['product_specifications_clean']

type_donnee = 'TSNE_ROBERTA_PRODS_CLEAN'

# Instanciation du modèle
bert_prods_clean = SentenceTransformer('roberta-large-nli-mean-tokens')

# Encodage
vector_roberta_prods_clean = \
    bert_prods_clean.encode(data_prods_clean, show_progress_bar=False)

# Réduction de dimension tsne
tsne_roberta_prods_clean = TSNE(verbose=1, perplexity=50, n_iter=5000)
X_proj_tsne_roberta_prods_clean = \
    tsne_roberta_prods_clean.fit_transform(vector_roberta_prods_clean)
# Dataframe pour clustering
df_tsne_roberta_prods_clean = pd.DataFrame({'VAR1' : X_proj_tsne_roberta_prods_clean[:, 0],
                                           'VAR2' : X_proj_tsne_roberta_prods_clean[:, 1],
                                           'CATEGORIE' : data['category']})

Xtrain, Xtest, ytrain, ytest = train_test_split(df_tsne_roberta_prods_clean[cols], data.category_label, test_size=0.2, stratify=data.category_label, random_state=1944)

knn_clf_sup = KNeighborsClassifier()
knn_clf_sup.fit(Xtrain, ytrain)
knn_predictions = knn_clf_sup.predict(Xtrain)
metrics_report("knn", ytrain, knn_predictions, models_performance)

svm_clf_sup = OneVsRestClassifier(LinearSVC(), n_jobs=-1)
svm_clf_sup.fit(Xtrain, ytrain)
svm_predictions = svm_clf_sup.predict(Xtrain)
metrics_report("svm", ytrain, svm_predictions, models_performance)

rf_clf_sup = RandomForestClassifier(n_jobs=-1)
rf_clf_sup.fit(Xtrain, ytrain)
rf_predictions = rf_clf_sup.predict(Xtrain)
metrics_report("Random Forest", ytrain, rf_predictions, models_performance)

gnb_clf_sup = OneVsRestClassifier(GaussianNB())
gnb_clf_sup.fit(Xtrain, ytrain)
gnb_predictions = gnb_clf_sup.predict(Xtrain)
metrics_report("Naive Bayes", ytrain, gnb_predictions, models_performance)    

gb_clf_sup = OneVsRestClassifier(GradientBoostingClassifier())
gb_clf_sup.fit(Xtrain, ytrain)
gb_predictions = gb_clf_sup.predict(Xtrain)
metrics_report("Gradient Boosting", ytrain, gb_predictions, models_performance)

etc_clf_sup = ExtraTreesClassifier(random_state=0)
etc_clf_sup.fit(Xtrain, ytrain)
etc_predictions = etc_clf_sup.predict(Xtrain)
metrics_report("extra tree classifier", ytrain, etc_predictions, models_performance)

result_df = pd.DataFrame.from_dict(models_performance, orient="index")
result_df

from sklearn.ensemble import ExtraTreesClassifier

etc_model = ExtraTreesClassifier(random_state=0).fit(Xtrain, ytrain) 
etc_predictions = etc_model.predict(Xtest) 

# model accuracy for X_test   
accuracy_etc = etc_model.score(Xtest, ytest) 
print("Accuracy de l'algorithme etc :", round(accuracy_etc, 3))

conf_mat = confusion_matrix(le.inverse_transform(ytest), le.inverse_transform(etc_model.predict(Xtest)))
print(conf_mat)
#test_labels = le.inverse_transform(y_test)
labels_transform_train = conf_mat_transform(ytrain, etc_model.predict(Xtrain))
labels_transform_test = conf_mat_transform(ytest, etc_model.predict(Xtest))

accuracy_tsne_roberta_prods_clean_test = accuracy_score(ytest, labels_transform_test)
accuracy_tsne_roberta_prods_clean_train = accuracy_score(ytrain, labels_transform_train)

print()
print("ACCURACY {} (Test): {:.3}".format(type_donnee, accuracy_tsne_roberta_prods_clean_test))
print("ACCURACY {} (Train): {:.3}".format(type_donnee, accuracy_tsne_roberta_prods_clean_train))
print()

print()
print(classification_report(ytest, labels_transform_test))

df_cm = pd.DataFrame(conf_mat, index = [label for label in list_labels],
                  columns = [i for i in "0123456"])
plt.figure(figsize = (6,4))
sns.heatmap(df_cm, annot=True, cmap="Blues")
plt.xlabel('numero Cluster')
plt.ylabel('categorie Depart')
plt.title(f'matrice confusion {type_donnee}')
plt.show()

# Variable dont on veut extraire les features
data['product_specifications_lem'] = \
    data['product_specifications_lem'].fillna('').astype(str)
data_prods_lem = data['product_specifications_lem']

type_donnee = 'TSNE_ROBERTA_PRODS_LEM'

# Instanciation du modèle
bert_prods_lem = SentenceTransformer('roberta-large-nli-mean-tokens')

# Encodage
vector_roberta_prods_lem = \
    bert_prods_lem.encode(data_prods_lem, show_progress_bar=False)

# Réduction de dimension tsne
tsne_roberta_prods_lem = TSNE(verbose=1, perplexity=50, n_iter=5000)
X_proj_tsne_roberta_prods_lem = \
    tsne_roberta_prods_lem.fit_transform(vector_roberta_prods_lem)
# Dataframe pour clustering
df_tsne_roberta_prods_lem = pd.DataFrame({'VAR1' : X_proj_tsne_roberta_prods_lem[:, 0],
                                           'VAR2' : X_proj_tsne_roberta_prods_lem[:, 1],
                                           'CATEGORIE' : data['category']})

Xtrain, Xtest, ytrain, ytest = train_test_split(df_tsne_roberta_prods_lem[cols], data.category_label, test_size=0.2, stratify=data.category_label, random_state=1944)

knn_clf_sup = KNeighborsClassifier()
knn_clf_sup.fit(Xtrain, ytrain)
knn_predictions = knn_clf_sup.predict(Xtrain)
metrics_report("knn", ytrain, knn_predictions, models_performance)

svm_clf_sup = OneVsRestClassifier(LinearSVC(), n_jobs=-1)
svm_clf_sup.fit(Xtrain, ytrain)
svm_predictions = svm_clf_sup.predict(Xtrain)
metrics_report("svm", ytrain, svm_predictions, models_performance)

rf_clf_sup = RandomForestClassifier(n_jobs=-1)
rf_clf_sup.fit(Xtrain, ytrain)
rf_predictions = rf_clf_sup.predict(Xtrain)
metrics_report("Random Forest", ytrain, rf_predictions, models_performance)

gnb_clf_sup = OneVsRestClassifier(GaussianNB())
gnb_clf_sup.fit(Xtrain, ytrain)
gnb_predictions = gnb_clf_sup.predict(Xtrain)
metrics_report("Naive Bayes", ytrain, gnb_predictions, models_performance)    

gb_clf_sup = OneVsRestClassifier(GradientBoostingClassifier())
gb_clf_sup.fit(Xtrain, ytrain)
gb_predictions = gb_clf_sup.predict(Xtrain)
metrics_report("Gradient Boosting", ytrain, gb_predictions, models_performance)

etc_clf_sup = ExtraTreesClassifier(random_state=0)
etc_clf_sup.fit(Xtrain, ytrain)
etc_predictions = etc_clf_sup.predict(Xtrain)
metrics_report("extra tree classifier", ytrain, etc_predictions, models_performance)

result_df = pd.DataFrame.from_dict(models_performance, orient="index")
result_df

from sklearn.ensemble import ExtraTreesClassifier

etc_model = ExtraTreesClassifier(random_state=0).fit(Xtrain, ytrain) 
etc_predictions = etc_model.predict(Xtest) 

# model accuracy for X_test   
accuracy_etc = etc_model.score(Xtest, ytest) 
print("Accuracy de l'algorithme etc :", round(accuracy_etc, 3))

conf_mat = confusion_matrix(le.inverse_transform(ytest), le.inverse_transform(etc_model.predict(Xtest)))
print(conf_mat)
#test_labels = le.inverse_transform(y_test)
labels_transform_train = conf_mat_transform(ytrain, etc_model.predict(Xtrain))
labels_transform_test = conf_mat_transform(ytest, etc_model.predict(Xtest))

accuracy_tsne_roberta_prods_lem_test = accuracy_score(ytest, labels_transform_test)
accuracy_tsne_roberta_prods_lem_train = accuracy_score(ytrain, labels_transform_train)

print()
print("ACCURACY {} (Test): {:.3}".format(type_donnee, accuracy_tsne_roberta_prods_lem_test))
print("ACCURACY {} (Train): {:.3}".format(type_donnee, accuracy_tsne_roberta_prods_lem_train))
print()

print()
print(classification_report(ytest, labels_transform_test))

df_cm = pd.DataFrame(conf_mat, index = [label for label in list_labels],
                  columns = [i for i in "0123456"])
plt.figure(figsize = (6,4))
sns.heatmap(df_cm, annot=True, cmap="Blues")
plt.xlabel('numero Cluster')
plt.ylabel('categorie Depart')
plt.title(f'matrice confusion {type_donnee}')
plt.show()

# Variable dont on veut extraire les features
data['product_specifications_stem'] = \
    data['product_specifications_stem'].fillna('').astype(str)
data_prods_stem = data['product_specifications_stem']

type_donnee = 'TSNE_ROBERTA_PRODS_STEM'

# Instanciation du modèle
bert_prods_stem = SentenceTransformer('roberta-large-nli-mean-tokens')

# Encodage
vector_roberta_prods_stem = \
    bert_prods_stem.encode(data_prods_stem, show_progress_bar=False)

# Réduction de dimension tsne
tsne_roberta_prods_stem = TSNE(verbose=1, perplexity=50, n_iter=5000)
X_proj_tsne_roberta_prods_stem = \
    tsne_roberta_prods_stem.fit_transform(vector_roberta_prods_stem)
# Dataframe pour clustering
df_tsne_roberta_prods_stem = pd.DataFrame({'VAR1' : X_proj_tsne_roberta_prods_stem[:, 0],
                                           'VAR2' : X_proj_tsne_roberta_prods_stem[:, 1],
                                           'CATEGORIE' : data['category']})

Xtrain, Xtest, ytrain, ytest = train_test_split(df_tsne_roberta_prods_stem[cols], data.category_label, test_size=0.2, stratify=data.category_label, random_state=1944)

knn_clf_sup = KNeighborsClassifier()
knn_clf_sup.fit(Xtrain, ytrain)
knn_predictions = knn_clf_sup.predict(Xtrain)
metrics_report("knn", ytrain, knn_predictions, models_performance)

svm_clf_sup = OneVsRestClassifier(LinearSVC(), n_jobs=-1)
svm_clf_sup.fit(Xtrain, ytrain)
svm_predictions = svm_clf_sup.predict(Xtrain)
metrics_report("svm", ytrain, svm_predictions, models_performance)

rf_clf_sup = RandomForestClassifier(n_jobs=-1)
rf_clf_sup.fit(Xtrain, ytrain)
rf_predictions = rf_clf_sup.predict(Xtrain)
metrics_report("Random Forest", ytrain, rf_predictions, models_performance)

gnb_clf_sup = OneVsRestClassifier(GaussianNB())
gnb_clf_sup.fit(Xtrain, ytrain)
gnb_predictions = gnb_clf_sup.predict(Xtrain)
metrics_report("Naive Bayes", ytrain, gnb_predictions, models_performance)    

gb_clf_sup = OneVsRestClassifier(GradientBoostingClassifier())
gb_clf_sup.fit(Xtrain, ytrain)
gb_predictions = gb_clf_sup.predict(Xtrain)
metrics_report("Gradient Boosting", ytrain, gb_predictions, models_performance)

etc_clf_sup = ExtraTreesClassifier(random_state=0)
etc_clf_sup.fit(Xtrain, ytrain)
etc_predictions = etc_clf_sup.predict(Xtrain)
metrics_report("extra tree classifier", ytrain, etc_predictions, models_performance)

result_df = pd.DataFrame.from_dict(models_performance, orient="index")
result_df

from sklearn.ensemble import ExtraTreesClassifier

etc_model = ExtraTreesClassifier(random_state=0).fit(Xtrain, ytrain) 
etc_predictions = etc_model.predict(Xtest) 

# model accuracy for X_test   
accuracy_etc = etc_model.score(Xtest, ytest) 
print("Accuracy de l'algorithme etc :", round(accuracy_etc, 3))

conf_mat = confusion_matrix(le.inverse_transform(ytest), le.inverse_transform(etc_model.predict(Xtest)))
print(conf_mat)
#test_labels = le.inverse_transform(y_test)
labels_transform_train = conf_mat_transform(ytrain, etc_model.predict(Xtrain))
labels_transform_test = conf_mat_transform(ytest, etc_model.predict(Xtest))

accuracy_tsne_roberta_prods_stem_test = accuracy_score(ytest, labels_transform_test)
accuracy_tsne_roberta_prods_stem_train = accuracy_score(ytrain, labels_transform_train)

print()
print("ACCURACY {} (Test): {:.3}".format(type_donnee, accuracy_tsne_roberta_prods_stem_test))
print("ACCURACY {} (Train): {:.3}".format(type_donnee, accuracy_tsne_roberta_prods_stem_train))
print()

print()
print(classification_report(ytest, labels_transform_test))

df_cm = pd.DataFrame(conf_mat, index = [label for label in list_labels],
                  columns = [i for i in "0123456"])
plt.figure(figsize = (6,4))
sns.heatmap(df_cm, annot=True, cmap="Blues")
plt.xlabel('numero Cluster')
plt.ylabel('categorie Depart')
plt.title(f'matrice confusion {type_donnee}')
plt.show()

# Variable dont on veut extraire les features
data['product_name'] = \
    data['product_name'].fillna('').astype(str)
data_prod_orig = data['product_name']

type_donnee = 'TSNE_ROBERTA_PROD_ORIG'

# Instanciation du modèle
bert_prod_orig = SentenceTransformer('roberta-large-nli-mean-tokens')

# Encodage
vector_roberta_prod_orig = \
    bert_prod_orig.encode(data_prod_orig, show_progress_bar=False)

# Réduction de dimension tsne
tsne_roberta_prod_orig = TSNE(verbose=1, perplexity=50, n_iter=5000)
X_proj_tsne_roberta_prod_orig = \
    tsne_roberta_prod_orig.fit_transform(vector_roberta_prod_orig)
# Dataframe pour clustering
df_tsne_roberta_prod_orig = pd.DataFrame({'VAR1' : X_proj_tsne_roberta_prod_orig[:, 0],
                                           'VAR2' : X_proj_tsne_roberta_prod_orig[:, 1],
                                           'CATEGORIE' : data['category']})

Xtrain, Xtest, ytrain, ytest = train_test_split(df_tsne_roberta_prod_orig[cols], data.category_label, test_size=0.2, stratify=data.category_label, random_state=1944)

knn_clf_sup = KNeighborsClassifier()
knn_clf_sup.fit(Xtrain, ytrain)
knn_predictions = knn_clf_sup.predict(Xtrain)
metrics_report("knn", ytrain, knn_predictions, models_performance)

svm_clf_sup = OneVsRestClassifier(LinearSVC(), n_jobs=-1)
svm_clf_sup.fit(Xtrain, ytrain)
svm_predictions = svm_clf_sup.predict(Xtrain)
metrics_report("svm", ytrain, svm_predictions, models_performance)

rf_clf_sup = RandomForestClassifier(n_jobs=-1)
rf_clf_sup.fit(Xtrain, ytrain)
rf_predictions = rf_clf_sup.predict(Xtrain)
metrics_report("Random Forest", ytrain, rf_predictions, models_performance)

gnb_clf_sup = OneVsRestClassifier(GaussianNB())
gnb_clf_sup.fit(Xtrain, ytrain)
gnb_predictions = gnb_clf_sup.predict(Xtrain)
metrics_report("Naive Bayes", ytrain, gnb_predictions, models_performance)    

gb_clf_sup = OneVsRestClassifier(GradientBoostingClassifier())
gb_clf_sup.fit(Xtrain, ytrain)
gb_predictions = gb_clf_sup.predict(Xtrain)
metrics_report("Gradient Boosting", ytrain, gb_predictions, models_performance)

etc_clf_sup = ExtraTreesClassifier(random_state=0)
etc_clf_sup.fit(Xtrain, ytrain)
etc_predictions = etc_clf_sup.predict(Xtrain)
metrics_report("extra tree classifier", ytrain, etc_predictions, models_performance)

result_df = pd.DataFrame.from_dict(models_performance, orient="index")
result_df

from sklearn.ensemble import ExtraTreesClassifier

etc_model = ExtraTreesClassifier(random_state=0).fit(Xtrain, ytrain) 
etc_predictions = etc_model.predict(Xtest) 

# model accuracy for X_test   
accuracy_etc = etc_model.score(Xtest, ytest) 
print("Accuracy de l'algorithme etc :", round(accuracy_etc, 3))

conf_mat = confusion_matrix(le.inverse_transform(ytest), le.inverse_transform(etc_model.predict(Xtest)))
print(conf_mat)
#test_labels = le.inverse_transform(y_test)
labels_transform_train = conf_mat_transform(ytrain, etc_model.predict(Xtrain))
labels_transform_test = conf_mat_transform(ytest, etc_model.predict(Xtest))

accuracy_tsne_roberta_prod_orig_test = accuracy_score(ytest, labels_transform_test)
accuracy_tsne_roberta_prod_orig_train = accuracy_score(ytrain, labels_transform_train)

print()
print("ACCURACY {} (Test): {:.3}".format(type_donnee, accuracy_tsne_roberta_prod_orig_test))
print("ACCURACY {} (Train): {:.3}".format(type_donnee, accuracy_tsne_roberta_prod_orig_train))
print()

print()
print(classification_report(ytest, labels_transform_test))

df_cm = pd.DataFrame(conf_mat, index = [label for label in list_labels],
                  columns = [i for i in "0123456"])
plt.figure(figsize = (6,4))
sns.heatmap(df_cm, annot=True, cmap="Blues")
plt.xlabel('numero Cluster')
plt.ylabel('categorie Depart')
plt.title(f'matrice confusion {type_donnee}')
plt.show()

# Variable dont on veut extraire les features
data['product_name_lem'] = \
    data['product_name_lem'].fillna('').astype(str)
data_prod_lem = data['product_name_lem']

type_donnee = 'TSNE_ROBERTA_PROD_LEM'

# Instanciation du modèle
bert_prod_lem = SentenceTransformer('roberta-large-nli-mean-tokens')

# Encodage
vector_roberta_prod_lem = \
    bert_prod_lem.encode(data_prod_lem, show_progress_bar=False)

# Réduction de dimension tsne
tsne_roberta_prod_lem = TSNE(verbose=1, perplexity=50, n_iter=5000)
X_proj_tsne_roberta_prod_lem = \
    tsne_roberta_prod_lem.fit_transform(vector_roberta_prod_lem)
# Dataframe pour clustering
df_tsne_roberta_prod_lem = pd.DataFrame({'VAR1' : X_proj_tsne_roberta_prod_lem[:, 0],
                                           'VAR2' : X_proj_tsne_roberta_prod_lem[:, 1],
                                           'CATEGORIE' : data['category']})

Xtrain, Xtest, ytrain, ytest = train_test_split(df_tsne_roberta_prod_lem[cols], data.category_label, test_size=0.2, stratify=data.category_label, random_state=1944)

knn_clf_sup = KNeighborsClassifier()
knn_clf_sup.fit(Xtrain, ytrain)
knn_predictions = knn_clf_sup.predict(Xtrain)
metrics_report("knn", ytrain, knn_predictions, models_performance)

svm_clf_sup = OneVsRestClassifier(LinearSVC(), n_jobs=-1)
svm_clf_sup.fit(Xtrain, ytrain)
svm_predictions = svm_clf_sup.predict(Xtrain)
metrics_report("svm", ytrain, svm_predictions, models_performance)

rf_clf_sup = RandomForestClassifier(n_jobs=-1)
rf_clf_sup.fit(Xtrain, ytrain)
rf_predictions = rf_clf_sup.predict(Xtrain)
metrics_report("Random Forest", ytrain, rf_predictions, models_performance)

gnb_clf_sup = OneVsRestClassifier(GaussianNB())
gnb_clf_sup.fit(Xtrain, ytrain)
gnb_predictions = gnb_clf_sup.predict(Xtrain)
metrics_report("Naive Bayes", ytrain, gnb_predictions, models_performance)    

gb_clf_sup = OneVsRestClassifier(GradientBoostingClassifier())
gb_clf_sup.fit(Xtrain, ytrain)
gb_predictions = gb_clf_sup.predict(Xtrain)
metrics_report("Gradient Boosting", ytrain, gb_predictions, models_performance)

etc_clf_sup = ExtraTreesClassifier(random_state=0)
etc_clf_sup.fit(Xtrain, ytrain)
etc_predictions = etc_clf_sup.predict(Xtrain)
metrics_report("extra tree classifier", ytrain, etc_predictions, models_performance)

result_df = pd.DataFrame.from_dict(models_performance, orient="index")
result_df

from sklearn.ensemble import ExtraTreesClassifier

etc_model = ExtraTreesClassifier(random_state=0).fit(Xtrain, ytrain) 
etc_predictions = etc_model.predict(Xtest) 

# model accuracy for X_test   
accuracy_etc = etc_model.score(Xtest, ytest) 
print("Accuracy de l'algorithme etc :", round(accuracy_etc, 3))

conf_mat = confusion_matrix(le.inverse_transform(ytest), le.inverse_transform(etc_model.predict(Xtest)))
print(conf_mat)
#test_labels = le.inverse_transform(y_test)
labels_transform_train = conf_mat_transform(ytrain, etc_model.predict(Xtrain))
labels_transform_test = conf_mat_transform(ytest, etc_model.predict(Xtest))

accuracy_tsne_roberta_prod_lem_test = accuracy_score(ytest, labels_transform_test)
accuracy_tsne_roberta_prod_lem_train = accuracy_score(ytrain, labels_transform_train)

print()
print("ACCURACY {} (Test): {:.3}".format(type_donnee, accuracy_tsne_roberta_prod_lem_test))
print("ACCURACY {} (Train): {:.3}".format(type_donnee, accuracy_tsne_roberta_prod_lem_train))
print()

print()
print(classification_report(ytest, labels_transform_test))

df_cm = pd.DataFrame(conf_mat, index = [label for label in list_labels],
                  columns = [i for i in "0123456"])
plt.figure(figsize = (6,4))
sns.heatmap(df_cm, annot=True, cmap="Blues")
plt.xlabel('numero Cluster')
plt.ylabel('categorie Depart')
plt.title(f'matrice confusion {type_donnee}')
plt.show()

# Variable dont on veut extraire les features
data['product_name_stem'] = \
    data['product_name_stem'].fillna('').astype(str)
data_prod_stem = data['product_name_stem']

type_donnee = 'TSNE_ROBERTA_PROD_STEM'

# Instanciation du modèle
bert_prod_stem = SentenceTransformer('roberta-large-nli-mean-tokens')

# Encodage
vector_roberta_prod_stem = \
    bert_prod_stem.encode(data_prod_stem, show_progress_bar=False)

# Réduction de dimension tsne
tsne_roberta_prod_stem = TSNE(verbose=1, perplexity=50, n_iter=5000)
X_proj_tsne_roberta_prod_stem = \
    tsne_roberta_prod_stem.fit_transform(vector_roberta_prod_stem)
# Dataframe pour clustering
df_tsne_roberta_prod_stem = pd.DataFrame({'VAR1' : X_proj_tsne_roberta_prod_stem[:, 0],
                                           'VAR2' : X_proj_tsne_roberta_prod_stem[:, 1],
                                           'CATEGORIE' : data['category']})

Xtrain, Xtest, ytrain, ytest = train_test_split(df_tsne_roberta_prod_stem[cols], data.category_label, test_size=0.2, stratify=data.category_label, random_state=1944)

knn_clf_sup = KNeighborsClassifier()
knn_clf_sup.fit(Xtrain, ytrain)
knn_predictions = knn_clf_sup.predict(Xtrain)
metrics_report("knn", ytrain, knn_predictions, models_performance)

svm_clf_sup = OneVsRestClassifier(LinearSVC(), n_jobs=-1)
svm_clf_sup.fit(Xtrain, ytrain)
svm_predictions = svm_clf_sup.predict(Xtrain)
metrics_report("svm", ytrain, svm_predictions, models_performance)

rf_clf_sup = RandomForestClassifier(n_jobs=-1)
rf_clf_sup.fit(Xtrain, ytrain)
rf_predictions = rf_clf_sup.predict(Xtrain)
metrics_report("Random Forest", ytrain, rf_predictions, models_performance)

gnb_clf_sup = OneVsRestClassifier(GaussianNB())
gnb_clf_sup.fit(Xtrain, ytrain)
gnb_predictions = gnb_clf_sup.predict(Xtrain)
metrics_report("Naive Bayes", ytrain, gnb_predictions, models_performance)    

gb_clf_sup = OneVsRestClassifier(GradientBoostingClassifier())
gb_clf_sup.fit(Xtrain, ytrain)
gb_predictions = gb_clf_sup.predict(Xtrain)
metrics_report("Gradient Boosting", ytrain, gb_predictions, models_performance)

etc_clf_sup = ExtraTreesClassifier(random_state=0)
etc_clf_sup.fit(Xtrain, ytrain)
etc_predictions = etc_clf_sup.predict(Xtrain)
metrics_report("extra tree classifier", ytrain, etc_predictions, models_performance)

result_df = pd.DataFrame.from_dict(models_performance, orient="index")
result_df

etc_model = ExtraTreesClassifier(random_state=0).fit(Xtrain, ytrain) 
etc_predictions = etc_model.predict(Xtest) 

# model accuracy for X_test   
accuracy_etc = etc_model.score(Xtest, ytest) 
print("Accuracy de l'algorithme etc :", round(accuracy_etc, 3))

conf_mat = confusion_matrix(le.inverse_transform(ytest), le.inverse_transform(etc_model.predict(Xtest)))
print(conf_mat)
#test_labels = le.inverse_transform(y_test)
labels_transform_train = conf_mat_transform(ytrain, etc_model.predict(Xtrain))
labels_transform_test = conf_mat_transform(ytest, etc_model.predict(Xtest))

accuracy_tsne_roberta_prod_stem_test = accuracy_score(ytest, labels_transform_test)
accuracy_tsne_roberta_prod_stem_train = accuracy_score(ytrain, labels_transform_train)

print()
print("ACCURACY {} (Test): {:.3}".format(type_donnee, accuracy_tsne_roberta_prod_stem_test))
print("ACCURACY {} (Train): {:.3}".format(type_donnee, accuracy_tsne_roberta_prod_stem_train))
print()

print()
print(classification_report(ytest, labels_transform_test))

df_cm = pd.DataFrame(conf_mat, index = [label for label in list_labels],
                  columns = [i for i in "0123456"])
plt.figure(figsize = (6,4))
sns.heatmap(df_cm, annot=True, cmap="Blues")
plt.xlabel('numero Cluster')
plt.ylabel('categorie Depart')
plt.title(f'matrice confusion {type_donnee}')
plt.show()

# Variable dont on veut extraire les features
data['var_orig'] = data['product_name'] + data['brand']
+ data['description'] + data['product_specifications']
data['var_orig'] = \
    data['var_orig'].fillna('').astype(str)
data_var_orig = data['var_orig']

type_donnee = 'TSNE_ROBERTA_VAR_ORIG'

# Instanciation du modèle
bert_var_orig = SentenceTransformer('roberta-large-nli-mean-tokens')

# Encodage
vector_roberta_var_orig = \
    bert_var_orig.encode(data_var_orig, show_progress_bar=False)

# Réduction de dimension tsne
tsne_roberta_var_orig = TSNE(verbose=1, perplexity=40, n_iter=5000)
X_proj_tsne_roberta_var_orig = \
    tsne_roberta_var_orig.fit_transform(vector_roberta_var_orig)
# Dataframe pour clustering
df_tsne_roberta_var_orig = pd.DataFrame({'VAR1' : X_proj_tsne_roberta_var_orig[:, 0],
                                           'VAR2' : X_proj_tsne_roberta_var_orig[:, 1],
                                           'CATEGORIE' : data['category']})

Xtrain, Xtest, ytrain, ytest = train_test_split(df_tsne_roberta_var_orig[cols], data.category_label, test_size=0.2, stratify=data.category_label, random_state=1944)

knn_clf_sup = KNeighborsClassifier()
knn_clf_sup.fit(Xtrain, ytrain)
knn_predictions = knn_clf_sup.predict(Xtrain)
metrics_report("knn", ytrain, knn_predictions, models_performance)

svm_clf_sup = OneVsRestClassifier(LinearSVC(), n_jobs=-1)
svm_clf_sup.fit(Xtrain, ytrain)
svm_predictions = svm_clf_sup.predict(Xtrain)
metrics_report("svm", ytrain, svm_predictions, models_performance)

rf_clf_sup = RandomForestClassifier(n_jobs=-1)
rf_clf_sup.fit(Xtrain, ytrain)
rf_predictions = rf_clf_sup.predict(Xtrain)
metrics_report("Random Forest", ytrain, rf_predictions, models_performance)

gnb_clf_sup = OneVsRestClassifier(GaussianNB())
gnb_clf_sup.fit(Xtrain, ytrain)
gnb_predictions = gnb_clf_sup.predict(Xtrain)
metrics_report("Naive Bayes", ytrain, gnb_predictions, models_performance)    

gb_clf_sup = OneVsRestClassifier(GradientBoostingClassifier())
gb_clf_sup.fit(Xtrain, ytrain)
gb_predictions = gb_clf_sup.predict(Xtrain)
metrics_report("Gradient Boosting", ytrain, gb_predictions, models_performance)

etc_clf_sup = ExtraTreesClassifier(random_state=0)
etc_clf_sup.fit(Xtrain, ytrain)
etc_predictions = etc_clf_sup.predict(Xtrain)
metrics_report("extra tree classifier", ytrain, etc_predictions, models_performance)

result_df = pd.DataFrame.from_dict(models_performance, orient="index")
result_df

etc_model = ExtraTreesClassifier(random_state=0).fit(Xtrain, ytrain) 
etc_predictions = etc_model.predict(Xtest) 

# model accuracy for X_test   
accuracy_etc = etc_model.score(Xtest, ytest) 
print("Accuracy de l'algorithme etc :", round(accuracy_etc, 3))

conf_mat = confusion_matrix(le.inverse_transform(ytest), le.inverse_transform(etc_model.predict(Xtest)))
print(conf_mat)
#test_labels = le.inverse_transform(y_test)
labels_transform_train = conf_mat_transform(ytrain, etc_model.predict(Xtrain))
labels_transform_test = conf_mat_transform(ytest, etc_model.predict(Xtest))

accuracy_tsne_roberta_var_orig_test = accuracy_score(ytest, labels_transform_test)
accuracy_tsne_roberta_var_orig_train = accuracy_score(ytrain, labels_transform_train)

print()
print("ACCURACY {} (Test): {:.3}".format(type_donnee, accuracy_tsne_roberta_var_orig_test))
print("ACCURACY {} (Train): {:.3}".format(type_donnee, accuracy_tsne_roberta_var_orig_train))
print()

print()
print(classification_report(ytest, labels_transform_test))

df_cm = pd.DataFrame(conf_mat, index = [label for label in list_labels],
                  columns = [i for i in "0123456"])
plt.figure(figsize = (6,4))
sns.heatmap(df_cm, annot=True, cmap="Blues")
plt.xlabel('numero Cluster')
plt.ylabel('categorie Depart')
plt.title(f'matrice confusion {type_donnee}')
plt.show()

# Variable dont on veut extraire les features
data['var_stem'] = data['product_name_stem'] + data['brand_stem']
+ data['desc_stem'] + data['product_specifications_stem']
data['var_stem'] = \
    data['var_stem'].fillna('').astype(str)
data_var_stem = data['var_stem']

type_donnee = 'TSNE_ROBERTA_VAR_STEM'

# Instanciation du modèle
bert_var_stem = SentenceTransformer('roberta-large-nli-mean-tokens')

# Encodage
vector_roberta_var_stem = \
    bert_var_stem.encode(data_var_stem, show_progress_bar=False)

# Réduction de dimension tsne
tsne_roberta_var_stem = TSNE(verbose=1, perplexity=50, n_iter=5000)
X_proj_tsne_roberta_var_stem = \
    tsne_roberta_var_stem.fit_transform(vector_roberta_var_stem)
# Dataframe pour clustering
df_tsne_roberta_var_stem = pd.DataFrame({'VAR1' : X_proj_tsne_roberta_var_stem[:, 0],
                                           'VAR2' : X_proj_tsne_roberta_var_stem[:, 1],
                                           'CATEGORIE' : data['category']})

Xtrain, Xtest, ytrain, ytest = train_test_split(df_tsne_roberta_var_stem[cols], data.category_label, test_size=0.2, stratify=data.category_label, random_state=1944)

knn_clf_sup = KNeighborsClassifier()
knn_clf_sup.fit(Xtrain, ytrain)
knn_predictions = knn_clf_sup.predict(Xtrain)
metrics_report("knn", ytrain, knn_predictions, models_performance)

svm_clf_sup = OneVsRestClassifier(LinearSVC(), n_jobs=-1)
svm_clf_sup.fit(Xtrain, ytrain)
svm_predictions = svm_clf_sup.predict(Xtrain)
metrics_report("svm", ytrain, svm_predictions, models_performance)

rf_clf_sup = RandomForestClassifier(n_jobs=-1)
rf_clf_sup.fit(Xtrain, ytrain)
rf_predictions = rf_clf_sup.predict(Xtrain)
metrics_report("Random Forest", ytrain, rf_predictions, models_performance)

gnb_clf_sup = OneVsRestClassifier(GaussianNB())
gnb_clf_sup.fit(Xtrain, ytrain)
gnb_predictions = gnb_clf_sup.predict(Xtrain)
metrics_report("Naive Bayes", ytrain, gnb_predictions, models_performance)    

gb_clf_sup = OneVsRestClassifier(GradientBoostingClassifier())
gb_clf_sup.fit(Xtrain, ytrain)
gb_predictions = gb_clf_sup.predict(Xtrain)
metrics_report("Gradient Boosting", ytrain, gb_predictions, models_performance)

etc_clf_sup = ExtraTreesClassifier(random_state=0)
etc_clf_sup.fit(Xtrain, ytrain)
etc_predictions = etc_clf_sup.predict(Xtrain)
metrics_report("extra tree classifier", ytrain, etc_predictions, models_performance)

result_df = pd.DataFrame.from_dict(models_performance, orient="index")
result_df

etc_model = ExtraTreesClassifier(random_state=0).fit(Xtrain, ytrain) 
etc_predictions = etc_model.predict(Xtest) 

# model accuracy for X_test   
accuracy_etc = etc_model.score(Xtest, ytest) 
print("Accuracy de l'algorithme etc :", round(accuracy_etc, 3))

conf_mat = confusion_matrix(le.inverse_transform(ytest), le.inverse_transform(etc_model.predict(Xtest)))
print(conf_mat)
#test_labels = le.inverse_transform(y_test)
labels_transform_train = conf_mat_transform(ytrain, etc_model.predict(Xtrain))
labels_transform_test = conf_mat_transform(ytest, etc_model.predict(Xtest))

accuracy_tsne_roberta_var_stem _test = accuracy_score(ytest, labels_transform_test)
accuracy_tsne_roberta_var_stem _train = accuracy_score(ytrain, labels_transform_train)

print()
print("ACCURACY {} (Test): {:.3}".format(type_donnee, accuracy_tsne_roberta_var_stem _test))
print("ACCURACY {} (Train): {:.3}".format(type_donnee, accuracy_tsne_roberta_var_stem _train))
print()

print()
print(classification_report(ytest, labels_transform_test))

df_cm = pd.DataFrame(conf_mat, index = [label for label in list_labels],
                  columns = [i for i in "0123456"])
plt.figure(figsize = (6,4))
sns.heatmap(df_cm, annot=True, cmap="Blues")
plt.xlabel('numero Cluster')
plt.ylabel('categorie Depart')
plt.title(f'matrice confusion {type_donnee}')
plt.show()

# Variable dont on veut extraire les features
data['var_prod_stem'] = data['product_name_stem']
+ data['product_specifications_stem']
data['var_prod_stem'] = \
    data['var_prod_stem'].fillna('').astype(str)
data_var_prod_stem = data['var_prod_stem']

type_donnee = 'TSNE_ROBERTA_VAR_PROD_STEM'

# Instanciation du modèle
bert_var_prod_stem = SentenceTransformer('roberta-large-nli-mean-tokens')

# Encodage
vector_roberta_var_prod_stem = \
    bert_var_prod_stem.encode(data_var_prod_stem, show_progress_bar=False)

# Réduction de dimension tsne
tsne_roberta_var_prod_stem = TSNE(verbose=1, perplexity=50, n_iter=5000)
X_proj_tsne_roberta_var_prod_stem = \
    tsne_roberta_var_prod_stem.fit_transform(vector_roberta_var_prod_stem)
# Dataframe pour clustering
df_tsne_roberta_var_prod_stem = pd.DataFrame({'VAR1' : X_proj_tsne_roberta_var_prod_stem[:, 0],
                                           'VAR2' : X_proj_tsne_roberta_var_prod_stem[:, 1],
                                           'CATEGORIE' : data['category']})

Xtrain, Xtest, ytrain, ytest = train_test_split(df_tsne_roberta_var_prod_stem[cols], data.category_label, test_size=0.2, stratify=data.category_label, random_state=1944)

knn_clf_sup = KNeighborsClassifier()
knn_clf_sup.fit(Xtrain, ytrain)
knn_predictions = knn_clf_sup.predict(Xtrain)
metrics_report("knn", ytrain, knn_predictions, models_performance)

svm_clf_sup = OneVsRestClassifier(LinearSVC(), n_jobs=-1)
svm_clf_sup.fit(Xtrain, ytrain)
svm_predictions = svm_clf_sup.predict(Xtrain)
metrics_report("svm", ytrain, svm_predictions, models_performance)

rf_clf_sup = RandomForestClassifier(n_jobs=-1)
rf_clf_sup.fit(Xtrain, ytrain)
rf_predictions = rf_clf_sup.predict(Xtrain)
metrics_report("Random Forest", ytrain, rf_predictions, models_performance)

gnb_clf_sup = OneVsRestClassifier(GaussianNB())
gnb_clf_sup.fit(Xtrain, ytrain)
gnb_predictions = gnb_clf_sup.predict(Xtrain)
metrics_report("Naive Bayes", ytrain, gnb_predictions, models_performance)    

gb_clf_sup = OneVsRestClassifier(GradientBoostingClassifier())
gb_clf_sup.fit(Xtrain, ytrain)
gb_predictions = gb_clf_sup.predict(Xtrain)
metrics_report("Gradient Boosting", ytrain, gb_predictions, models_performance)

etc_clf_sup = ExtraTreesClassifier(random_state=0)
etc_clf_sup.fit(Xtrain, ytrain)
etc_predictions = etc_clf_sup.predict(Xtrain)
metrics_report("extra tree classifier", ytrain, etc_predictions, models_performance)

result_df = pd.DataFrame.from_dict(models_performance, orient="index")
result_df

etc_model = ExtraTreesClassifier(random_state=0).fit(Xtrain, ytrain) 
etc_predictions = etc_model.predict(Xtest) 

# model accuracy for X_test   
accuracy_etc = etc_model.score(Xtest, ytest) 
print("Accuracy de l'algorithme etc :", round(accuracy_etc, 3))

conf_mat = confusion_matrix(le.inverse_transform(ytest), le.inverse_transform(etc_model.predict(Xtest)))
print(conf_mat)
#test_labels = le.inverse_transform(y_test)
labels_transform_train = conf_mat_transform(ytrain, etc_model.predict(Xtrain))
labels_transform_test = conf_mat_transform(ytest, etc_model.predict(Xtest))

accuracy_tsne_roberta_var_prod_stem_test = accuracy_score(ytest, labels_transform_test)
accuracy_tsne_roberta_var_prod_stem_train = accuracy_score(ytrain, labels_transform_train)

print()
print("ACCURACY {} (Test): {:.3}".format(type_donnee, accuracy_tsne_roberta_var_prod_stem_test))
print("ACCURACY {} (Train): {:.3}".format(type_donnee, accuracy_tsne_roberta_var_prod_stem_train))
print()

print()
print(classification_report(ytest, labels_transform_test))

df_cm = pd.DataFrame(conf_mat, index = [label for label in list_labels],
                  columns = [i for i in "0123456"])
plt.figure(figsize = (6,4))
sns.heatmap(df_cm, annot=True, cmap="Blues")
plt.xlabel('numero Cluster')
plt.ylabel('categorie Depart')
plt.title(f'matrice confusion {type_donnee}')
plt.show()



type_donnee = 'TSNE_DISTILSST2_PROD_STEM'

# Instanciation du modèle
bert_distilsst2_prods_stem = \
    SentenceTransformer('distilbert-base-uncased-finetuned-sst-2-english')
# Encodage
vector_distilsst2_prods_stem = \
    bert_distilsst2_prods_stem.encode(data_prods_stem,
                                     show_progress_bar=False)

# Réduction de dimension tsne
tsne_distilsst2_prods_stem = TSNE(verbose=1, perplexity=80, n_iter=5000)
X_proj_tsne_distilsst2_prods_stem = \
    tsne_distilsst2_prods_stem.fit_transform(vector_distilsst2_prods_stem)
# Dataframe pour clustering
df_tsne_distilsst2_prods_stem = \
    pd.DataFrame({'VAR1' : X_proj_tsne_distilsst2_prods_stem[:, 0],
                  'VAR2' : X_proj_tsne_distilsst2_prods_stem[:, 1],
                  'CATEGORIE' : data['category']})

Xtrain, Xtest, ytrain, ytest = train_test_split(df_tsne_distilsst2_prods_stem[cols], data.category_label, test_size=0.2, stratify=data.category_label, random_state=1944)

knn_clf_sup = KNeighborsClassifier()
knn_clf_sup.fit(Xtrain, ytrain)
knn_predictions = knn_clf_sup.predict(Xtrain)
metrics_report("knn", ytrain, knn_predictions, models_performance)

svm_clf_sup = OneVsRestClassifier(LinearSVC(), n_jobs=-1)
svm_clf_sup.fit(Xtrain, ytrain)
svm_predictions = svm_clf_sup.predict(Xtrain)
metrics_report("svm", ytrain, svm_predictions, models_performance)

rf_clf_sup = RandomForestClassifier(n_jobs=-1)
rf_clf_sup.fit(Xtrain, ytrain)
rf_predictions = rf_clf_sup.predict(Xtrain)
metrics_report("Random Forest", ytrain, rf_predictions, models_performance)

gnb_clf_sup = OneVsRestClassifier(GaussianNB())
gnb_clf_sup.fit(Xtrain, ytrain)
gnb_predictions = gnb_clf_sup.predict(Xtrain)
metrics_report("Naive Bayes", ytrain, gnb_predictions, models_performance)    

gb_clf_sup = OneVsRestClassifier(GradientBoostingClassifier())
gb_clf_sup.fit(Xtrain, ytrain)
gb_predictions = gb_clf_sup.predict(Xtrain)
metrics_report("Gradient Boosting", ytrain, gb_predictions, models_performance)

etc_clf_sup = ExtraTreesClassifier(random_state=0)
etc_clf_sup.fit(Xtrain, ytrain)
etc_predictions = etc_clf_sup.predict(Xtrain)
metrics_report("extra tree classifier", ytrain, etc_predictions, models_performance)

result_df = pd.DataFrame.from_dict(models_performance, orient="index")
result_df

etc_model = ExtraTreesClassifier(random_state=0).fit(Xtrain, ytrain) 
etc_predictions = etc_model.predict(Xtest) 

# model accuracy for X_test   
accuracy_etc = etc_model.score(Xtest, ytest) 
print("Accuracy de l'algorithme etc :", round(accuracy_etc, 3))

conf_mat = confusion_matrix(le.inverse_transform(ytest), le.inverse_transform(etc_model.predict(Xtest)))
print(conf_mat)
#test_labels = le.inverse_transform(y_test)
labels_transform_train = conf_mat_transform(ytrain, etc_model.predict(Xtrain))
labels_transform_test = conf_mat_transform(ytest, etc_model.predict(Xtest))

accuracy_tsne_distilsst2_prods_stem_test = accuracy_score(ytest, labels_transform_test)
accuracy_tsne_distilsst2_prods_stem_train = accuracy_score(ytrain, labels_transform_train)

print()
print("ACCURACY {} (Test): {:.3}".format(type_donnee, accuracy_tsne_distilsst2_prods_stem_test))
print("ACCURACY {} (Train): {:.3}".format(type_donnee, accuracy_tsne_distilsst2_prods_stem_train))
print()

print()
print(classification_report(ytest, labels_transform_test))

df_cm = pd.DataFrame(conf_mat, index = [label for label in list_labels],
                  columns = [i for i in "0123456"])
plt.figure(figsize = (6,4))
sns.heatmap(df_cm, annot=True, cmap="Blues")
plt.xlabel('numero Cluster')
plt.ylabel('categorie Depart')
plt.title(f'matrice confusion {type_donnee}')
plt.show()

# Suppression des waring, info
tf.get_logger().setLevel('ERROR')

use_module_url = 'https://tfhub.dev/google/universal-sentence-encoder/4'

# Variable dont on veut extraire les features
data['product_specifications'] = \
    data['product_specifications'].fillna('').astype(str)
data_prods_orig = data['product_specifications']

type_donnee = 'TSNE_USE_PRODS_ORIG'

# Chargement du modèle
use_model_prods_orig = hub.load(use_module_url)
print('Module %s Chargé' % use_module_url)

vector_use_prods_orig = use_model_prods_orig(data_prods_orig)

# Réduction de dimension tsne
tsne_use_prods_orig = TSNE(verbose=1, perplexity=80, n_iter=5000)
X_proj_tsne_use_prods_orig = \
    tsne_use_prods_orig.fit_transform(vector_use_prods_orig)
# Dataframe pour clustering
df_tsne_use_prods_orig = \
    pd.DataFrame({'VAR1' : X_proj_tsne_use_prods_orig[:, 0],
                  'VAR2' : X_proj_tsne_use_prods_orig[:, 1],
                  'CATEGORIE' : data['category']})

Xtrain, Xtest, ytrain, ytest = train_test_split(df_tsne_use_prods_orig[cols], data.category_label, test_size=0.2, stratify=data.category_label, random_state=1944)

knn_clf_sup = KNeighborsClassifier()
knn_clf_sup.fit(Xtrain, ytrain)
knn_predictions = knn_clf_sup.predict(Xtrain)
metrics_report("knn", ytrain, knn_predictions, models_performance)

svm_clf_sup = OneVsRestClassifier(LinearSVC(), n_jobs=-1)
svm_clf_sup.fit(Xtrain, ytrain)
svm_predictions = svm_clf_sup.predict(Xtrain)
metrics_report("svm", ytrain, svm_predictions, models_performance)

rf_clf_sup = RandomForestClassifier(n_jobs=-1)
rf_clf_sup.fit(Xtrain, ytrain)
rf_predictions = rf_clf_sup.predict(Xtrain)
metrics_report("Random Forest", ytrain, rf_predictions, models_performance)

gnb_clf_sup = OneVsRestClassifier(GaussianNB())
gnb_clf_sup.fit(Xtrain, ytrain)
gnb_predictions = gnb_clf_sup.predict(Xtrain)
metrics_report("Naive Bayes", ytrain, gnb_predictions, models_performance)    

gb_clf_sup = OneVsRestClassifier(GradientBoostingClassifier())
gb_clf_sup.fit(Xtrain, ytrain)
gb_predictions = gb_clf_sup.predict(Xtrain)
metrics_report("Gradient Boosting", ytrain, gb_predictions, models_performance)

etc_clf_sup = ExtraTreesClassifier(random_state=0)
etc_clf_sup.fit(Xtrain, ytrain)
etc_predictions = etc_clf_sup.predict(Xtrain)
metrics_report("extra tree classifier", ytrain, etc_predictions, models_performance)

result_df = pd.DataFrame.from_dict(models_performance, orient="index")
result_df

etc_model = ExtraTreesClassifier(random_state=0).fit(Xtrain, ytrain) 
etc_predictions = etc_model.predict(Xtest) 

# model accuracy for X_test   
accuracy_etc = etc_model.score(Xtest, ytest) 
print("Accuracy de l'algorithme etc :", round(accuracy_etc, 3))

conf_mat = confusion_matrix(le.inverse_transform(ytest), le.inverse_transform(etc_model.predict(Xtest)))
print(conf_mat)
#test_labels = le.inverse_transform(y_test)
labels_transform_train = conf_mat_transform(ytrain, etc_model.predict(Xtrain))
labels_transform_test = conf_mat_transform(ytest, etc_model.predict(Xtest))

accuracy_tsne_use_prods_orig_test = accuracy_score(ytest, labels_transform_test)
accuracy_tsne_use_prods_orig_train = accuracy_score(ytrain, labels_transform_train)

print()
print("ACCURACY {} (Test): {:.3}".format(type_donnee, accuracy_tsne_use_prods_orig_test))
print("ACCURACY {} (Train): {:.3}".format(type_donnee, accuracy_tsne_use_prods_orig_train))
print()

print()
print(classification_report(ytest, labels_transform_test))

df_cm = pd.DataFrame(conf_mat, index = [label for label in list_labels],
                  columns = [i for i in "0123456"])
plt.figure(figsize = (6,4))
sns.heatmap(df_cm, annot=True, cmap="Blues")
plt.xlabel('numero Cluster')
plt.ylabel('categorie Depart')
plt.title(f'matrice confusion {type_donnee}')
plt.show()


#############################################################


# import nbformat
# from nbconvert import PythonExporter
# import os

# def transfer_code_cells(source_notebook, target_notebook):
#     # Charger le notebook source
#     with open(source_notebook, 'r', encoding='utf-8') as f:
#         nb = nbformat.read(f, as_version=4)
    
#     # Initialiser l'exportateur Python de nbconvert
#     exporter = PythonExporter()
    
#     # Créer un répertoire temporaire pour sauvegarder le fichier Python
#     temp_dir = 'temp_dir'
#     os.makedirs(temp_dir, exist_ok=True)
    
#     # Extraire les cellules de code et les sauvegarder dans un fichier Python temporaire
#     code_cells = []
#     for cell in nb['cells']:
#         if cell['cell_type'] == 'code':
#             code_cells.append(cell['source'])
#     with open(os.path.join(temp_dir, 'temp_code.py'), 'w', encoding='utf-8') as f:
#         f.write('\n\n'.join(code_cells))
    
#     # Insérer le fichier Python temporaire dans le notebook cible
#     with open(os.path.join(temp_dir, 'temp_code.py'), 'r', encoding='utf-8') as f:
#         code = f.read()
#         code_cell = nbformat.v4.new_code_cell(source=code)
#         nb['cells'] = [code_cell] + nb['cells']
    
#     # Enregistrer le notebook cible avec les cellules de code ajoutées
#     with open(target_notebook, 'w', encoding='utf-8') as f:
#         nbformat.write(nb, f)
    
#     # Supprimer le répertoire temporaire et son contenu
#     os.system('rm -rf ' + temp_dir)

# # Exemple d'utilisation
# source_notebook = 'P6_01_04_NLP_TE_CLASS_SUPERVISEE.ipynb'
# target_notebook = 'P6_notebook_classif_auto_bien_conso.ipynb'

# transfer_code_cells(source_notebook, target_notebook)

























