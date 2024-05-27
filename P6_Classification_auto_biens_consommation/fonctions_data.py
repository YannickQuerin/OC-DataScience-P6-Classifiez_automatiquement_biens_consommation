#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Chargement des librairies

import datetime
import time
import sys
import os
import string
import json
from datetime import datetime
from IPython.display import display, HTML
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import missingno as msno 
import seaborn as sns
from IPython.display import display
from wordcloud import WordCloud
from statsmodels.graphics.gofplots import qqplot
import re
import plotly.express as px
from sklearn.cluster import MiniBatchKMeans

# Réduction dimension
from sklearn.manifold import TSNE
# Traitement images
from PIL import Image as Image_PIL, ImageOps, ImageFilter
import cv2
from scipy.ndimage import gaussian_filter

import numpy as np
import pandas as pd
import fonctions_data
import texthero as hero
import cprint
import matplotlib as mpl
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import module_name
from module_name import *


# Algorithmes de classification supervisée
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cluster import MiniBatchKMeans

# Feature extraction
import seaborn as sns
import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, \
    HashingVectorizer

# Traitement de text
import texthero as hero
from texthero import preprocessing
import nltk
import gensim
from nltk.corpus import stopwords
from nltk import word_tokenize, FreqDist
from stop_words import get_stop_words
from gensim.models import Word2Vec
import multiprocessing

# Réduction dimension
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn import decomposition
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

# BERT
import transformers
from sentence_transformers import SentenceTransformer

# USE
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import tensorflow_hub as hub
from sklearn.manifold import TSNE

# Clustering
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score
from sklearn.metrics import adjusted_rand_score, silhouette_score, homogeneity_score, recall_score, f1_score

# normalisation, encoding
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# Warnings
import warnings
warnings.filterwarnings('ignore')

# Versions

__version__ = '0.0.0'
print('Version des librairies utilisées :')
print('Python                : ' + sys.version)
print('NumPy                 : ' + np.version.full_version)
print('Pandas                : ' + pd.__version__)
print('Matplotlib            : ' + mpl.__version__)
print('Seaborn               : ' + sns.__version__)
print('Sklearn               : ' + sklearn.__version__)
print('Outils dataframe      : ' + fonctions_data.__version__)
now = datetime.now().isoformat()
print('Lancé le           : ' + now)


# In[ ]:

# --------------------------------------------------------------------
# -- DESCRIPTION DES VARIABLES STATISTIQUES
# --------------------------------------------------------------------


def stat_descriptives(data,list_var):
    '''
    Fonction prenant un dataframe en entrée et retourne les variables, avec ses statistiques
    
    '''

    df = pd.DataFrame(columns=['Variable name', 'Mean', 'Median', 'Skew', 'Kurtosis', \
     'Variance', 'Stdev', 'min','25%','50%','75%','max'])
    
    for col in list_var:
        var_type = data[col].dtypes
        if var_type != 'object':       
            df = df.append(pd.DataFrame([[col, data[col].mean(),data[col].median(), \
            data[col].skew(),data[col].kurtosis(),data[col].var(ddof=0),data[col].std(ddof=0), \
            data[col].min(),data[col].quantile(0.25),data[col].quantile(0.5),data[col].quantile(0.75), \
            data[col].max()]], columns=['Variable name', 'Mean', 'Median', 'Skew', 'Kurtosis', \
            'Variance', 'Stdev', 'min','25%','50%','75%','max']))
    
    df = df.reset_index(drop=True)
    return df

#

def null_var(df, tx_seuil=50):
    null_tx = ((df.isnull().sum() / df.shape[0])*100).sort_values(ascending=False).reset_index()
    null_tx.columns = ['Variable','Taux_de_Null']
    high_null_tx = null_tx[null_tx.Taux_de_Null >= tx_seuil]
    return high_null_tx

#

def fill_var(df, tx_min, tx_max):
    fill_tx = (100 - (df.isnull().sum() / df.shape[0])*100).sort_values(ascending=False).reset_index()
    fill_tx.columns = ['Variable','Taux_de_remplissage']
    high_fill_tx = fill_tx[(fill_tx.Taux_de_remplissage >= tx_min) & (fill_tx.Taux_de_remplissage <= tx_max)]
    return high_fill_tx

#

# --------------------------------------------------------------------
# -- DESCRIPTION DES VARIABLES
# --------------------------------------------------------------------


def  get_nutri_col(data,cols_suppr=False):
        columns_nutri = ['energy_100g',
                             'nutrition_score_fr_100g',
                             'saturated_fat_100g',
                             'sugars_100g',
                             'proteins_100g',
                             'fat_100g',
                             'carbohydrates_100g',
                             'salt_100g',
                             'fiber_100g']
        if cols_suppr:                      
            return data[columns_nutri].drop(cols_suppr,axis=1).columns.to_list()
        else:
            return data[columns_nutri].columns.to_list()
        
#

def rempl_caracteres(data, anc_car, nouv_car):
    """
    Remplacer les caractères avant par les caractères après
    dans le nom des variables du dataframe
    Parameters
    ----------
    @param IN : dataframe : DataFrame, obligatoire
                car_avant : le caractère à remplacer
                car_apres : le caractère de remplacement
    @param OUT : dataframe modifié
    """
    # traces des variables à renommer
    cols_a_renom = data.columns[data.columns.str.contains(
        anc_car)]
    print(f'{len(cols_a_renom)} variables renommées \
          \'{anc_car}\' en \'{nouv_car}\' : \n\n {cols_a_renom.tolist()}')

    return data.columns.str.replace(anc_car, nouv_car)
        

# In[ ]:

def affichage_types_var(df_type, types, type_par_var, graph):
    """ Permet un aperçu du type des variables
    Parameters
    ----------
    @param IN : df_work : dataframe, obligatoire
                types : Si True lance dtypes, obligatoire
                type_par_var : Si True affiche tableau des types de
                               chaque variable, obligatoire
                graph : Si True affiche pieplot de répartition des types
    @param OUT :None.
    """

    if types:
        # 1. Type des variables
        print("-------------------------------------------------------------")
        print("Type de variable pour chacune des variables\n")
        display(df_type.dtypes)

    if type_par_var:
        # 2. Compter les types de variables
        #print("Répartition des types de variable\n")
        values = df_type.dtypes.value_counts()
        nb_tot = values.sum()
        percentage = round((100 * values / nb_tot), 2)
        table = pd.concat([values, percentage], axis=1)
        table.columns = [
            'Nombre par type de variable',
            '% des types de variable']
        display(table[table['Nombre par type de variable'] != 0]
                .sort_values('% des types de variable', ascending=False)
                .style.background_gradient('seismic'))

    if graph:
        # 3. Schéma des types de variable
        # print("\n----------------------------------------------------------")
        #print("Répartition schématique des types de variable \n")
        # Répartition des types de variables
        plt.figure(figsize=(5,5))
        df_type.dtypes.value_counts().plot.pie( autopct='%.0f%%', pctdistance=0.85, radius=1.2)
        #plt.pie(df_type.dtypes.value_counts(), labels = df_type.dtypes.unique(), autopct='%.0f%%', pctdistance=0.85, radius=1.2)
        centre_circle = plt.Circle((0, 0), 0.8, fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        plt.title(label="Repartiton des types de variables",loc="left", fontstyle='italic')
        plt.show()

        
#

def get_val_manq(df_type, pourcentage, affiche_val_manq):
    """Indicateurs sur les variables manquantes
       @param in : df_work dataframe obligatoire
                   pourcentage : boolean si True affiche le nombre heatmap
                   affiche_heatmap : boolean si True affiche la heatmap
       @param out : none
    """

    # 1. Nombre de valeurs manquantes totales
    nb_nan_tot = df_type.isna().sum().sum()
    nb_donnees_tot = np.product(df_type.shape)
    pourc_nan_tot = round((nb_nan_tot / nb_donnees_tot) * 100, 2)
    print(
        f'Valeurs manquantes :{nb_nan_tot} NaN pour {nb_donnees_tot} données ({pourc_nan_tot} %)')

    if pourcentage:
        print("-------------------------------------------------------------")
        print("Nombre et pourcentage de valeurs manquantes par variable\n")
        # 2. Visualisation du nombre et du pourcentage de valeurs manquantes
        # par variable
        values = df_type.isnull().sum()
        percentage = 100 * values / len(df_type)
        table = pd.concat([values, percentage.round(2)], axis=1)
        table.columns = [
            'Nombres de valeurs manquantes',
            '% de valeurs manquantes']
        display(table[table['Nombres de valeurs manquantes'] != 0]
                .sort_values('% de valeurs manquantes', ascending=False))

    if affiche_val_manq:
        print("-------------------------------------------------------------")
        print("Heatmap de visualisation des valeurs manquantes")
        # 3. Heatmap de visualisation des valeurs manquantes
        msno.matrix(df_type)

#

def detail_type_var(data, type_var='all'):
    """
    Retourne la description des variables qualitatives/quantitatives
    ou toutes les variables du dataframe transmis :
    type, nombre de nan, % de nan et desc
    Parameters
    ----------
    @param IN : dataframe : DataFrame, obligatoire
                type_var = 'all' ==> tous les types de variables
                           'cat' ==> variables catégorielles
                           'num' ==> variables quantitative
    @param OUT : dataframe de description des variables
    """
    n_df = data.shape[0]

    if type_var == 'num':
        det_var = data.describe()
    elif type_var == 'cat':
        det_var = data.describe(exclude=[np.number])
    else:
        det_var = data.describe(include='all')
    
    det_type = pd.DataFrame(data[det_var.columns].dtypes, columns=['type']).T
    nb_nan = n_df - det_var.loc['count'].T
    pourcentage_nan = nb_nan * 100 / n_df
    det_nan = pd.DataFrame([nb_nan, pourcentage_nan], index=['nb_nan', '%_nan'])
    det_var = pd.concat([det_type, det_nan, det_var])
    
    return det_var



#
        
# --------------------------------------------------------------------
# -- SUPRESSION VARIABLES POUR UN TAUX DE NAN (%)
# --------------------------------------------------------------------
def clean_nan(data, taux_nan):
#     """
#     Supprime les variables à partir d'un taux en % de nan.
#     Affiche les variables supprimées et les variables conservées
#     ----------
#     @param IN : dataframe : DataFrame, obligatoire
#                 seuil : on conserve toutes les variables dont taux de nan <80%
#                         entre 0 et 100, integer
#     @param OUT : dataframe modifié
#     """
    qty_nan = round((data.isna().sum() / data.shape[0]) * 100, 2)
    cols = data.columns.tolist()
    
    # Conservation seulement des variables avec valeurs manquantes >= 80%
    cols_conservées = qty_nan[qty_nan.values < taux_nan].index.tolist()
    
    cols_suppr = [col for col in cols if col not in cols_conservées]

    data = data[qty_nan[qty_nan.values < taux_nan].index.tolist()]

    print(f'Liste des variables éliminées :\n{cols_suppr}\n')

    print(f'Liste des variables conservées :\n{cols_conservées}')

    return data
    

# In[ ]:

def trace_dispersion_boxplot_qqplot(dataframe, variable, titre, unite):
    """
    Suivi des dipsersions : boxplot
    ----------
    @param IN : dataframe : DataFrame, obligatoire
                variable : colonne dont on veut voir les outliers
                titre :titre pour les graphiques (str)
                unite : unité pour ylabel boxplot (str)
    @param OUT :None
    """
    # Boxplot + qqplot
    fig = plt.figure(figsize=(15, 6))

    data = dataframe[variable]

    ax1 = fig.add_subplot(1, 2, 1)
    box = sns.boxplot(data=data, color='violet', ax=ax1)
    box.set(ylabel=unite)

    plt.grid(False)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2 = qqplot(data,
                 line='r',
                 **{'markersize': 5,
                    'mec': 'k',
                    'color': 'violet'},
                 ax=ax2)
    plt.grid(False)

    fig.suptitle(titre, fontweight='bold', size=14)
    plt.show()

# In[ ]:

def plot_var_filling (df, tx_min, tx_max, graph, axe, col):
    
    if graph:
            filling_var = fill_var(df, tx_min, tx_max)

            font_title = {'family': 'serif',
                          'color':  '#114b98',
                          'weight': 'bold',
                          'size': 18,
                         }
            
        
            sns.set(font_scale=1.2)
            sns.barplot(ax = axe, x="Taux_de_remplissage", y="Variable", data=filling_var, color = col)

    

# In[ ]:

def plot_columns_boxplots(data, columns=[], ncols=2, color="goldenrod"):
    if len(columns) == 0:
        columns = data.columns.values
        
    if len(columns) == 1:
        plt.figure(figsize=(9,3))
        sns.boxplot(x=data[columns[0]], color=color)
        
    else:
        fig, axs = plt.subplots(figsize=(20,20), ncols=ncols, nrows=math.ceil(len(columns) / ncols))
        for index, column in enumerate(columns):
            row_index = math.floor(index / ncols)
            col_index = index % ncols
            sns.boxplot(x=data[column], ax=axs[row_index][col_index], color=color)


# In[ ]:


# --------------------------------------------------------------------
# -- HISTPLOT BOXPLOT QQPLOT
# --------------------------------------------------------------------


def trace_histplot_boxplot_qqplot(dataframe, var):
    """
    Suivi des dipsersions : boxplot et qqplot
    ----------
    @param IN : dataframe : DataFrame, obligatoire
                var : colonne dont on veut voir les outliers
    @param OUT :None
    """
    # Boxplot + qqplot
    fig = plt.figure(figsize=(15, 6))
    fig.subplots_adjust(hspace=0.1, wspace=0.4)
    fig.suptitle('Distribution de ' + str(var), fontsize=16)

    data = dataframe[var]

    ax0 = fig.add_subplot(1, 3, 1)
    sns.histplot(data, kde=True, color='goldenrod', ax=ax0)
    plt.xticks(rotation=60)

    ax1 = fig.add_subplot(1, 3, 2)
    sns.boxplot(data=data, color='goldenrod', ax=ax1)
    plt.grid(False)

    ax2 = fig.add_subplot(1, 3, 3)
    qqplot(data,
           line='r',
           **{'markersize': 5,
              'mec': 'k',
              'color': 'orange'},
           ax=ax2)
    plt.grid(False)
    plt.show()


def trace_multi_histplot_boxplot_qqplot(dataframe, liste_var):
    """
    Suivi des dipsersions : boxplot et qqplot
    ----------
    @param IN : dataframe : DataFrame, obligatoire
                liste_var : colonnes dont on veut voir les outliers
    @param OUT :None
    """
    for col in liste_var:
        trace_histplot_boxplot_qqplot(dataframe, col)


def trace_histplot(
        dataframe,
        variable,
        col,
        titre,
        xlabel,
        xlim_bas,
        xlim_haut,
        ylim_bas,
        ylim_haut,
        kde=True,
        mean_median_mode=True,
        mean_median_zoom=False):
    """
    Histplot pour les variables quantitatives général + histplot
    ----------
    @param IN : dataframe : DataFrame, obligatoire
                variable : colonne dont on veut voir les histplot
                titre : titre du graphique (str)
                xlabel:légende des abscisses
                xlim_bas : limite du zoom supérieur bas(int)
                xlim_haut : limite du zoom supérieur haut(int)
                ylim_bas : limite du zoom inférieur bas(int)
                ylim_haut : limite du zoom inférieur haut(int)
                kde : boolean pour tracer la distribution normale
                mean_median_mode : boolean pour tracer la moyenne, médiane et mode
                mean_median_zoom : boolean pour tracer la moyenne et médiane sur le graphique zoomé
    @param OUT :None
    """
    # Distplot général + zoom
    
    fig = plt.figure(figsize=(15, 8))
    fig.suptitle(titre, fontsize=20, y=1.03)
    data = dataframe[variable]
    
    ax = fig.add_subplot(2, 1, 1)
    ax = sns.boxplot(x=data, color=col)
    ax.set_xlim(xlim_bas, xlim_haut)
    ax.set_ylim(ylim_bas, ylim_haut)
    plt.grid(False)
    plt.xticks([], [])
    

    ax = fig.add_subplot(2, 1, 2)
    ax = sns.histplot(data, kde=kde, color=col)

    if mean_median_mode:
        ax.vlines(data.mean(), *ax.get_ylim(), color='red', ls='-', lw=1.5)
        ax.vlines(
            data.median(),
            *ax.get_ylim(),
            color='green',
            ls='-.',
            lw=1.5)
        ax.vlines(
            data.mode()[0],
            *ax.get_ylim(),
            color='goldenrod',
            ls='--',
            lw=1.5)
    ax.legend(['mode', 'mean', 'median'])
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel('Nombre de produits', fontsize=12)
    plt.grid(False)
    
      
    plt.show()        
        
        

def trace_pieplot(dataframe, variable, titre, legende, liste_colors):
    """
    Suivi des dipsersions : bosplot et qqplot
    ----------
    @param IN : dataframe : DataFrame, obligatoire
                variable : colonne dont on veut voir les outliers (str)
                titre :titre pour les graphiques (str)
                legende : titre de la légende
                liste_colors : liste des couleurs
    @param OUT :None
    """

    plt.figure(figsize=(7, 7))
    plt.title(titre, size=16)
    nb_par_var = dataframe[variable].sort_values().value_counts()
    nb_par_var = nb_par_var.loc[sorted(nb_par_var.index)]
    explode = [0.1]
    for i in range(len(nb_par_var) - 1):
        explode.append(0)
    wedges, texts, autotexts = plt.pie(
        nb_par_var, labels=nb_par_var.index, autopct='%1.1f%%', colors=liste_colors, textprops={
            'fontsize': 16, 'color': 'black', 'backgroundcolor': 'w'}, explode=explode)
    axes = plt.gca()
    axes.legend(
        wedges,
        nb_par_var.index,
        title=legende,
        loc='center right',
        fontsize=14,
        bbox_to_anchor=(
            1,
            0,
            0.5,
            1))
    plt.show()

#    

def aff_eboulis_plot(pca):
    tx_var_exp = pca.explained_variance_ratio_
    scree = tx_var_exp * 100
    plt.bar(np.arange(len(scree)) + 1, scree, color='SteelBlue')
    ax1 = plt.gca()
    ax2 = ax1.twinx()
    ax2.plot(np.arange(len(scree)) + 1, scree.cumsum(), c='green', marker='o')
    ax2.set_ylabel('Taux cumulatif de l\'inertie')
    ax1.set_xlabel('Rang de l\'axe d\'inertie')
    ax1.set_ylabel('Pourcentage d\'inertie')
    for i, p in enumerate(ax1.patches):
        ax1.text(
            p.get_width() /
            5 +
            p.get_x(),
            p.get_height() +
            p.get_y() +
            0.3,
            '{:.0f}%'.format(
                tx_var_exp[i] *
                100),
            fontsize=8,
            color='k')
    plt.title('Eboulis des valeurs propres')
    plt.gcf().set_size_inches(8, 4)
    plt.grid(False)
    plt.show(block=False)

 
    
    
def affiche_cercle(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:
 
            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(7,6))
 
            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])
 
            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:],
                   angles='xy', scale_units='xy', scale=1, color="black")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
             
            # affichage des noms des variables 
            if labels is not None: 
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
             
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)
 
            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
         
            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')
 
            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))
 
            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)

###


def affiche_correlation_circle(pcs, pca, labels, axis_ranks=[(0, 1)],
                               long=6, larg=6):
    ''' Affiche les graphiques de cercle de corrélation de l'ACP pour les
        différents plans factoriels.
        Parameters
        ----------------
        pcs : PCA composants, obligatoire.
        labels : nom des différentes composantes, obligatoire.
        axis_ranks : liste de tuple de plan factoriel (0, 1) par défaut.
        long : longueur de la figure, facultatif (8 par défaut).
        larg : largeur de la figure, facultatif (8 par défaut).
        Returns
        ---------------
        None
    '''
    for i, (d1, d2) in enumerate(axis_ranks):

        fig, axes = plt.subplots(figsize=(long, larg))

        for i, (x_value, y_value) in enumerate(zip(pcs[d1, :], pcs[d2, :])):
            if(x_value > 0.2 or y_value > 0.2):
                plt.plot([0, x_value], [0, y_value], color='k')
                plt.text(x_value, y_value, labels[i], fontsize='14')

        circle = plt.Circle((0, 0), 1, facecolor='none', edgecolor='k')
        axes.set_aspect(1)
        axes.add_artist(circle)

        plt.plot([-1, 1], [0, 0], color='grey', ls='--')
        plt.plot([0, 0], [-1, 1], color='grey', ls='--')

        plt.xlim([-1, 1])
        plt.ylim([-1, 1])

        # nom des axes, avec le pourcentage d'inertie expliqué
        axes.set_xlabel(
            'PC{} ({}%)'.format(
                d1 +
                1,
                round(
                    100 *
                    pca.explained_variance_ratio_[d1],
                    1)),
            fontsize=16)
        axes.set_ylabel(
            'PC{} ({}%)'.format(
                d2 +
                1,
                round(
                    100 *
                    pca.explained_variance_ratio_[d2],
                    1)),
            fontsize=16)
        axes.set_title('PCA correlation circle (PC{} and PC{})'.format(
            d1 + 1, d2 + 1), fontsize=18)

#############################

def affiche_proj_RD(
        dataframe,
        X_projection,
        x_label,
        y_label,
        titre):

    # Constitution du dataframe de travail
    dataframe_work = pd.DataFrame()
    dataframe_work['VAR1'] = X_projection[:, 0]
    dataframe_work['VAR2'] = X_projection[:, 1]
    dataframe_work['CATEGORIE'] = dataframe['category']

    # VIsualisation des 2 premières composantes
    plt.figure(figsize=[25, 15])

    sns.set_palette('Paired')
    sns.scatterplot(x='VAR1', y='VAR2', data=dataframe_work, hue='CATEGORIE',
                    s=100, alpha=1)
    plt.title(titre, fontsize=40)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=34)
    plt.xlabel(x_label, fontsize=34)
    plt.ylabel(y_label, fontsize=34)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.grid(False)
    plt.show()

###############################




# --------------------------------------------------------------------
# -- AFFICHE LE PLAN FACTORIEL
# --------------------------------------------------------------------

def display_factorial_planes(X_projected, n_comp, pca, axis_ranks, labels=None, alpha=1, illustrative_var=None):
    for d1,d2 in axis_ranks:
        if d2 < n_comp:
 
            # initialisation de la figure       
            fig = plt.figure(figsize=(7,6))
        
            # affichage des points
            if illustrative_var is None:
                plt.scatter(X_projected[:, d1], X_projected[:, d2], alpha=alpha)
            else:
                illustrative_var = np.array(illustrative_var)
                for value in np.unique(illustrative_var):
                    selected = np.where(illustrative_var == value)
                    plt.scatter(X_projected[selected, d1], X_projected[selected, d2], alpha=alpha, label=value)
                plt.legend()

            # affichage des labels des points
            if labels is not None:
                for i,(x,y) in enumerate(X_projected[:,[d1,d2]]):
                    plt.text(x, y, labels[i],
                              fontsize='14', ha='center',va='center') 
                
            # détermination des limites du graphique
            boundary = np.max(np.abs(X_projected[:, [d1,d2]])) * 1.1
            plt.xlim([-boundary,boundary])
            plt.ylim([-boundary,boundary])
        
            # affichage des lignes horizontales et verticales
            plt.plot([-100, 100], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-100, 100], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Projection des individus (sur F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)
    
    
# --------------------------------------------------------------------
# -- KDE PLOT graphe
# --------------------------------------------------------------------    
def plot_graph(df_work):
    """Graph densité pour 1 ou plusieurs colonne d'un dataframe
       @param in : df_work dataframe obligatoire
       @param out : none
    """

    plt.figure(figsize=(10, 5))
    axes = plt.axes()

    label_patches = []
    colors = ['Blue', 'SeaGreen', 'Sienna', 'DodgerBlue', 'Purple','Green']

    i = 0
    for col in df_work.columns:
        label = col
        sns.kdeplot(df_work[col], color=colors[i])
        label_patch = mpatches.Patch(
            color=colors[i],
            label=label)
        label_patches.append(label_patch)
        i += 1
    plt.xlabel('')
    plt.legend(
        handles=label_patches,
        bbox_to_anchor=(
            1.05,
            1),
        loc=2,
        borderaxespad=0.,
        facecolor='white')
    plt.grid(False)
    axes.set_facecolor('white')

    plt.show()    
    
    

####

def suppr_ponct(val):
    """
    Suppression de la ponctuation au texte transmis en paramètres.
    Parameters
    ----------
    val : texte dont on veut supprimer la ponctuation
    Returns
    -------
    Texte sans ponctuation
    """
    if isinstance(val, str):  # éviter les nan
        val = val.lower()
        val = re.compile('[éèêë]+').sub("e", val)
        val = re.compile('[àâä]+').sub("a", val)
        val = re.compile('[ùûü]+').sub("u", val)
        val = re.compile('[îï]+').sub("i", val)
        val = re.compile('[ôö]+').sub("o", val)
        return re.compile('[^A-Za-z" "]+').sub("", val)
    return val


####


from wordcloud import WordCloud
from nltk.stem.snowball import EnglishStemmer
import spacy
import nltk
nltk.download('averaged_perceptron_tagger')

def tokenize_clean(document, stopwords=None, keep_tags=None, # ('NN' or 'JJ')
                   lemmatizer=None, stemmer=None):
    # 1 - tokenizing the words in each description
    tokenizer = nltk.RegexpTokenizer(r'[A-Za-z]+')
    li_words = tokenizer.tokenize(document)
    # 2 - lower case
    li_words = [s.lower() for s in li_words]
    # 3 - keep only certain tags
    if keep_tags is not None:
        li_words = [word for word,tag in nltk.pos_tag(li_words)\
            if tag in keep_tags]
    if stopwords is None: stopwords=[]
    # 4 - lemmatizing or stemming
    if lemmatizer is not None:
        lem_doc = lemmatizer(' '.join(li_words))
        li_words = [token.lemma_ for token in lem_doc]
    elif stemmer is not None:
        li_words = [stemmer.stem(s) for s in li_words]
    # 5 - removing stopwords
    li_words = [s for s in li_words if s not in stopwords]

    return li_words



################################


def affiche_wordcloud_hue(dataframe, variable, var_hue, nb_mots):

    liste_hue = dataframe[var_hue].unique().tolist()
    for cat in liste_hue:
        print('Mots les plus fréquents de la catégorie : ' + cat)
        hero.wordcloud(dataframe[dataframe[var_hue] == cat][variable],
                       max_words=nb_mots)
        plt.show()

##################################

##TRACE DIAGRAMME CIRCULAIRE
##################################


def trace_pieplot(dataframe, variable, titre, legende, portion_detachee, taille=(800, 600)):

    # Compte le nombre d'occurrences de chaque valeur de la variable
    nb_par_var = dataframe[variable].value_counts()

    # Crée une liste de pourcentages de détachement
    pull_values = [0] * len(nb_par_var)
    pull_values[portion_detachee] = 0.2  # Pourcentage de détachement pour la portion spécifiée

    # Crée une figure avec les données de la variable
    fig = go.Figure(data=[go.Pie(labels=nb_par_var.index,
                                 values=nb_par_var,
                                 pull=pull_values,
                                 marker=dict(colors=px.colors.qualitative.Plotly,
                                             line=dict(color='#000000', width=1)))])
    
    # Ajoute un titre au diagramme
    fig.update_layout(title=titre)

    # Ajoute une légende
    fig.update_layout(legend_title=legende)

    # Définir la taille de la figure
    fig.update_layout(width=taille[0], height=taille[1])



    

    # Affiche le diagramme
    fig.show()


###########################################
##PIEPLOT DISTRIBUTION TAILLE IMAGES
###########################################



def distribution_variables_plages(dataframe, variable, liste_bins):
    """
    Retourne les plages des pourcentages des valeurs pour le découpage transmis
    Parameters
    ----------
    @param IN : dataframe : DataFrame, obligatoire
                variable : variable à découper obligatoire
                liste_bins: liste des découpages facultatif int ou pintervallindex
    @param OUT : dataframe des plages de nan
    """
    s_gpe_cut = pd.cut(dataframe[variable], bins=liste_bins).value_counts().sort_index()
    df_cut = pd.DataFrame({'Plage': s_gpe_cut.index.astype(str), 'nb_données': s_gpe_cut.values})
    df_cut['%_données'] = (df_cut['nb_données'] * 100) / len(dataframe)

    return df_cut

##########################################
## PIXEL: Traitement d'images
##########################################

def afficher_vwords(image, keypoints):
    '''
    Afficher les 16 premiers Visual Words d'une image.
    Parameters
    ----------
    image : image, obligatoire.
    keypoints : les visual words de l'image, obligatoire.
    Returns
    -------
    None.
    '''
    plt.figure(figsize=(10, 10))
    plt.title('SIFT Visual Words des 16 premiers descripteurs')
    for i, kp in enumerate(keypoints[:16]):
        x, y = int(kp.pt[0]), int(kp.pt[1])
        size = int(kp.size)
        left, upper = max(0, x - size // 2), max(0, y - size // 2)
        right, lower = min(image.shape[1], x + size // 2), min(image.shape[0], y + size // 2)
        cropped = image[upper:lower, left:right]
        plt.subplot(4, 4, i + 1)
        plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    plt.show()

##################################################

# Conversion de phrases, mots dans 'product_category_tree'  en 6 colonnes categories

def str_cleaning(ind, my_str, name_level_cols, max_depth = 6):

    my_str = my_str.replace("[\"", "").replace("\"]", "")
    tab_str = my_str.split(">>")
    size_tab_str = len(tab_str)
    tup_str = tuple([tab_str[i].strip() if i<size_tab_str else "" \
                     for i in np.arange(max_depth) ])
    return tup_str


####################################################################

## Fonction de nettoyage - LEMMATISATION, STEMMMATISATION

def tokenize_clean(document, stopwords=None, keep_tags=None, # ('NN' or 'JJ')
                   lemmatizer=None, stemmer=None):
    # 1 - tokenizing the words in each description
    tokenizer = nltk.RegexpTokenizer(r'[A-Za-z]+')
    li_words = tokenizer.tokenize(document)
    # 2 - lower case
    li_words = [s.lower() for s in li_words]
    # 3 - keep only certain tags
    if keep_tags is not None:
        li_words = [word for word,tag in nltk.pos_tag(li_words)\
            if tag in keep_tags]
    if stopwords is None: stopwords=[]
    # 4 - lemmatizing or stemming
    if lemmatizer is not None:
        lem_doc = lemmatizer(' '.join(li_words))
        li_words = [token.lemma_ for token in lem_doc]
    elif stemmer is not None:
        li_words = [stemmer.stem(s) for s in li_words]
    # 5 - removing stopwords
    li_words = [s for s in li_words if s not in stopwords]

    return li_words

########################################################################

## WORDCLOUD par catégorie

def affiche_wordcloud_hue(dataframe, variable, var_hue, nb_mots):

    liste_hue = dataframe[var_hue].unique().tolist()
    for cat in liste_hue:
        print('Mots les plus fréquents de la catégorie : ' + cat)
        hero.wordcloud(dataframe[dataframe[var_hue] == cat][variable],
                       max_words=nb_mots)
        plt.show()

#######################################################################

# Fonction de vectorisation

def vectorize_text_data(data, vectorizer_method):
    # Instanciation et création du vocabulaire
    vectorizer = vectorizer_method.fit(data)
    
    # Encodage/Vectorisation
    vector_data = vectorizer.transform(data)
    
    # Résumé de la vectorisation
    print(vector_data.shape)
    
    # Transformation en DataFrame
    df_vector_data = pd.DataFrame(vector_data.toarray())
    
    return df_vector_data

#####################################################################

# Calcul du K-Means, et de métriques ARI, Homagénéité

def kmeans(df, df_cat):
    
    df_copy = df.copy()
    
    model = KMeans(n_clusters=7, random_state=0)    
    model = model.fit(df_copy)   
    labels = model.labels_

    
    df_copy['cluster_id'] = labels
    df_copy['categories'] = df_cat
        
    # encoder les categories
    lab_encod = LabelEncoder()
    df_copy["label"] = lab_encod.fit_transform(df_copy["categories"])

    
    # Indice de Rand ajusté :
    ARI = round(adjusted_rand_score(df_copy.cluster_id, df_copy.categories), 4)
    # Homogénéité score
    score_homogeneite = round(homogeneity_score(df_copy.cluster_id, df_copy.categories), 4)

    print("-----------------------------------------")
    print(f" ARI  : {ARI}")
    print(f" Homogénéité  : {score_homogeneite}")
    print("-----------------------------------------")
    
    return df_copy, ARI, score_homogeneite, labels

##########################################################################

## Fonction: Réduction de dimension ACP

def acp(df, threshold=0.99):
    pca = PCA()
    X_pca = pca.fit_transform(df)

    # Conversion en dataframe pandas
    df_pca = pd.DataFrame(X_pca, index=df.index, columns=[f'PC{i}' for i in range(1, X_pca.shape[1] + 1)])
    
    
    # Réduction du nombre de dimensions pour atteindre le seuil de variance spécifié
    n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) > threshold) + 1
    print(f"Nombre de features après réduction ACP pour {threshold * 100}% de variance : {n_components}")

    # Création du DataFrame avec la variance souhaitée
    df_pca_var = df_pca.iloc[:, :n_components]

    return df_pca_var

##########################################################################

## Fonction: Réduction de dimension TNSE

def tsne(df, perplex):
    tsne = TSNE(n_components=2, n_iter=2000, random_state=0, perplexity=perplex)
    X_tsne = tsne.fit_transform(df)
    X_tsne = pd.DataFrame(X_tsne, index=df.index, columns=[f"DIM{c}" for c in range(1, X_tsne.shape[1] + 1)])
    return X_tsne

def tsne_red(df, perplex) :
    
 
    tsne_red = TSNE(n_components=2, n_iter=2000, random_state=0, perplexity=perplex) # init pca fctionne pas avec matrice sparse
    X_tsne = tsne_red.fit_transform(df)
    
    columns = ['DIM' + str(c) for c in range(1, X_tsne.shape[1]+1, 1)]
    X_tsne = pd.DataFrame(X_tsne, index=df.index, columns=columns)
        
  
    return X_tsne


############################################################################

## Fonction: Graphe de projection des clusters

def affiche_projection_par_clusters(X_projected, labels, ARI, homogeneity_score, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 10))

    # Représentation par catégories réelles
    scatter1 = ax1.scatter(X_projected.iloc[:, 0], X_projected.iloc[:, 1], c=X_projected.label, cmap='Set1')
    ax1.legend(handles=scatter1.legend_elements()[0], labels=set(X_projected.categories), title="Catégorie", loc="best", fontsize=12)
    ax1.set_title(f"Représentation des catégories par catégories réelles ({title})\nARI : {ARI}, Homogénéité : {homogeneity_score}", fontsize=20)

    # Représentation par clusters
    scatter2 = ax2.scatter(X_projected.iloc[:, 0], X_projected.iloc[:, 1], c=labels, cmap='Set1')
    ax2.legend(handles=scatter2.legend_elements()[0], labels=set(labels), title="Clusters", loc="best", fontsize=12)
    ax2.set_title("Représentation des catégories par clusters", fontsize=20)

    plt.tight_layout()
    plt.show()

##############################################################################

## Fonctoins: Matrice de confusion, avec le rapport de classification (Accuracy,...)

def conf_mat(y_true, y_pred):
    
    list_cat = data_work['category'].sort_values().unique().tolist()
    conf_mat = confusion_matrix(y_true, y_pred)
    #df_cm = pd.DataFrame(conf_mat, index=list_cat, columns=list_cat)
    df_cm = pd.DataFrame(conf_mat, index = [label for label in list_cat])
    plt.figure(figsize=(6, 4))
    sns.heatmap(df_cm, annot=True, cmap="Blues", fmt='g')

def conf_mat_transform(y_true, y_pred):
    conf_mat = confusion_matrix(y_true, y_pred)
    corresp = np.argmax(conf_mat, axis=0)
    labels = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    labels['y_pred_transform'] = labels['y_pred'].apply(lambda x: corresp[x])
    return labels['y_pred_transform']

def conf_mat_metrics(df):
    cls_labels_transform = conf_mat_transform(df["label"], df["cluster_id"])
    conf_mat = confusion_matrix(df["label"], cls_labels_transform)
    # creation dataframe
    list_labels = df['categories'].sort_values().unique().tolist()
    df_cm = pd.DataFrame(conf_mat, index = [label for label in list_labels])
    # plot   
    plt.figure(figsize = (6,4))
    sns.heatmap(df_cm, annot=True, cmap="Blues", fmt='g')
    # classification report accuracy, precision, recall
    report = classification_report(df["label"], cls_labels_transform, output_dict=True)
    df_report = pd.DataFrame(report).transpose()
    display(df_report)


##################################################################################

## Fonction: Calcul métriques K-Means

def calcul_metrics_kmeans(liste_visual_words, dataframe_metrique, type_algo, nb_cluster):
    # Liste pour stocker les métriques
    metrics = []

    # Recherche des hyperparamètres
    for var_k in nb_cluster:
        # Top début d'exécution
        time_start = time.time()
        # Initialisation de l'algorithme
        cls = MiniBatchKMeans(n_clusters=var_k, random_state=0)
        # Entraînement de l'algorithme
        cls.fit(liste_visual_words)
        # Top fin d'exécution
        time_end = time.time()
        # Calcul la dispersion
        dispersion = cls.inertia_      
        # Durée d'exécution
        time_execution = time_end - time_start      
        # Stockage des métriques dans un dictionnaire
        metrics.append({
            'Type_données': type_algo,
            'Param_k': var_k,
            'dispersion': dispersion,
            'Durée (s)': time_execution
        })

    # Ajout des métriques au dataframe
    dataframe_metrique = dataframe_metrique.append(pd.DataFrame(metrics), ignore_index=True)

    return dataframe_metrique

####################################################################################

## Fonction: Graphe de répartition des clusters

def affiche_repartition_par_clusters(clusters_labels):

    series_client_cluster = pd.Series(clusters_labels).value_counts()
    nb_client = series_client_cluster.sum()
    df_visu_client_cluster = pd.DataFrame(
        {'Clusters': series_client_cluster.index,
         'Nombre': series_client_cluster.values})
    df_visu_client_cluster['%'] = round(
        (df_visu_client_cluster['Nombre']) * 100 / nb_client, 2)
    df_visu_client_cluster = df_visu_client_cluster.sort_values(by='Clusters')
    display(df_visu_client_cluster.style.hide_index())
    
    # Barplot de la distribution
    sns.set_style('white')
    sns.barplot(x='Clusters', y='Nombre',
                data=df_visu_client_cluster, color='purple')
    plt.title('Répartition par clusters', fontsize=16)
    plt.xlabel('Clusters', fontsize=16)
    plt.ylabel('Nombre', fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(False)
    plt.show()

###########################################################################################

## TEXT EMBEDDINGS: Fonctions de vectorisation (WORD2VEC)

def creer_vecteur_moyen_par_mot(data, text_dim, w2v_model):

    vect_moy = np.zeros((text_dim,), dtype='float32')
    num_words = 0.

    for word in data.split():
        if word in w2v_model.wv.key_to_index:
            vect_moy = np.add(vect_moy, w2v_model[word])
            num_words += 1.

    if num_words != 0.:
        vect_moy = np.divide(vect_moy, num_words)

    return vect_moy


def word2vec_vectorisation(data, text_dim, w2v_model):
    '''
    Vectorisation.
    Parameters
    ----------
    data : variable à vectoriser, obligatoire.
    text_dim : taille du vecteur, obligatoire.
    w2v_model : modèle Word2Vec entraîné, obligatoire.
    Returns
    -------
    w2v_vector : les words vectorisés.
    '''
    w2v_vector = np.zeros((data.shape[0], text_dim), dtype='float32')

    for i in range(len(data)):
        w2v_vector[i] = creer_vecteur_moyen_par_mot(
            data[i], text_dim, w2v_model)
        
    return w2v_vector    



# def creer_vecteur_moyen_par_mot(data, text_dim, w2v_model):
#     vect_moy = np.zeros((text_dim,), dtype='float32')
#     num_words = 0.

#     for word in data.split():
#         if word in w2v_model.wv.key_to_index:
#             vect_moy = np.add(vect_moy, w2v_model.wv.get_vector(word))
#             num_words += 1.

#     if num_words != 0.:
#         vect_moy = np.divide(vect_moy, num_words)

#     return vect_moy


# def word2vec_vectorisation(data, text_dim, w2v_model):
#     w2v_vector = np.zeros((len(data), text_dim), dtype='float32')

#     for i in range(len(data)):
#         w2v_vector[i] = creer_vecteur_moyen_par_mot(
#             data[i], text_dim, w2v_model)

#     return w2v_vector

    

###########################################################################################

## Fonction: Graphe de pieplot (Statistique descriptives)

def trace_pieplot(dataframe, variable, titre, legende, portion_detachee, taille=(800, 600)):

    # Compte le nombre d'occurrences de chaque valeur de la variable
    nb_par_var = dataframe[variable].value_counts()

    # Crée une liste de pourcentages de détachement
    pull_values = [0] * len(nb_par_var)
    pull_values[portion_detachee] = 0.2  # Pourcentage de détachement pour la portion spécifiée

    # Crée une figure avec les données de la variable
    fig = go.Figure(data=[go.Pie(labels=nb_par_var.index,
                                 values=nb_par_var,
                                 pull=pull_values,
                                 marker=dict(colors=px.colors.qualitative.Plotly,
                                             line=dict(color='#000000', width=1)))])
    
    # Ajoute un titre au diagramme
    fig.update_layout(title=titre)

    # Ajoute une légende
    fig.update_layout(legend_title=legende)

    # Définir la taille de la figure
    fig.update_layout(width=taille[0], height=taille[1])

    # Affiche le diagramme
    fig.show()


#############################################################################################

## Fonction: Pré-traitement d'images(histogramme de pixels)

def display_image_histopixel(image, titre):
    '''
    Afficher côte à côte l'image et l'histogramme de répartition des pixels.
    Parameters
    ----------
    image : image à afficher, obligatoire.
    Returns
    -------
    None.
    '''
    fig, axs = plt.subplots(1, 3, figsize=(40, 10))

    axs[0].imshow(image, cmap='gray')
    axs[0].set_title(titre, fontsize=30)
    axs[0].tick_params(axis='both', which='major', labelsize=30)

    hist, _ = np.histogram(np.array(image).flatten(), bins=256)
    #axs[1].bar(range(len(hist)), hist)
    axs[1].bar(range(len(hist[0:255])), hist[0:255])
    axs[1].set_title('Histogramme de répartition des pixels', fontsize=30)
    axs[1].set_xlabel('Niveau de gris', fontsize=30)
    axs[1].set_ylabel('Nombre de pixels', fontsize=30)
    axs[1].tick_params(axis='both', which='major', labelsize=30)

    axs[2].hist(np.array(image).flatten(), bins=range(256), cumulative=True)
    axs[2].set_title('Histogramme cumulé des pixels', fontsize=30)
    axs[2].set_xlabel('Niveau de gris', fontsize=24)
    axs[2].set_ylabel('Fréquence cumulée de pixels', fontsize=30)
    axs[2].tick_params(axis='both', which='major', labelsize=30)

    plt.show()

################################################################################################


# fonction de pre-traitement d'image 

def preprocess_image(image):
 
    
    # Variable pour redimensionner images
    dim = (224, 224)
    
    # Nom de l'image
    file_dir = os.path.split(image)
    
    # Chargement de l'image originale
    img = Image_PIL.open(image)

    # Correction de l'exposition PILS (étirement de l'histogramme)
    img = ImageOps.autocontrast(img, 1)

    # Conversion en niveau de gris de l'image
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)

    # Correction du contraste OpenCV CLAHE (égalisation de l'histogramme)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
    img = clahe.apply(img)

    # Réduction du bruit avec l'algorithme Non-local Means Denoising d'OpenCV
    img = cv2.fastNlMeansDenoising(img, None, 5, 7, 21)

    # Redimensionnement en 224 * 224
    img = cv2.resize(np.array(img), dim, interpolation=cv2.INTER_AREA)

    # Sauvegarde de l'image dans le répertoire data/Images_process
    cv2.imwrite('Data/Flipkart/Images_process/' + file_dir[1], img)

    return 'Data/Flipkart/Images_process/' + file_dir[1]


#############################################################################################


def process_image_opencv(image):
    '''
    Correction des images uniquement avec la librairie OpenCV.
    Parameters
    ----------
    image : image localisée dans un répertoire, obligatoire.
    Returns
    -------
    None
    '''
    file_dir = os.path.split(image)
    img = np.array(Image_PIL.open(image))

    # Conversion en niveau de gris de l'image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Suppression du bruit avec un kernel
    img = cv2.GaussianBlur(img, (5, 5), cv2.BORDER_DEFAULT)

    # Egalisation de l'histogramme
    img = cv2.equalizeHist(img)

    # redimension
    dim = (224, 224)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # save image
    cv2.imwrite('Data/Flipkart/Images_process2/' + file_dir[1], img)

    return 'Data/Flipkart/Images_process2/' + file_dir[1]


###################################################################################

# -- Constituer le vecteur des features
# --------------------------------------------------------------------

def constituer_dataframe_vectors(dataframe, variable):
    '''
    Transformer les np.array des histogrammes en autant de colonnes pour
    être utilisable par TSNE et le clustering.
    Parameters
    ----------
    dataframe : dataframe des histogrammes des images.
    variable : nom de la variable contenant les histogrammes à extraire.
    Returns
    -------
    dataframe_vecteurs : le dataframe avec chacune des BOVW de chaque image
    dans une colonne.
    '''
    vectors = np.column_stack(dataframe[variable].values.tolist())
    dataframe_vecteurs = pd.DataFrame(vectors).T

    return dataframe_vecteurs

####################################################################################


def orb_constituter_bovw(dataframe, rep_images, dest_var_bovw, orb_labels,
                         orb_lien_img_des):
    '''
    ORB Charger les images et extraire les features visual words.
    Parameters
    ----------
    dataframe : dataframe de travail, obligatoire.
    rep_images : répertoire de localisation des images dont on veut extraire
                 les features (obligatoire)
    dest_var_bovw : variable des sotckages des BOVW,obligatoire.
    orb_labels : labels.
    orb_lien_img_des : Lien de destination des images.
    Returns
    -------
    None.
    '''
    # Constitution des histogrammes à partir des labels KMeans
    size = orb_labels.shape[0]
    #* orb_labels.shape[1]
    data_images = []
    images = dataframe[rep_images]
    for i in range(len(images)):
        # create a numpy to hold the histogram for each image
        data_images.insert(i, np.zeros((1000, 1)))

    # Sauvegarde des BOVW de chaque image
    dataframe[dest_var_bovw] = ""
    for i in range(size):
        label = orb_labels[i]
        # Get this descriptors image id
        image_id = orb_lien_img_des[i]
        # data_images is a list of the same size as the number of images
        images_data = data_images[image_id]
        # data is a numpy array of size (dictionary_size, 1) filled with zeros
        images_data[label] += 1
        dataframe[dest_var_bovw][image_id] = images_data.flatten()


###################################################################################


def afficher_vwords(image, keypoints):
    '''
    Afficher les 16 premiers Visual Words d'une image.
    Parameters
    ----------
    image : image, obligatoire.
    keypoints : les visual words de l'image, obligatoire.
    Returns
    -------
    None.
    '''
    plt.figure(figsize=(10, 10))
    plt.title('SIFT Visual Words des 16 premiers descripteurs')
    for i, kp in enumerate(keypoints[:16]):
        x, y = int(kp.pt[0]), int(kp.pt[1])
        size = int(kp.size)
        left, upper = max(0, x - size // 2), max(0, y - size // 2)
        right, lower = min(image.shape[1], x + size // 2), min(image.shape[0], y + size // 2)
        cropped = image[upper:lower, left:right]
        plt.subplot(4, 4, i + 1)
        plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
    plt.show()


#####################################################################################

## Fonction: Chargement des images dans un dossier

def charger_images_repertoire(repertoire, nombre_images=350):
    fichiers = os.listdir(repertoire)[:nombre_images]
    images = {filename: cv2.imread(os.path.join(repertoire, filename), 0) for filename in fichiers}
    return images

######################################################################################


def sift_extraire_features(images):
    '''
    Extraire les descripteurs et keypoints avec SIFT.
    Parameters
    ----------
    images : les images dont on veut extraire les descripteurs et centres
             d'intérêt, obligatoire.
    Returns
    -------
    list des descripteurs et des vecteurs SIFT.
    '''
    sift = cv2.xfeatures2d.SIFT_create(nOctaveLayers=3, contrastThreshold=0.02, edgeThreshold=10, sigma=1.6)

    descriptor_list = []
    sift_vectors = {}
    for key, value in images.items():
        if value is not None and len(value.shape) == 2:
            kp, des = sift.detectAndCompute(value, None)
            if des is not None:
                descriptor_list.extend(des)
                sift_vectors[key] = [des]
            else:
                descriptor_list.append(np.zeros((128,)))
                sift_vectors[key] = [np.zeros((128,))]
        else:
            print(f"Image {key} invalide.")
    return [descriptor_list, sift_vectors]


##########################################################################################


def calcul_metrics_kmeans(liste_visual_words, dataframe_metrique, type_algo,
                           nb_cluster):
    '''
    Calcul des métriques de KMeans en fonction de différents paramètres.
    Parameters
    ----------
    liste_visual_words : données, obligatoire.
    dataframe_metrique : dataframe de sauvegarde des résultats, obligatoire.
    type_algo : string type d'algorithme SIFT, ORB..., obligatoire.
    random_seed : nombre aléatoire pour la reproductibilité, obligatoire.
    nb_cluster : liste du nombre de clusters à
    Returns
    -------
    dataframe_metrique : résultat des métriques
    '''
    # Cette fonction permet de calculer le nombre de clusters le plus
    # optimal pour notre analyse : la dispersion

    dispersion = []
    donnees = []
    temps = []

    result_k = []

    # Hyperparametre tuning

    # Recherche des hyperparamètres
    for var_k in nb_cluster:

        # Top début d'exécution
        time_start = time.time()

        # Initialisation de l'algorithme
        cls = KMeans(n_clusters=var_k,
                     init='k-means++',
                     random_state=0)

        # Entraînement de l'algorithme
        cls.fit(liste_visual_words)

        # Prédictions
        # preds = cls.predict(liste_visual_words)

        # Top fin d'exécution
        time_end = time.time()

        # Calcul du score de coefficient de silhouette
        # silh = silhouette_score(liste_visual_words, preds)
        # Calcul la dispersion
        disp = cls.inertia_
        # Calcul de l'indice davies-bouldin
        # db = davies_bouldin_score(liste_visual_words, preds)
        # Durée d'exécution
        time_execution = time_end - time_start
        display('Paramètre ' + str(var_k) + ' terminé')

        # silhouette.append(silh)
        dispersion.append(disp)
        # davies_bouldin.append(db)
        donnees.append(type_algo)
        temps.append(time_execution)

        result_k.append(var_k)

    dataframe_metrique = dataframe_metrique.append(pd.DataFrame({
        'Type_données': donnees,
        'Param_k': result_k,
        # 'coef_silh': silhouette,
        'dispersion': dispersion,
        # 'davies_bouldin': davies_bouldin,
        'Durée (s)': temps
    }), ignore_index=True)

    return dataframe_metrique

#################################################################################

## Fonction: Construction histogramme par image

def image_class(all_bovw, centers):
    dict_feature = {}
    for key, value in all_bovw.items():
        category = []
        for img in value:
            histogram = np.zeros(len(centers))
            for each_feature in img:
                ind = np.where(np.array(centers) == each_feature)[0]
                if len(ind) > 0:
                    histogram[ind[0]] += 1
            category.append(histogram)
        dict_feature[key] = category
    return dict_feature


###################################################################################

## Fonction: extraction des features - algorithme ORB

def orb_extract_features(rep_images):
    '''
    ORB Charger les images et extraire les features visual words.
    Parameters
    ----------
    rep_images : répertoire de localisation des images dont on veut extraire
                 les features (obligatoire)
    Returns
    -------
    descriptors : liste des descripteurs.
    lien_img_des : lien entre image et descripteur.
    '''
    # Liste des descripteurs
    descriptor_list = []

    # Detecteur de feature ORB
    orb = cv2.ORB_create(nfeatures=1500)

    # Répertoire des images
    images = rep_images

    # Conserver le lien entre l'image et le descripteur
    lien_img_des = []

    # Pour toutes les images
    for i in range(len(images)):
        im = images[i]
        img = cv2.imread(im, cv2.IMREAD_GRAYSCALE)
        kp, des = orb.detectAndCompute(img, None)

        if des is not None:  # Vérifier si des n'est pas None
            descriptor_list.append(des)

            # Lien image - descripteur
            for j in range(len(des)):
                lien_img_des.append(i)

    descriptors = np.array(descriptor_list[0]) if descriptor_list else np.zeros((0, 32), dtype=np.float32)
    for descriptor in descriptor_list[1:]:
        if descriptor is not None:
            descriptors = np.vstack((descriptors, descriptor))

    return descriptors, lien_img_des

#######################################################################################



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

###########################################################################################


def conf_mat_transform(y_true,y_pred) :
    conf_mat = confusion_matrix(y_true,y_pred)
    
    corresp = np.argmax(conf_mat, axis=0)
    print ("Correspondance des clusters : ", corresp)
    labels = pd.Series(y_true, name="y_true").to_frame()
    labels['y_pred'] = y_pred
    labels['y_pred_transform'] = labels['y_pred'].apply(lambda x : corresp[x]) 
    
    return labels['y_pred_transform']

##############################################################################################

## Fonction: evaluation du modèle à partir des données génératrices

def evaluate_model(model, test_generator):
    # Obtenir les prédictions pour l'ensemble de test
    y_pred = model.predict(test_generator)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Convertir les étiquettes en vecteurs de classe
    y_true = test_generator.classes

    # Obtenir les noms de classe pour les étiquettes réelles
    class_names = list(test_generator.class_indices.keys())

    # Calculer la matrice de confusion
    confusion_mtx = confusion_matrix(y_true, y_pred_classes)

    # Afficher la matrice de confusion sous forme graphique
    sns.heatmap(confusion_mtx, xticklabels=class_names, yticklabels=class_names, cmap='Blues', annot=True, fmt='d')
    plt.title('Matrice de confusion')
    plt.xlabel('Prédictions')
    plt.ylabel('Valeurs réelles')
    plt.show()

    # Afficher le rapport de classification
    print(classification_report(y_true, y_pred_classes, target_names=class_names))


###############################################################################################





############################################################################################



















        
