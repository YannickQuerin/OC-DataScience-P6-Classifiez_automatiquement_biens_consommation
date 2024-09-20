# Classification Automatique des Biens de Consommation

Ce projet vise à classifier automatiquement des biens de consommation à partir de données textuelles et d'images, afin d'améliorer la précision des recommandations de produits sur une plateforme de place de marché.

## Table des Matières

1. [Contexte et Données](#contexte-et-données)
2. [Traitement des Données Textuelles](#traitement-des-données-textuelles)
3. [Traitement des Données Images](#traitement-des-données-images)
4. [Combinaison des Données Textuelles et Images](#combinaison-des-données-textuelles-et-images)
5. [Conclusion](#conclusion)

---

## Contexte et Données

### Contexte du Projet
L'objectif de ce projet est de développer un moteur de recommandation automatique basé sur les descriptions et les images des produits. Ce moteur permettra à une plateforme de commerce en ligne de classifier les articles dans les bonnes catégories de manière automatisée.

**Objectifs du projet :**
- Améliorer la précision des recommandations de produits.
- Garantir une attribution correcte des catégories aux articles.
- Faciliter l'automatisation de la classification pour l'entreprise.

### Jeu de Données

- **Sources :** Le dataset provient du site Flipkart (Inde).
- **Taille du dataset :** 1050 lignes et 15 variables.
- **Types de données :**
  - Données textuelles : descriptions de produits, noms de marques, spécifications produits.
  - Données images : images des produits.

- **Variable cible :** `product_category_tree`, contenant des catégories hiérarchiques comme "Computers >> Laptop Accessories >> USB Gadgets".

---

## Traitement des Données Textuelles

Le traitement des données textuelles repose sur des techniques de **NLP (Natural Language Processing)**, incluant :

### Préprocessing des Données Textuelles
1. **Tokenisation**
2. **Normalisation**
3. **Lemmatisation**
4. **Racinisation**

### Extraction de Features
- **Bag of Words**
- **Word Embeddings** (Word2Vec, BERT, Sentence-Transformers, etc.)
- **CountVectorizer** et **TfidfVectorizer**

### Réduction de Dimensionnalité
- Utilisation de **ACP (Analyse en Composantes Principales)** et **TSNE** pour réduire la dimensionnalité des vecteurs de texte.

### Classification Non Supervisée (KMeans)
- Évaluation avec des métriques telles que **ARI (Adjusted Rand Index)** et **Accuracy** pour mesurer les performances.

### Classification Supervisée
- Evaluation de la précision sur les ensembles d'entraînement et de test.

---

## Traitement des Données Images

Le traitement des données images s'appuie sur les techniques de vision par ordinateur.

### Préprocessing des Images
1. **Redimensionnement** à 224x224 pixels.
2. **Conversion en niveaux de gris.**
3. **Correction de l'exposition et du contraste.**

### Extraction de Features
- Utilisation de techniques comme **SIFT**, **ORB** et des réseaux de neurones convolutifs **CNN** (ex. : **VGG16**, **VGG19**, **InceptionV3**).

### Classification Non Supervisée (KMeans)
- Analyse et interprétation des clusters avec des techniques de réduction de dimensionnalité (ACP + TSNE).

### Classification Supervisée
- Utilisation de l'apprentissage supervisé pour améliorer les performances des modèles, avec une évaluation de la précision sur l'entraînement et les tests.

---

## Combinaison des Données Textuelles et Images

L'approche combine des embeddings textuels et les features extraites des images pour améliorer la performance de la classification.

- **Text Embeddings (USE 5)** et **InceptionV3** pour les images.
- Résultats :
  - ARI combiné : 0.65
  - Accuracy combinée : 0.85

---

## Conclusion

L'étude a démontré la faisabilité d'une approche combinant traitement de texte et traitement d'images pour classifier automatiquement des biens de consommation.

- **Meilleures Performances :**
  - **Modèle Textuel (USE 5)** : ARI = 0.70, Accuracy = 0.85.
  - **Modèle Image (InceptionV3)** : ARI = 0.53, Accuracy = 0.75.
  - **Combinaison Textes + Images** : ARI = 0.65, Accuracy = 0.85.

Cette approche améliore la précision des recommandations et garantit une meilleure classification des produits.
