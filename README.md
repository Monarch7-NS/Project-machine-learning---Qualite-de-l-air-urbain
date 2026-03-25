# Qualité de l'Air & Séparation de Sources de Pollution

Bienvenue sur le dépôt de mon projet d'analyse non supervisée (Module DATA 832).

L'objectif de ce projet est de plonger dans des données brutes de capteurs urbains pour tenter de **démêler des sources de pollution invisibles** — trafic routier, chauffage, réactions photochimiques — en utilisant exclusivement des méthodes d'apprentissage non supervisé.

Plutôt que d'empiler des algorithmes complexes, j'ai fait le choix de privilégier la **rigueur du traitement des données** (gestion intelligente des valeurs manquantes) et **l'interprétation physique** des résultats mathématiques. Parce qu'un bon modèle qu'on ne comprend pas, ça ne vaut pas grand chose.

---

## Le jeu de données

Les données proviennent de l'**UCI Machine Learning Repository** ([Air Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Air+Quality)).

| Caractéristique | Détail |
|---|---|
| Volume | 9 357 mesures horaires |
| Période | Mars 2004 – Avril 2005 |
| Capteurs | CO, NOx, NO2, Benzène (C6H6), Ozone, + 5 capteurs chimiques bruts |
| Format | CSV, séparateur `;`, décimale `,` |

**Le vrai défi : le nettoyage.** Le dataset arrive avec des valeurs manquantes encodées en `-200`, une colonne NMHC vide à 90%, et des colonnes fantômes vides en fin de fichier. J'ai choisi une **interpolation temporelle** plutôt qu'une imputation par moyenne, parce que la qualité de l'air évolue de façon continue — sauter d'une mesure à l'autre sans tenir compte du temps, c'est physiquement absurde.

---

## Structure du projet

```
mini_project_ml2/
│
├── python/
│   ├── ML.py              # script principal — data → modèles → graphiques
│   └── what.py            # version alternative / brouillon
│
├── figures/               # les 14 graphiques générés automatiquement
│   ├── 01_distributions.png
│   ├── 02_correlation_matrix.png
│   ├── 03_saisonnalite.png
│   ├── 04_pca.png
│   ├── 05_biplot_pca.png
│   ├── 06_ica_vs_pca.png
│   ├── 07_scatter_pca_ica.png
│   ├── 08_kmeans_selection.png
│   ├── 09_confusion_matrix.png
│   ├── 10_knn_distance.png
│   ├── 11_scatter_clusters.png
│   ├── 12_timeseries_annotated.png
│   ├── 13_heatmap_pollutants_ica.png
│   └── 14_anomalies_temporelles.png
│
└── README.md
```

---

## Pipeline d'analyse

Le script suit une progression logique en 5 étapes :

### 0. Chargement & nettoyage
Téléchargement direct depuis UCI, suppression des artefacts, remplacement des `-200` par `NaN`, création d'un index `DatetimeIndex` propre, interpolation temporelle.

### 1. Exploration & saisonnalité
Distributions de chaque polluant, matrice de corrélation, profils horaires et mensuels. Cette étape sert surtout à se faire une intuition sur les données avant de lancer quoi que ce soit.

### 2. ACP — Réduction de dimension
Standardisation, puis ACP complète. On regarde le scree plot pour choisir le nombre de composantes qui expliquent 90% de la variance. Un biplot PC1 vs PC2 permet de voir quelles variables "tirent" dans quelle direction.

### 3. ICA — Séparation de sources
FastICA avec le même nombre de composantes que l'ACP. L'idée : là où l'ACP maximise la variance, l'ICA maximise l'indépendance statistique des composantes. En pratique, ça donne des "sources" qui correspondent mieux à des phénomènes physiques réels. On vérifie via le kurtosis que les signaux sont bien non-gaussiens (condition nécessaire pour que l'ICA fonctionne).

### 4. Clustering
- **K-Means** : méthode du coude + score de silhouette pour choisir k. On retient k=3 (air bon / moyen / mauvais). L'ARI par rapport à des pseudo-labels construits manuellement sert à calibrer l'interprétation.
- **DBSCAN** : détection d'anomalies dans l'espace ICA. On estime `eps` de façon adaptative via les distances k-NN (percentile 97) pour éviter de fixer un seuil à la main.

### 5. Visualisations finales
Superposition des clusters K-Means et des anomalies DBSCAN sur la série temporelle de CO, heatmap complète de la mixing matrix ICA, et distribution temporelle des anomalies par heure/mois.

---

## Principaux résultats

**ACP vs ICA : deux façons de décomposer la réalité**

L'ACP résume 90% de l'information en 3 composantes, mais ces axes n'ont pas de sens physique direct — ce sont juste des directions de variance maximale. L'ICA va plus loin : en cherchant l'indépendance, elle isole des sources qui ressemblent à de vrais phénomènes (un axe "trafic diesel", un axe "photochimie ozone"). Le kurtosis des composantes ICA est systématiquement plus élevé que celui des variables brutes — signe que la séparation fonctionne.

**Clustering K-Means (k=3)**

L'ARI de ~0.16 face aux pseudo-labels peut sembler décevant, mais c'est en réalité attendu : la pollution est un continuum, pas trois états discrets. Les frontières que K-Means impose sont artificielles par nature. Le score de silhouette confirme que k=3 reste le meilleur compromis parmi ceux testés.

**Anomalies DBSCAN : une découverte inattendue**

DBSCAN isole environ 4.7% des mesures comme "bruit". Ce qui est intéressant, c'est que ces anomalies ne sont pas aléatoires : elles ont des concentrations de NOx extrêmement élevées et se produisent majoritairement **vers 4h du matin en décembre**. C'est la signature de l'**inversion thermique hivernale** — quand une couche d'air chaud piège les polluants près du sol la nuit, en l'absence de vent et de circulation. L'algorithme a trouvé ça tout seul, sans qu'on lui dise quoi chercher.

---

## Installation et exécution

**Prérequis :** Python 3.8+

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

```bash
python python/ML.py
```

Le script télécharge le dataset automatiquement depuis UCI et génère les 14 figures dans le répertoire courant. Pas besoin de télécharger quoi que ce soit manuellement.

---

## Notes personnelles

Ce projet m'a appris une chose surtout : **passer 80% du temps sur les données, c'est pas une perte de temps, c'est le travail**. L'interpolation temporelle plutôt que la moyenne, le choix d'exclure NMHC plutôt que de l'imputer à l'aveugle, le eps adaptatif pour DBSCAN — ce sont ces décisions-là qui font la différence sur la qualité des résultats, bien plus que le choix de l'algorithme final.

L'ICA m'a particulièrement surpris. Je m'attendais à ce que ce soit une boîte noire de plus. En pratique, les composantes indépendantes ont une cohérence physique réelle qu'on ne retrouve pas avec l'ACP — c'est rassurant de voir que les maths et la physique s'alignent.
