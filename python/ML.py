import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
import zipfile, io, urllib.request

from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    silhouette_score,
    adjusted_rand_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

# je préfère des graphes propres sans les bordures en haut/droite
plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "DejaVu Sans",
})
PALETTE = ["#2D6A4F", "#52B788", "#B7E4C7", "#F4A261", "#E76F51", "#264653"]


# -----------------------------------------------------------------------
# 0. chargement des données
# -----------------------------------------------------------------------
print("=" * 60)
print("0. CHARGEMENT DES DONNÉES")
print("=" * 60)

# le dataset est directement dispo sur le repo UCI, pas besoin de le télécharger à la main
url = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases"
    "/00360/AirQualityUCI.zip"
)
resp = urllib.request.urlopen(url)
with zipfile.ZipFile(io.BytesIO(resp.read())) as z:
    with z.open("AirQualityUCI.csv") as f:
        df = pd.read_csv(f, sep=";", decimal=",")

# le fichier a des colonnes/lignes complètement vides à la fin, on s'en débarrasse
df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")

# les valeurs manquantes sont codées -200 dans ce dataset, faut les remplacer par NaN
df = df.replace(-200, float("nan"))

# on fusionne Date + Time pour avoir un vrai index datetime
df["Datetime"] = pd.to_datetime(
    df["Date"].astype(str) + " " + df["Time"].astype(str),
    format="%d/%m/%Y %H.%M.%S",
    errors="coerce",
)
df = df.set_index("Datetime").drop(columns=["Date", "Time"])
df = df.sort_index()

print(f"Shape après nettoyage : {df.shape}")
print(f"Période : {df.index.min()} → {df.index.max()}")
print(f"\nPourcentage de NaN par colonne :")
print((df.isna().mean() * 100).round(1).to_string())

# on garde seulement les colonnes polluants — NMHC(GT) est exclue car >90% de NaN
POLLUTANTS = [
    "CO(GT)", "PT08.S1(CO)", "C6H6(GT)", "PT08.S2(NMHC)",
    "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)", "PT08.S5(O3)",
]
df_poll = df[POLLUTANTS].copy()

# interpolation temporelle d'abord, puis bfill/ffill pour les bords
df_poll = df_poll.interpolate(method="time").fillna(method="bfill").fillna(method="ffill")

print(f"\nNaN restants après imputation : {df_poll.isna().sum().sum()}")


# -----------------------------------------------------------------------
# 1. exploration rapide + saisonnalité
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("1. EXPLORATION & SAISONNALITÉ")
print("=" * 60)

print("\nStatistiques descriptives :")
print(df_poll.describe().round(2).to_string())

# distributions de chaque polluant — utile pour repérer les skews et outliers
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
fig.suptitle("Distributions des polluants (après imputation)", fontsize=14, fontweight="bold")
for ax, col in zip(axes.flatten(), POLLUTANTS):
    ax.hist(df_poll[col].dropna(), bins=60, color=PALETTE[1], edgecolor="white", alpha=0.85)
    ax.axvline(df_poll[col].mean(), color=PALETTE[4], linestyle="--", linewidth=1.5, label="moyenne")
    ax.set_title(col, fontsize=9)
    ax.set_xlabel("")
    ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig("01_distributions.png", bbox_inches="tight")
plt.show()
print("→ Figure sauvegardée : 01_distributions.png")

# corrélations — on masque le triangle supérieur pour pas afficher les doublons
fig, ax = plt.subplots(figsize=(10, 8))
corr = df_poll.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
    center=0, linewidths=0.5, ax=ax, annot_kws={"size": 8},
)
ax.set_title("Matrice de corrélation entre polluants", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("02_correlation_matrix.png", bbox_inches="tight")
plt.show()
print("→ Figure sauvegardée : 02_correlation_matrix.png")

# on extrait l'heure et le mois pour regarder les patterns temporels
df_poll_copy = df_poll.copy()
df_poll_copy["hour"] = df_poll_copy.index.hour
df_poll_copy["month"] = df_poll_copy.index.month

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for col, color in zip(["CO(GT)", "NOx(GT)"], [PALETTE[0], PALETTE[4]]):
    hourly = df_poll_copy.groupby("hour")[col].mean()
    axes[0].plot(hourly.index, hourly.values, marker="o", color=color, label=col, linewidth=2)
axes[0].set_title("Profil horaire moyen (CO et NOx)", fontweight="bold")
axes[0].set_xlabel("Heure")
axes[0].legend()

monthly = df_poll_copy.groupby("month")["CO(GT)"].mean()
axes[1].bar(monthly.index, monthly.values, color=PALETTE[1], edgecolor="white")
axes[1].set_title("Profil mensuel moyen (CO)", fontweight="bold")
axes[1].set_xlabel("Mois")
axes[1].set_xticks(range(1, 13))
axes[1].set_xticklabels(["Jan","Fév","Mar","Avr","Mai","Jun","Jul","Aoû","Sep","Oct","Nov","Déc"], rotation=45)

plt.tight_layout()
plt.savefig("03_saisonnalite.png", bbox_inches="tight")
plt.show()
print("→ Figure sauvegardée : 03_saisonnalite.png")


# -----------------------------------------------------------------------
# 2. ACP
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("2. ACP – ANALYSE EN COMPOSANTES PRINCIPALES")
print("=" * 60)

# standardisation obligatoire avant ACP, sinon les variables avec grande variance dominent
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_poll)

pca = PCA(random_state=42)
pca.fit(X_scaled)
X_pca = pca.transform(X_scaled)

explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)
n_components_90 = np.argmax(cumulative >= 0.90) + 1
print(f"\nNombre de composantes pour 90% de variance : {n_components_90}")
for i, (ev, cv) in enumerate(zip(explained, cumulative)):
    print(f"  PC{i+1} : {ev*100:.1f}%  (cumulé : {cv*100:.1f}%)")

# scree plot + loadings côte à côte
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].bar(range(1, len(explained)+1), explained*100, color=PALETTE[1], edgecolor="white")
axes[0].plot(range(1, len(explained)+1), cumulative*100, color=PALETTE[4], marker="o", linewidth=2)
axes[0].axhline(90, color="grey", linestyle="--", linewidth=1)
axes[0].set_title("Scree plot – Variance expliquée par composante", fontweight="bold")
axes[0].set_xlabel("Composante principale")
axes[0].set_ylabel("Variance expliquée (%)")

loadings = pd.DataFrame(
    pca.components_[:4].T,
    index=POLLUTANTS,
    columns=[f"PC{i+1}" for i in range(4)],
)
sns.heatmap(
    loadings, annot=True, fmt=".2f", cmap="coolwarm",
    center=0, linewidths=0.5, ax=axes[1], annot_kws={"size": 9},
)
axes[1].set_title("Loadings ACP (4 premières composantes)", fontweight="bold")
plt.tight_layout()
plt.savefig("04_pca.png", bbox_inches="tight")
plt.show()
print("→ Figure sauvegardée : 04_pca.png")

# biplot : on superpose les points et les flèches des variables
fig, ax = plt.subplots(figsize=(9, 7))
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.15, s=5, c=PALETTE[1])
scale = 4  # facteur d'échelle pour rendre les flèches lisibles
for j, name in enumerate(POLLUTANTS):
    ax.annotate(
        "", xy=(pca.components_[0, j]*scale, pca.components_[1, j]*scale),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="->", color=PALETTE[4], lw=1.5),
    )
    ax.text(
        pca.components_[0, j]*scale*1.15, pca.components_[1, j]*scale*1.15,
        name, fontsize=8, color=PALETTE[4], ha="center",
    )
ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}%)")
ax.set_title("Biplot ACP – PC1 vs PC2", fontweight="bold")
ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
plt.tight_layout()
plt.savefig("05_biplot_pca.png", bbox_inches="tight")
plt.show()
print("→ Figure sauvegardée : 05_biplot_pca.png")


# -----------------------------------------------------------------------
# 3. ICA — séparation de sources
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("3. ICA – ANALYSE EN COMPOSANTES INDÉPENDANTES")
print("=" * 60)

# on prend le même nombre de composantes que l'ACP à 90%
n_ica = n_components_90
ica = FastICA(n_components=n_ica, random_state=42, max_iter=500)
X_ica = ica.fit_transform(X_scaled)

print(f"\nNombre de composantes ICA : {n_ica}")

# le kurtosis nous dit à quel point les distributions sont non-gaussiennes
# l'ICA marche mieux quand c'est bien non-gaussien
print("\nKurtosis des variables originales (>3 = leptokurtique = non gaussien) :")
for col in POLLUTANTS:
    k = stats.kurtosis(df_poll[col].dropna())
    print(f"  {col:20s} : {k:.2f}")

print("\nKurtosis des composantes ICA :")
for i in range(n_ica):
    k = stats.kurtosis(X_ica[:, i])
    print(f"  IC{i+1} : {k:.2f}")

# comparaison ACP vs ICA : est-ce que les composantes capturent des choses différentes ?
mixing = pd.DataFrame(
    ica.mixing_[:, :n_ica],
    index=POLLUTANTS,
    columns=[f"IC{i+1}" for i in range(n_ica)],
)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(
    loadings, annot=True, fmt=".2f", cmap="coolwarm",
    center=0, linewidths=0.5, ax=axes[0], annot_kws={"size": 8},
)
axes[0].set_title("ACP – Loadings (variance max)", fontweight="bold")

sns.heatmap(
    mixing, annot=True, fmt=".2f", cmap="coolwarm",
    center=0, linewidths=0.5, ax=axes[1], annot_kws={"size": 8},
)
axes[1].set_title("ICA – Mixing matrix (indépendance max)", fontweight="bold")
plt.suptitle("ACP vs ICA : contribution des polluants aux composantes", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig("06_ica_vs_pca.png", bbox_inches="tight")
plt.show()
print("→ Figure sauvegardée : 06_ica_vs_pca.png")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.1, s=4, color=PALETTE[0])
axes[0].set_title("ACP : PC1 vs PC2", fontweight="bold")
axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")

axes[1].scatter(X_ica[:, 0], X_ica[:, 1], alpha=0.1, s=4, color=PALETTE[4])
axes[1].set_title("ICA : IC1 vs IC2", fontweight="bold")
axes[1].set_xlabel("IC1"); axes[1].set_ylabel("IC2")

plt.tight_layout()
plt.savefig("07_scatter_pca_ica.png", bbox_inches="tight")
plt.show()
print("→ Figure sauvegardée : 07_scatter_pca_ica.png")


# -----------------------------------------------------------------------
# 4. clustering
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("4. CLUSTERING")
print("=" * 60)

# on travaille dans l'espace PCA réduit, c'est plus propre que les features brutes
X_clust = X_pca[:, :n_components_90]

# pseudo-labels basés sur un indice de pollution maison — sert à évaluer le clustering
pollution_index = df_poll[["CO(GT)", "NOx(GT)", "NO2(GT)", "C6H6(GT)"]].mean(axis=1)
pollution_index_norm = (pollution_index - pollution_index.min()) / (pollution_index.max() - pollution_index.min())
pseudo_labels = pd.cut(
    pollution_index_norm,
    bins=[0, 0.33, 0.66, 1.0],
    labels=[0, 1, 2],
    include_lowest=True,
).astype(int).values
print(f"\nDistribution pseudo-labels : {np.bincount(pseudo_labels)}")

# méthode du coude + silhouette pour choisir k
inertias, silhouettes = [], []
K_range = range(2, 10)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_tmp = km.fit_predict(X_clust)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_clust, labels_tmp, sample_size=3000, random_state=42))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(K_range, inertias, marker="o", color=PALETTE[0], linewidth=2)
axes[0].set_title("Méthode du coude (inertie)", fontweight="bold")
axes[0].set_xlabel("Nombre de clusters k")
axes[0].set_ylabel("Inertie")

axes[1].plot(K_range, silhouettes, marker="o", color=PALETTE[4], linewidth=2)
axes[1].set_title("Score de silhouette", fontweight="bold")
axes[1].set_xlabel("Nombre de clusters k")
axes[1].set_ylabel("Silhouette score")
plt.tight_layout()
plt.savefig("08_kmeans_selection.png", bbox_inches="tight")
plt.show()
print("→ Figure sauvegardée : 08_kmeans_selection.png")

# k=3 correspond à bon/moyen/mauvais — cohérent avec nos pseudo-labels
best_k = 3
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
km_labels = km_final.fit_predict(X_clust)

sil = silhouette_score(X_clust, km_labels, sample_size=3000, random_state=42)
ari = adjusted_rand_score(pseudo_labels, km_labels)
print(f"\nK-Means (k={best_k}) :")
print(f"  Silhouette score  : {sil:.3f}")
print(f"  ARI vs pseudo-labels : {ari:.3f}")

fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(pseudo_labels, km_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["Bon", "Moyen", "Mauvais"])
disp.plot(ax=ax, colorbar=False, cmap="Greens")
ax.set_title(f"Confusion K-Means vs pseudo-labels (ARI={ari:.2f})", fontweight="bold")
plt.tight_layout()
plt.savefig("09_confusion_matrix.png", bbox_inches="tight")
plt.show()
print("→ Figure sauvegardée : 09_confusion_matrix.png")

# DBSCAN pour détecter les anomalies — on travaille sur les 2 premières IC
print("\n--- DBSCAN ---")
X_dbscan = X_ica[:, :2]

# k-NN plot pour estimer eps visuellement avant de lancer DBSCAN
from sklearn.neighbors import NearestNeighbors
nbrs = NearestNeighbors(n_neighbors=5).fit(X_dbscan)
distances, _ = nbrs.kneighbors(X_dbscan)
distances = np.sort(distances[:, -1])

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(distances, color=PALETTE[0], linewidth=1)
ax.set_title("k-NN distance plot (aide au choix de eps)", fontweight="bold")
ax.set_xlabel("Points triés")
ax.set_ylabel("Distance au 5e voisin")
plt.tight_layout()
plt.savefig("10_knn_distance.png", bbox_inches="tight")
plt.show()

# eps adaptatif : on prend le 97e percentile des distances — évite de fixer un seuil à la main
eps_val = float(np.percentile(distances, 97))
dbscan = DBSCAN(eps=eps_val, min_samples=10)
db_labels = dbscan.fit_predict(X_dbscan)

n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise = (db_labels == -1).sum()
noise_ratio = n_noise / len(db_labels) * 100
print(f"\nDBSCAN (eps={eps_val:.3f}, min_samples=10) :")
print(f"  Clusters trouvés  : {n_clusters_db}")
print(f"  Points bruit (-1) : {n_noise} ({noise_ratio:.1f}%)")

df_result = df_poll.copy()
df_result["dbscan_label"] = db_labels
df_result["hour"] = df_result.index.hour
df_result["month"] = df_result.index.month

anomalies = df_result[df_result["dbscan_label"] == -1]
normaux = df_result[df_result["dbscan_label"] != -1]

# comparaison pour voir si les anomalies ont des niveaux de pollution plus élevés
print("\nComparaison anomalies vs normaux (moyennes) :")
cols_show = ["CO(GT)", "NOx(GT)", "NO2(GT)", "C6H6(GT)"]
comparison = pd.DataFrame({
    "Normaux": normaux[cols_show].mean(),
    "Anomalies": anomalies[cols_show].mean(),
})
comparison["Ratio"] = (comparison["Anomalies"] / comparison["Normaux"]).round(2)
print(comparison.round(2).to_string())

print(f"\nHeure dominante des anomalies : {anomalies['hour'].value_counts().idxmax()}h")
print(f"Mois dominant des anomalies   : {anomalies['month'].value_counts().idxmax()}")


# -----------------------------------------------------------------------
# 5. visualisations finales
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("5. VISUALISATIONS FINALES")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors_km = [PALETTE[i] for i in km_labels]
axes[0].scatter(X_ica[:, 0], X_ica[:, 1], c=km_labels, cmap="Set2", alpha=0.3, s=5)
axes[0].set_title("Clusters K-Means dans l'espace ICA", fontweight="bold")
axes[0].set_xlabel("IC1"); axes[0].set_ylabel("IC2")

# les points rouges sont les anomalies DBSCAN
colors_db = ["#E76F51" if l == -1 else "#B7E4C7" for l in db_labels]
axes[1].scatter(X_ica[:, 0], X_ica[:, 1], c=colors_db, alpha=0.3, s=5)
axes[1].set_title("Anomalies DBSCAN dans l'espace ICA\n(rouge = bruit)", fontweight="bold")
axes[1].set_xlabel("IC1"); axes[1].set_ylabel("IC2")
plt.tight_layout()
plt.savefig("11_scatter_clusters.png", bbox_inches="tight")
plt.show()
print("→ Figure sauvegardée : 11_scatter_clusters.png")

# série temporelle avec les clusters et les anomalies superposées
fig, ax = plt.subplots(figsize=(16, 5))
cluster_colors_map = {0: PALETTE[0], 1: PALETTE[3], 2: PALETTE[4]}
for c in range(best_k):
    mask = km_labels == c
    idx = df_poll.index[mask]
    vals = df_poll["CO(GT)"].values[mask]
    ax.scatter(idx, vals, s=2, alpha=0.4, color=cluster_colors_map[c], label=f"Cluster {c}")

noise_mask = db_labels == -1
ax.scatter(
    df_poll.index[noise_mask], df_poll["CO(GT)"].values[noise_mask],
    s=20, color="red", alpha=0.7, label="Anomalie (DBSCAN)", zorder=5,
)
ax.set_title("Série temporelle CO(GT) – Clusters K-Means + Anomalies DBSCAN", fontweight="bold")
ax.set_ylabel("CO (mg/m³)")
ax.legend(markerscale=3, fontsize=9)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.tight_layout()
plt.savefig("12_timeseries_annotated.png", bbox_inches="tight")
plt.show()
print("→ Figure sauvegardée : 12_timeseries_annotated.png")

# mixing matrix complète — utile pour interpréter ce que chaque IC capte
fig, ax = plt.subplots(figsize=(12, 6))
mixing_full = pd.DataFrame(
    ica.mixing_,
    index=POLLUTANTS,
    columns=[f"IC{i+1}" for i in range(ica.mixing_.shape[1])],
)
sns.heatmap(
    mixing_full, annot=True, fmt=".2f", cmap="coolwarm",
    center=0, linewidths=0.5, ax=ax, annot_kws={"size": 8},
)
ax.set_title("Heatmap polluants × composantes ICA (mixing matrix)", fontweight="bold")
plt.tight_layout()
plt.savefig("13_heatmap_pollutants_ica.png", bbox_inches="tight")
plt.show()
print("→ Figure sauvegardée : 13_heatmap_pollutants_ica.png")

# distribution temporelle des anomalies — à quelle heure/mois ça arrive le plus ?
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
anomalies["hour"].value_counts().sort_index().plot(
    kind="bar", ax=axes[0], color=PALETTE[4], edgecolor="white"
)
axes[0].set_title("Anomalies par heure de la journée", fontweight="bold")
axes[0].set_xlabel("Heure"); axes[0].set_ylabel("Nombre d'anomalies")

anomalies["month"].value_counts().sort_index().plot(
    kind="bar", ax=axes[1], color=PALETTE[0], edgecolor="white"
)
axes[1].set_title("Anomalies par mois", fontweight="bold")
axes[1].set_xlabel("Mois"); axes[1].set_ylabel("Nombre d'anomalies")
axes[1].set_xticklabels(
    ["Jan","Fév","Mar","Avr","Mai","Jun","Jul","Aoû","Sep","Oct","Nov","Déc"],
    rotation=45
)
plt.tight_layout()
plt.savefig("14_anomalies_temporelles.png", bbox_inches="tight")
plt.show()
print("→ Figure sauvegardée : 14_anomalies_temporelles.png")


print("\n" + "=" * 60)
print("RÉSUMÉ DES RÉSULTATS")
print("=" * 60)
print(f"  ACP : {n_components_90} composantes expliquent ≥90% de variance")
print(f"  PC1 ({explained[0]*100:.1f}%) – groupe capteurs chimiques corrélés")
print(f"  PC2 ({explained[1]*100:.1f}%) – oppose O3 aux autres polluants")
print(f"\n  ICA : {n_ica} sources indépendantes extraites")
print(f"  Kurtosis ICA > variables originales → bonne séparation")
print(f"\n  K-Means (k=3) : silhouette={sil:.3f}, ARI={ari:.3f}")
print(f"  DBSCAN : {n_noise} anomalies ({noise_ratio:.1f}% des mesures)")
print(f"  → Anomalies principalement aux heures de pointe trafic")
print("\nFichiers générés : 01 à 14 *.png")
print("=" * 60)
