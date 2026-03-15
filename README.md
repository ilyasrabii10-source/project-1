# 🫀 HeartGuard — Prédiction du Risque d'Insuffisance Cardiaque

> **Coding Week · Centrale Casablanca · Mars 2026**  
> Fouad Ghadi · Yassine Ait Bella · Rabi Ilyas · Yahiaoui Ziyad · Chakir Mohamed

---

## 📌 À propos du projet

HeartGuard est une application clinique d'aide à la décision médicale. Elle permet à un médecin de saisir les données biologiques d'un patient et d'obtenir en temps réel une estimation du risque d'insuffisance cardiaque fatale, accompagnée d'une explication visuelle des facteurs qui ont influencé cette prédiction.

Le projet repose sur un pipeline Machine Learning complet : nettoyage des données, gestion du déséquilibre de classes, entraînement de plusieurs modèles, sélection du meilleur, et déploiement via une interface Streamlit.

---

## 📁 Structure du projet

```
project-1/
├── data/
│   ├── heart.csv                        # Dataset original UCI
│   └── heart_balanced.csv               # Dataset après optimisation mémoire
├── notebooks/
│   └── eda.ipynb                        # Analyse exploratoire des données
├── src/
│   ├── data_processing.py               # Nettoyage, optimisation mémoire
│   ├── train_model.py                   # Entraînement des 4 modèles
│   ├── EvaluateModel.py                 # Évaluation et courbe ROC
│   ├── heart_model.pkl                  # Random Forest (meilleur modèle)
│   ├── heart_model_xgb.pkl              # XGBoost
│   ├── heart_model_lgbm.pkl             # LightGBM
│   ├── heart_model_lr.pkl               # Logistic Regression
│   └── metrics.json                     # Précision test sauvegardée
├── app/
│   └── app.py                           # Interface Streamlit
├── tests/
│   └── test_data_processing.py          # Tests automatisés
├── .github/
│   └── workflows/
│       └── ci.yml                       # Pipeline CI/CD GitHub Actions
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🚀 Installation & Lancement

### 1. Cloner le dépôt
```bash
git clone https://github.com/ZiyadYahiaouiT35/project-1.git
cd project-1
```

### 2. Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3. Préparer les données
```bash
python src/data_processing.py
```

### 4. Entraîner les modèles
```bash
python src/train_model.py
```

### 5. Lancer l'application
```bash
streamlit run app/app.py
```

Ouvrir ensuite : **http://localhost:8501**

---

## 📊 Dataset

- **Source :** [UCI ML Repository — Heart Failure Clinical Records](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records)
- **Taille :** 299 patients, 13 variables cliniques
- **Variable cible :** `DEATH_EVENT` (0 = survie, 1 = décès)

---

## 🔍 Analyse exploratoire (EDA)

### Valeurs manquantes
Aucune valeur manquante détectée dans le dataset original. Aucun traitement d'imputation nécessaire.

### Outliers
Des valeurs extrêmes ont été identifiées sur `creatinine_phosphokinase`, `platelets` et `serum_creatinine`. Elles ont été conservées car elles correspondent à des cas cliniques réels et non à des erreurs de saisie.

### Déséquilibre de classes
Le dataset est **déséquilibré** :
- 68% de patients survivants
- 32% de patients décédés

**Méthode choisie : SMOTE (Synthetic Minority Over-sampling Technique)**

SMOTE a été appliqué **uniquement sur les données d'entraînement**, après la séparation train/test, pour éviter toute fuite de données (data leakage). Cette décision est importante : appliquer SMOTE avant le split avait gonflé artificiellement la précision à **91.5%** — après correction, le score honnête est **83.3%**.

**Impact :** amélioration du recall sur la classe minoritaire (décédés), réduction du biais du modèle, distribution équilibrée à 203 décédés / 203 survivants dans le train set.

### Corrélations
`serum_creatinine`, `ejection_fraction` et `time` sont les variables les plus corrélées avec `DEATH_EVENT`. Aucune multicolinéarité critique détectée.

---

## 🤖 Modèles évalués

Quatre modèles ont été entraînés et comparés sur les métriques suivantes : **Accuracy, ROC-AUC, Precision, Recall, F1-Score**.

| Modèle | Accuracy | ROC-AUC | F1-Score |
|--------|----------|---------|---------|
| **Random Forest** ✅ | **83.3%** | **~0.89** | **~0.83** |
| XGBoost | ~81% | ~0.87 | ~0.80 |
| LightGBM | ~80% | ~0.86 | ~0.79 |
| Logistic Regression | ~76% | ~0.82 | ~0.75 |

### ✅ Meilleur modèle : Random Forest

Le Random Forest a obtenu les meilleures performances sur toutes les métriques. Sa robustesse face aux valeurs aberrantes et sa capacité à capturer des interactions non-linéaires entre les variables cliniques en font le choix le plus adapté à ce type de données médicales hétérogènes.

---

## 🔬 Explication SHAP

SHAP (SHapley Additive exPlanations) a été intégré pour rendre les prédictions du modèle transparentes et interprétables pour les médecins.

Pour chaque patient analysé, l'interface affiche :
- Un **graphique à barres** montrant l'impact de chaque variable sur la prédiction (rouge = augmente le risque, vert = le réduit)
- Un **tableau détaillé** des valeurs SHAP par feature
- Un **top 3 des facteurs décisifs** sous forme de cartes visuelles

### Features les plus influentes (résultats SHAP) :
1. **`time`** (durée de suivi) — variable la plus prédictive, 0.330 d'importance
2. **`serum_creatinine`** — indicateur clé de la fonction rénale, 0.184
3. **`ejection_fraction`** — capacité de pompage du cœur, 0.124
4. **`age`** — 0.087
5. **`serum_sodium`** — 0.063

### ⚠️ Note sur le tabagisme
Dans ce dataset, le tabagisme montre une corrélation contre-intuitive avec la mortalité (0.014 d'importance, légèrement négatif). Cela est dû à la petite taille de l'échantillon (299 patients) et **ne reflète pas la réalité clinique**. Un avertissement est affiché dans l'interface pour en informer le médecin.

---

## 💾 Optimisation mémoire

Une fonction `optimize_memory(df)` a été développée dans `src/data_processing.py`. Elle convertit automatiquement :
- `float64` → `float32`
- `int64` → `int32`

Résultat : réduction significative de l'empreinte mémoire du dataset sans aucune perte d'information.

---

## 🔧 Problèmes rencontrés & Solutions

### 1. Data Leakage — le faux 91.5%

C'est le problème le plus important que nous avons rencontré, et probablement le plus instructif.

Au départ, notre pipeline appliquait SMOTE sur l'ensemble du dataset **avant** la séparation train/test :

```
❌ Pipeline initial (data leakage)
Dataset complet → SMOTE → train_test_split → 91.5% (artificiel)
```

Le jeu de test contenait des copies synthétiques de données d'entraînement — le modèle était évalué sur des données qu'il avait indirectement vues. C'est ce qu'on appelle une **fuite de données**.

**La correction :**
```
✅ Pipeline corrigé
Dataset complet → train_test_split → SMOTE sur train uniquement → 83.3% (honnête)
```

Le score est passé de 91.5% à 83.3% — ce qui reflète la vraie performance du modèle sur des patients qu'il n'a jamais vus.

### 2. Structure du projet — les chemins cassés

Lors de la restructuration en dossiers (`src/`, `app/`, `data/`...), l'application ne trouvait plus les fichiers. Le problème venait de chemins relatifs.

**La correction :**
```python
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "heart_balanced.csv")
```

### 3. Le tabagisme — corrélation contre-intuitive

Le tabagisme semblait réduire le risque dans notre modèle. Ce n'est pas un bug mais une limitation statistique du dataset (299 patients). Nous avons choisi de conserver la variable mais d'afficher un avertissement dans l'interface.

---

## 🧪 Tests automatisés

```bash
pytest tests/ -v
```

Les tests couvrent :
- `test_optimize_memory_reduces_size` — la mémoire est réduite après optimisation
- `test_optimize_memory_types` — les types sont bien convertis en float32/int32
- `test_handle_missing_values` — les valeurs manquantes sont correctement traitées
- `test_handle_outliers_clips_values` — les outliers sont clippés
- `test_optimize_memory_preserves_row_count` — le nombre de lignes est préservé

Ces tests sont exécutés automatiquement via **GitHub Actions** à chaque push.

---

## ⚙️ CI/CD — GitHub Actions

Le fichier `.github/workflows/ci.yml` automatise :
1. Installation des dépendances
2. Exécution des tests pytest
3. Validation sur Python 3.10

---

## 🧠 Prompt Engineering

**Tâche sélectionnée :** Fonction `optimize_memory(df)`

**Prompt initial :**
> *"Write a Python function `optimize_memory(df)` that reduces memory usage of a pandas DataFrame by converting float64 to float32 and int64 to int32. Print memory before and after."*

**Résultat :** Fonctionnel mais incomplet — ne gérait pas les colonnes non-numériques.

**Prompt amélioré :**
> *"...Make sure to only apply downcasting to numeric columns using select_dtypes, and skip the DEATH_EVENT column."*

**Leçon :** Spécifier les cas limites dès le premier prompt réduit le nombre d'itérations et améliore directement la qualité du code généré.

---

## 🤖 Comment l'IA nous a aidés

Nous avons utilisé des outils d'IA générative (Claude, GitHub Copilot) de manière ciblée tout au long du projet.

**Ce qui a bien marché :**
- Débogage rapide des erreurs (`FileNotFoundError`, `ModuleNotFoundError`, data leakage)
- Génération de code boilerplate (`optimize_memory`, tests pytest, `ci.yml`)
- Explication de concepts (SHAP, data leakage, corrélations statistiques)
- Restructuration du projet et correction des chemins de fichiers

**Ce qui a moins bien marché :**
- Les suggestions sans contexte précis étaient souvent incorrectes
- L'IA ne connaissait pas notre code — il fallait toujours donner le contexte complet

**Leçons apprises :**
- Plus le prompt est précis et contextualisé, meilleure est la réponse
- Toujours tester le code généré — l'IA peut se tromper sur des détails de configuration
- L'IA est un assistant, pas un remplaçant — toutes les décisions de conception restaient les nôtres

---

## ❓ Questions critiques

**Le dataset était-il équilibré ?**  
Non — 68% survivants / 32% décédés. Traité avec SMOTE appliqué uniquement sur le train set. Impact : distribution équilibrée dans l'entraînement, recall amélioré sur la classe minoritaire.

**Quel modèle a le mieux performé ?**  
Random Forest — **83.3% accuracy**, ~0.89 ROC-AUC. C'est le score honnête après correction du data leakage.

**Quelles features médicales influencent le plus les prédictions ?**  
`time` (0.330), `serum_creatinine` (0.184), `ejection_fraction` (0.124), `age` (0.087), `serum_sodium` (0.063) — selon l'importance Random Forest et confirmé par SHAP.

**Quels insights le prompt engineering a-t-il apporté ?**  
La précision du prompt est directement liée à la qualité du résultat. Un prompt vague produit du code fonctionnel mais incomplet. Mentionner les cas limites (colonnes non-numériques, colonnes cibles) dès le départ évite des itérations inutiles.

---

## 🐳 Docker

```bash
docker build -t heartguard .
docker run -p 8501:8501 heartguard
```

---

*Centrale Casablanca · Coding Week · Mars 2026*
