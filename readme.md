# 🔴🟡 Puissance 4+
### IA · Web · Desktop · Base de Données

> Version avancée du Puissance 4 avec intelligence artificielle, interface web, version desktop, base de données PostgreSQL, sauvegarde, replay et analyse prédictive.

---

## 📋 Présentation

**Puissance 4+** a été conçu pour :

- jouer au Puissance 4 contre une IA
- analyser une position en temps réel
- prédire le meilleur coup possible
- estimer l'issue d'une partie (victoire / défaite / nul / incertain)
- sauvegarder les parties en base de données
- rejouer les parties enregistrées
- importer des parties réelles depuis BoardGameArena
- entraîner et améliorer l'IA avec des données réelles

---

## ✨ Fonctionnalités principales

### Jeu complet

Plusieurs modes de jeu disponibles :

- Humain vs IA
- IA vs IA
- Humain vs Humain
- Mode online (web)

### Intelligence Artificielle

Le moteur IA est partagé entre la version web et la version desktop (`ia_engine.py`).

**L'IA est capable de :**
- détecter les coups gagnants immédiats
- bloquer les coups adverses
- anticiper plusieurs tours à l'avance
- prédire le gagnant potentiel et en combien de coups
- évaluer si une position est gagnante, perdante, nulle ou incertaine

**Méthodes utilisées :**
- Minimax + Alpha-Beta Pruning
- Recherche itérative avec contrôle du temps
- Heuristiques de score (colonnes centrales, doubles menaces)
- Mode hybride : algorithmique + MLPClassifier (scikit-learn)

### Prédiction en temps réel

À chaque coup joué, le système affiche :
- le meilleur coup conseillé
- l'avantage estimé
- si un joueur peut forcer une victoire, et en combien de coups

Exemples d'affichage :
```
Rouge gagne en 3 coups
Jaune a l'avantage
Position équilibrée
Meilleur coup conseillé : colonne 5
```

### Sauvegarde & Replay

Chaque partie peut être sauvegardée, stockée en base, rechargée et rejouée coup par coup.

Informations sauvegardées : nom, taille du plateau, liste des coups, joueur de départ, mode de jeu, vainqueur, niveau de confiance, date de création.

### Import BoardGameArena

Pipeline complet d'import de parties jouées sur BoardGameArena :
1. Scraping via Selenium
2. Extraction des coups
3. Conversion au format interne
4. Sauvegarde JSON
5. Import PostgreSQL

---

## 🗂️ Architecture du projet

```
puissance4/
│
├── ia_engine.py              # Moteur IA partagé
├── .env                      # Variables d'environnement
│
├── web/
│   ├── app.py                # Backend FastAPI
│   ├── render.yaml           # Config déploiement Render
│   ├── requirements.txt      # Dépendances web
│   └── public/
│       ├── index.html        # Interface web
│       ├── style.css         # Design
│       └── game.js           # Logique frontend
│
├── desktop/
│   ├── game.py               # Version desktop (Tkinter)
│   ├── database_viewer.py    # Visualisation base de données
│   ├── bga_to_db.py          # Scraping BGA
│   └── bga_import.py         # Import BGA
│
├── base_de_donnees/
│   ├── database_schema.sql   # Schéma SQL
│   
└── donnees/
    ├── config.json
    ├── player_ids.json
    ├── player_cache.json
    └── scraped_moves/
```

---

## 🛠️ Technologies utilisées

| Couche | Technologies |
|--------|-------------|
| Backend | Python, FastAPI, PostgreSQL, psycopg2 |
| Frontend | HTML, CSS, JavaScript |
| IA | Minimax, Alpha-Beta, Heuristiques, scikit-learn (MLPClassifier) |
| Scraping | Selenium, Regex, JSON |
| Desktop | Python, Tkinter |
| Déploiement | Render |

---

## 🚀 Installation

### 1. Cloner le projet

```bash
git clone https://github.com/rayaneweb/puissance4.git
cd puissance4
```

### 2. Créer l'environnement virtuel

```bash
python -m venv .venv
source .venv/bin/activate
# Windows : .venv\Scripts\activate
```

### 3. Installer les dépendances

```bash
cd web
pip install -r requirements.txt
```

### 4. Configurer l'environnement

Créer un fichier `.env` à la racine :

```env
DATABASE_URL=postgresql://USER:PASSWORD@HOST:PORT/DBNAME
```

### 5. Lancer le backend

```bash
cd web
uvicorn app:app --reload
```

### 6. Ouvrir l'application

- Frontend : [http://127.0.0.1:8000](http://127.0.0.1:8000)
- API : [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

---

## 🖥️ Version desktop

```bash
cd desktop
python game.py
```

Pour visualiser la base de données :

```bash
python database_viewer.py
```

---

## 🌐 Endpoints de l'API

### IA
| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `POST` | `/api/ai/move` | Calcule le meilleur coup IA |
| `POST` | `/api/predict` | Prédit l'issue de la position |

### Parties
| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `POST` | `/api/games` | Sauvegarder une partie |
| `GET` | `/api/games` | Lister les parties |
| `GET` | `/api/games/{game_id}` | Récupérer une partie |
| `DELETE` | `/api/games/{game_id}` | Supprimer une partie |

### Online
| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `POST` | `/api/online/create` | Créer une partie en ligne |
| `POST` | `/api/online/join` | Rejoindre une partie |
| `GET` | `/api/online/{code}/state` | État de la partie |
| `POST` | `/api/online/{code}/move` | Jouer un coup |

### Santé
| Méthode | Endpoint | Description |
|---------|----------|-------------|
| `GET` | `/api/health` | Vérifier l'état du serveur |

---

## ☁️ Déploiement Render

- **Root Directory :** `web`
- **Build Command :** `pip install -r requirements.txt`
- **Start Command :** `uvicorn app:app --host 0.0.0.0 --port $PORT`
- **Variable d'environnement :** `DATABASE_URL=...`

---

## 🗃️ Base de données PostgreSQL

Table principale : `saved_games`

| Colonne | Description |
|---------|-------------|
| `game_id` | Identifiant unique |
| `save_name` | Nom de la sauvegarde |
| `rows_count` | Nombre de lignes |
| `cols_count` | Nombre de colonnes |
| `starting_color` | Joueur de départ |
| `moves` | Liste des coups |
| `winner` | Vainqueur |
| `status` | Statut de la partie |
| `confidence` | Niveau de confiance |
| `created_at` | Date de création |

---

## 🔮 Améliorations futures

- IA plus rapide avec meilleur modèle ML
- Mode spectateur avancé
- Statistiques détaillées & leaderboard
- Entraînement automatique de l'IA
- Export des parties
- Analyse visuelle par image

---

## 👥 Auteurs

- **Rayane Ait Braham**
- **Benrabah Salah**
