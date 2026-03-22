# ia_engine.py — Moteur IA Puissance 4
# Partagé entre game.py (desktop) et app.py (web)

import random
import os
import joblib
from typing import Optional

EMPTY = "."
RED = "R"
YELLOW = "Y"
CONNECT_N = 4

# Chemins du modèle — cherche dans plusieurs endroits au démarrage
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_file(filename):
    candidates = [
        os.path.join(_BASE_DIR, filename),  # racine (ia_engine.py est à la racine)
        os.path.join(_BASE_DIR, "desktop", filename),  # desktop/
        os.path.join(os.path.dirname(_BASE_DIR), filename),  # parent de ia_engine
        os.path.join(os.getcwd(), filename),  # cwd
        os.path.join(os.getcwd(), "desktop", filename),  # cwd/desktop
        os.path.join(os.path.dirname(os.getcwd()), filename),  # parent du cwd
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return os.path.join(_BASE_DIR, filename)


MODEL_PATH = _find_file("connect4_policy_9x9.pkl")
SCALER_PATH = _find_file("connect4_policy_9x9_scaler.pkl")


# ══════════════════════════════════════════════════════════════
# FONCTIONS DE BASE
# ══════════════════════════════════════════════════════════════


def other(token: str) -> str:
    return YELLOW if token == RED else RED


def valid_columns(board: list) -> list:
    return [c for c in range(len(board[0])) if board[0][c] == EMPTY]


def drop_in_grid(grid: list, col: int, token: str) -> Optional[tuple]:
    for r in range(len(grid) - 1, -1, -1):
        if grid[r][col] == EMPTY:
            grid[r][col] = token
            return (r, col)
    return None


def undo_in_grid(grid: list, col: int):
    """Annule le dernier jeton posé dans col."""
    for r in range(len(grid)):
        if grid[r][col] != EMPTY:
            grid[r][col] = EMPTY
            return


def copy_grid(grid: list) -> list:
    return [row[:] for row in grid]


def is_draw(board: list) -> bool:
    return all(board[0][c] != EMPTY for c in range(len(board[0])))


def check_win_cells(board: list, last_row: int, last_col: int, token: str) -> list:
    rows, cols = len(board), len(board[0])
    for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        cells = [(last_row, last_col)]
        r, c = last_row + dr, last_col + dc
        while 0 <= r < rows and 0 <= c < cols and board[r][c] == token:
            cells.append((r, c))
            r += dr
            c += dc
        r, c = last_row - dr, last_col - dc
        while 0 <= r < rows and 0 <= c < cols and board[r][c] == token:
            cells.insert(0, (r, c))
            r -= dr
            c -= dc
        if len(cells) >= CONNECT_N:
            return cells[:CONNECT_N]
    return []


# ══════════════════════════════════════════════════════════════
# DÉTECTION FIN DE PARTIE — BUG CORRIGÉ
# Ancien bug : diagonale / utilisait c + 3 < cols
#              ce qui ratait les diagonales montantes
# ══════════════════════════════════════════════════════════════


def terminal_state(grid: list) -> tuple:
    """Retourne (True, gagnant|None) ou (False, None)."""
    rows, cols = len(grid), len(grid[0])

    for r in range(rows):
        for c in range(cols):
            p = grid[r][c]
            if p == EMPTY:
                continue
            # Horizontal
            if c + 3 < cols and all(grid[r][c + i] == p for i in range(4)):
                return True, p
            # Vertical
            if r + 3 < rows and all(grid[r + i][c] == p for i in range(4)):
                return True, p
            # Diagonale descendante \
            if (
                r + 3 < rows
                and c + 3 < cols
                and all(grid[r + i][c + i] == p for i in range(4))
            ):
                return True, p
            # Diagonale montante /  ← bug corrigé ici
            if (
                r + 3 < rows
                and c - 3 >= 0
                and all(grid[r + i][c - i] == p for i in range(4))
            ):
                return True, p

    if is_draw(grid):
        return True, None
    return False, None


# ══════════════════════════════════════════════════════════════
# HEURISTIQUE
# ══════════════════════════════════════════════════════════════


def evaluate_window(window: list, player: str) -> int:
    opp = other(player)
    cp = window.count(player)
    co = window.count(opp)
    ce = window.count(EMPTY)

    if cp == 4:
        return 100000
    if co == 4:
        return -100000

    score = 0
    if cp == 3 and ce == 1:
        score += 50
    elif cp == 2 and ce == 2:
        score += 10
    if co == 3 and ce == 1:
        score -= 80
    elif co == 2 and ce == 2:
        score -= 10
    return score


def heuristic_score(grid: list, player: str) -> int:
    """
    Évalue le plateau du point de vue de `player`.

    ┌────────────────────────────────────────────────────────┐
    │  POINT D'EXTENSION — IA ENTRAÎNÉE                      │
    │  Remplace ce contenu par :                             │
    │      model = _load_trained_model()                     │
    │      features = extract_features(grid, player)         │
    │      return int(model.predict([features])[0])          │
    └────────────────────────────────────────────────────────┘
    """
    rows, cols = len(grid), len(grid[0])
    score = 0

    # Bonus centre
    center = cols // 2
    score += sum(3 for r in range(rows) if grid[r][center] == player)

    # Horizontal
    for r in range(rows):
        for c in range(cols - 3):
            score += evaluate_window([grid[r][c + i] for i in range(4)], player)

    # Vertical
    for c in range(cols):
        for r in range(rows - 3):
            score += evaluate_window([grid[r + i][c] for i in range(4)], player)

    # Diagonale descendante \
    for r in range(rows - 3):
        for c in range(cols - 3):
            score += evaluate_window([grid[r + i][c + i] for i in range(4)], player)

    # Diagonale montante /  ← corrigée aussi
    for r in range(rows - 3):
        for c in range(3, cols):
            score += evaluate_window([grid[r + i][c - i] for i in range(4)], player)

    return score


# ══════════════════════════════════════════════════════════════
# MINIMAX — BUG CORRIGÉ
#
# Ancien bug : le nœud MIN jouait opp MAIS évaluait encore
#   heuristic_score(..., player) → scores inversés → l'IA
#   choisissait les pires coups au lieu des meilleurs.
#
# Correction : `current` alterne correctement à chaque nœud.
#   L'heuristique est toujours évaluée pour `root_player`.
# ══════════════════════════════════════════════════════════════


def minimax(
    grid: list,
    depth: int,
    alpha: float,
    beta: float,
    maximizing: bool,
    root_player: str,
) -> int:
    """
    root_player : le joueur pour qui on cherche le meilleur coup.
    maximizing  : True = tour de root_player, False = tour de l'adversaire.
    """
    term, winner = terminal_state(grid)
    if term:
        if winner == root_player:
            return 1_000_000
        if winner == other(root_player):
            return -1_000_000
        return 0

    if depth == 0:
        return heuristic_score(grid, root_player)

    cols = valid_columns(grid)
    if not cols:
        return 0

    center = len(grid[0]) // 2
    cols.sort(key=lambda c: abs(c - center))

    # Le joueur qui joue CE coup
    current = root_player if maximizing else other(root_player)

    if maximizing:
        best = -(10**18)
        for col in cols:
            drop_in_grid(grid, col, current)
            val = minimax(grid, depth - 1, alpha, beta, False, root_player)
            undo_in_grid(grid, col)
            if val > best:
                best = val
            alpha = max(alpha, best)
            if alpha >= beta:
                break
        return best
    else:
        best = 10**18
        for col in cols:
            drop_in_grid(grid, col, current)
            val = minimax(grid, depth - 1, alpha, beta, True, root_player)
            undo_in_grid(grid, col)
            if val < best:
                best = val
            beta = min(beta, best)
            if alpha >= beta:
                break
        return best


# ══════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════


def best_move(board: list, player: str, depth: int, ai_mode: str) -> dict:
    """
    Calcule le meilleur coup pour `player`.
    Retourne {'col': int, 'scores': {col: score}}
    """
    cols = valid_columns(board)
    if not cols:
        return {"col": None, "scores": {}}

    if ai_mode == "random":
        return {"col": random.choice(cols), "scores": {}}

    if ai_mode == "lose":
        return {"col": random.choice(cols), "scores": {}}

    if ai_mode == "trained":
        model, scaler = _load_trained_model()
        if model is not None and scaler is not None:
            return _trained_best_move(board, player, cols, model, scaler)
        # Fallback minimax si modèle absent
        ai_mode = "minimax"

    # ── minimax ──────────────────────────────────────────────
    depth = max(1, min(8, depth))

    center = len(board[0]) // 2
    cols.sort(key=lambda c: abs(c - center))

    # ── Priorité 1 : coup gagnant immédiat ───────────────────
    for col in cols:
        drop_in_grid(board, col, player)
        term, winner = terminal_state(board)
        undo_in_grid(board, col)
        if term and winner == player:
            return {"col": col, "scores": {col: 1_000_000}}

    # ── Priorité 2 : bloquer victoire adversaire immédiate ───
    opp = other(player)
    for col in cols:
        drop_in_grid(board, col, opp)
        term, winner = terminal_state(board)
        undo_in_grid(board, col)
        if term and winner == opp:
            return {"col": col, "scores": {col: 999_999}}

    # ── Priorité 3 : minimax complet ─────────────────────────
    scores = {}
    best_val = -(10**18)
    best_col = cols[0]

    for col in cols:
        drop_in_grid(board, col, player)
        val = minimax(board, depth - 1, -(10**18), 10**18, False, player)
        undo_in_grid(board, col)
        scores[col] = val
        if val > best_val:
            best_val = val
            best_col = col

    return {"col": best_col, "scores": scores}


def extract_features(board: list, player: str) -> list:
    """
    Features à 3 canaux — identique à board_to_features() de train_policy.py :
      canal 1 : cases du joueur courant  (1 ou 0)
      canal 2 : cases de l'adversaire    (1 ou 0)
      canal 3 : cases vides              (1 ou 0)
    Taille : rows × cols × 3
    """
    opp = other(player)
    current_ch, opponent_ch, empty_ch = [], [], []
    for row in board:
        for cell in row:
            current_ch.append(1 if cell == player else 0)
            opponent_ch.append(1 if cell == opp else 0)
            empty_ch.append(1 if cell == EMPTY else 0)
    return current_ch + opponent_ch + empty_ch


# ══════════════════════════════════════════════════════════════
# MODÈLE ENTRAÎNÉ
# ══════════════════════════════════════════════════════════════

_cached_model = None
_cached_scaler = None


def _load_trained_model():
    global _cached_model, _cached_scaler
    if _cached_model is not None:
        return _cached_model, _cached_scaler

    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            _cached_model = joblib.load(MODEL_PATH)
            _cached_scaler = joblib.load(SCALER_PATH)
            print(f"[ia_engine] Modèle chargé : {MODEL_PATH}")
            return _cached_model, _cached_scaler
        except Exception as e:
            print(f"[ia_engine] Erreur chargement modèle : {e}")

    return None, None


def _trained_best_move(board: list, player: str, cols: list, model, scaler) -> dict:
    """
    Utilise le modèle MLP + scaler pour choisir le meilleur coup.
    - Prédit les probabilités pour chaque colonne
    - Filtre les colonnes invalides
    - Retourne la colonne avec la plus haute probabilité valide
    """
    import numpy as np

    features = extract_features(board, player)

    # Vérifier que la taille des features correspond au scaler
    expected = scaler.n_features_in_
    if len(features) != expected:
        print(
            f"[ia_engine] ⚠️  Grille {len(board)}x{len(board[0])} incompatible avec le modèle ({expected} features attendues) — fallback minimax"
        )
        return best_move(board, player, depth=4, ai_mode="minimax")

    X = np.array([features], dtype=np.float32)
    X_scaled = scaler.transform(X)

    # predict_proba donne la proba pour chaque colonne (classes = 0..cols-1)
    proba = model.predict_proba(X_scaled)[0]
    classes = list(model.classes_)

    # Choisir la colonne valide avec la plus haute proba
    best_col = None
    best_prob = -1.0
    col_scores = {}

    for col in cols:
        if col in classes:
            idx = classes.index(col)
            p = float(proba[idx])
        else:
            p = 0.0
        col_scores[col] = round(p, 4)
        if p > best_prob:
            best_prob = p
            best_col = col

    # Fallback si aucune colonne reconnue
    if best_col is None:
        best_col = cols[len(cols) // 2]

    return {"col": best_col, "scores": col_scores}


def reload_model():
    global _cached_model, _cached_scaler
    _cached_model = None
    _cached_scaler = None
    m, s = _load_trained_model()
    return m is not None
