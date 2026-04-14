# ia_engine.py — Moteur IA Puissance 4
# Partagé entre game.py (desktop) et app.py (web)
# Modes : random / trained / minimax / hybrid
# Version renforcée :
# - iterative deepening
# - time control
# - transposition table
# - move ordering
# - prédiction améliorée (gagnant + nombre de coups estimé)

import random
import os
import json
import time
import joblib
from typing import Optional

EMPTY = "."
RED = "R"
YELLOW = "Y"
CONNECT_N = 4

WIN_SCORE = 1_000_000_000
INF = 10**18

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 0 = seulement tout début, -1 = jamais
OPENING_BOOK_MAX_MOVES = 0

# Temps par défaut si on utilise l'itératif
DEFAULT_TIME_LIMIT_MS = 1800

# Transposition table
TT_EXACT = 0
TT_LOWER = 1
TT_UPPER = 2

# Killer moves / history heuristic
_MAX_PLY = 128
_killer_moves = [[None, None] for _ in range(_MAX_PLY)]
_history_heuristic = {}

# Cache modèle / book
_cached_model = None
_cached_scaler = None
_cached_opening_book = None

# Table de transposition
_tt = {}

# Contrôle temps
_search_deadline = None
_stop_search = False


# ══════════════════════════════════════════════════════════════
# FICHIERS
# ══════════════════════════════════════════════════════════════


def _find_file(filename):
    here = _BASE_DIR
    parent = os.path.dirname(here)
    grandparent = os.path.dirname(parent)

    candidates = [
        os.path.join(here, filename),
        os.path.join(here, "desktop", filename),
        os.path.join(here, "donnees", filename),
        os.path.join(parent, filename),
        os.path.join(parent, "desktop", filename),
        os.path.join(parent, "donnees", filename),
        os.path.join(grandparent, filename),
        os.path.join(grandparent, "desktop", filename),
        os.path.join(grandparent, "donnees", filename),
        os.path.join(os.getcwd(), filename),
        os.path.join(os.getcwd(), "desktop", filename),
        os.path.join(os.getcwd(), "donnees", filename),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return os.path.join(here, filename)


MODEL_PATH = _find_file("connect4_policy_9x9.pkl")
SCALER_PATH = _find_file("connect4_policy_9x9_scaler.pkl")
OPENING_BOOK_PATH = _find_file("opening_book.json")


# ══════════════════════════════════════════════════════════════
# BASE
# ══════════════════════════════════════════════════════════════


def other(token: str) -> str:
    return YELLOW if token == RED else RED


def copy_grid(grid: list) -> list:
    return [row[:] for row in grid]


def valid_columns(board: list) -> list:
    if not board or not board[0]:
        return []
    return [c for c in range(len(board[0])) if board[0][c] == EMPTY]


def drop_in_grid(grid: list, col: int, token: str) -> Optional[tuple]:
    for r in range(len(grid) - 1, -1, -1):
        if grid[r][col] == EMPTY:
            grid[r][col] = token
            return (r, col)
    return None


def undo_in_grid(grid: list, col: int):
    for r in range(len(grid)):
        if grid[r][col] != EMPTY:
            grid[r][col] = EMPTY
            return


def is_draw(board: list) -> bool:
    return all(board[0][c] != EMPTY for c in range(len(board[0])))


def count_moves(board: list) -> int:
    return sum(1 for row in board for cell in row if cell != EMPTY)


def board_to_moves(board: list, starting_player: str = RED) -> list:
    rows = len(board)
    cols = len(board[0])
    work = [row[:] for row in board]
    moves_rev = []

    while True:
        found = False
        for c in range(cols):
            for r in range(rows):
                if work[r][c] != EMPTY:
                    moves_rev.append(c)
                    work[r][c] = EMPTY
                    found = True
                    break
        if not found:
            break

    return list(reversed(moves_rev))


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


def terminal_state(grid: list) -> tuple:
    rows, cols = len(grid), len(grid[0])

    for r in range(rows):
        for c in range(cols):
            p = grid[r][c]
            if p == EMPTY:
                continue

            if c + 3 < cols and all(grid[r][c + i] == p for i in range(4)):
                return True, p

            if r + 3 < rows and all(grid[r + i][c] == p for i in range(4)):
                return True, p

            if (
                r + 3 < rows
                and c + 3 < cols
                and all(grid[r + i][c + i] == p for i in range(4))
            ):
                return True, p

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
# TEMPS
# ══════════════════════════════════════════════════════════════


def _time_up() -> bool:
    global _search_deadline, _stop_search
    if _stop_search:
        return True
    if _search_deadline is None:
        return False
    if time.perf_counter() >= _search_deadline:
        _stop_search = True
        return True
    return False


# ══════════════════════════════════════════════════════════════
# HEURISTIQUE
# ══════════════════════════════════════════════════════════════


def evaluate_window(window: list, player: str) -> int:
    opp = other(player)
    cp = window.count(player)
    co = window.count(opp)
    ce = window.count(EMPTY)

    if cp == 4:
        return 100_000
    if co == 4:
        return -100_000

    score = 0

    if cp == 3 and ce == 1:
        score += 160
    elif cp == 2 and ce == 2:
        score += 24
    elif cp == 1 and ce == 3:
        score += 2

    if co == 3 and ce == 1:
        score -= 190
    elif co == 2 and ce == 2:
        score -= 28
    elif co == 1 and ce == 3:
        score -= 2

    return score


def _count_potential_threes(grid: list, player: str) -> int:
    rows, cols = len(grid), len(grid[0])
    total = 0

    for r in range(rows):
        for c in range(cols - 3):
            w = [grid[r][c + i] for i in range(4)]
            if w.count(player) == 3 and w.count(EMPTY) == 1:
                total += 1

    for c in range(cols):
        for r in range(rows - 3):
            w = [grid[r + i][c] for i in range(4)]
            if w.count(player) == 3 and w.count(EMPTY) == 1:
                total += 1

    for r in range(rows - 3):
        for c in range(cols - 3):
            w = [grid[r + i][c + i] for i in range(4)]
            if w.count(player) == 3 and w.count(EMPTY) == 1:
                total += 1

    for r in range(rows - 3):
        for c in range(3, cols):
            w = [grid[r + i][c - i] for i in range(4)]
            if w.count(player) == 3 and w.count(EMPTY) == 1:
                total += 1

    return total


def heuristic_score(grid: list, player: str) -> int:
    rows, cols = len(grid), len(grid[0])
    score = 0
    opp = other(player)
    center = cols // 2

    for r in range(rows):
        if grid[r][center] == player:
            score += 14
        elif grid[r][center] == opp:
            score -= 14

    for r in range(rows):
        for c in range(cols):
            dist = abs(c - center)
            center_bonus = max(0, (cols // 2 + 1) - dist)
            if grid[r][c] == player:
                score += center_bonus
            elif grid[r][c] == opp:
                score -= center_bonus

    for r in range(rows):
        for c in range(cols - 3):
            score += evaluate_window([grid[r][c + i] for i in range(4)], player)

    for c in range(cols):
        for r in range(rows - 3):
            score += evaluate_window([grid[r + i][c] for i in range(4)], player)

    for r in range(rows - 3):
        for c in range(cols - 3):
            score += evaluate_window([grid[r + i][c + i] for i in range(4)], player)

    for r in range(rows - 3):
        for c in range(3, cols):
            score += evaluate_window([grid[r + i][c - i] for i in range(4)], player)

    my_now = _count_immediate_wins(grid, player)
    opp_now = _count_immediate_wins(grid, opp)
    score += my_now * 7000
    score -= opp_now * 9000

    score += _count_potential_threes(grid, player) * 50
    score -= _count_potential_threes(grid, opp) * 60

    return score


# ══════════════════════════════════════════════════════════════
# TACTIQUE
# ══════════════════════════════════════════════════════════════


def _order_columns_center_first(cols: list, width: int) -> list:
    center = width // 2
    return sorted(cols, key=lambda c: (abs(c - center), c))


def _find_immediate_winning_move(board: list, player: str) -> Optional[int]:
    cols = _order_columns_center_first(valid_columns(board), len(board[0]))
    for col in cols:
        pos = drop_in_grid(board, col, player)
        if not pos:
            continue
        won = bool(check_win_cells(board, pos[0], pos[1], player))
        undo_in_grid(board, col)
        if won:
            return col
    return None


def _find_all_immediate_wins(board: list, player: str) -> list:
    wins = []
    cols = _order_columns_center_first(valid_columns(board), len(board[0]))
    for col in cols:
        pos = drop_in_grid(board, col, player)
        if not pos:
            continue
        if check_win_cells(board, pos[0], pos[1], player):
            wins.append(col)
        undo_in_grid(board, col)
    return wins


def _count_immediate_wins(board: list, player: str) -> int:
    return len(_find_all_immediate_wins(board, player))


def _find_immediate_blocking_move(board: list, player: str) -> Optional[int]:
    opp = other(player)
    cols = _order_columns_center_first(valid_columns(board), len(board[0]))
    for col in cols:
        pos = drop_in_grid(board, col, opp)
        if not pos:
            continue
        opp_wins = bool(check_win_cells(board, pos[0], pos[1], opp))
        undo_in_grid(board, col)
        if opp_wins:
            return col
    return None


def _gives_opponent_immediate_win(board: list, player: str, col: int) -> bool:
    opp = other(player)
    pos = drop_in_grid(board, col, player)
    if not pos:
        return True

    opp_can_win = _find_immediate_winning_move(board, opp) is not None
    undo_in_grid(board, col)
    return opp_can_win


def _safe_columns(board: list, player: str) -> list:
    cols = _order_columns_center_first(valid_columns(board), len(board[0]))
    safe = [c for c in cols if not _gives_opponent_immediate_win(board, player, c)]
    return safe if safe else cols


def _creates_double_threat(board: list, player: str, col: int) -> bool:
    pos = drop_in_grid(board, col, player)
    if not pos:
        return False

    wins = _find_all_immediate_wins(board, player)
    undo_in_grid(board, col)
    return len(wins) >= 2


def _opponent_creates_double_threat_after(board: list, player: str, col: int) -> bool:
    opp = other(player)
    pos = drop_in_grid(board, col, player)
    if not pos:
        return True

    danger = False
    for oc in valid_columns(board):
        opos = drop_in_grid(board, oc, opp)
        if not opos:
            continue

        if len(_find_all_immediate_wins(board, opp)) >= 2:
            danger = True
            undo_in_grid(board, oc)
            break

        undo_in_grid(board, oc)

    undo_in_grid(board, col)
    return danger


# ══════════════════════════════════════════════════════════════
# OPENING BOOK
# ══════════════════════════════════════════════════════════════


def mirror_col(col: int, cols: int) -> int:
    return cols - 1 - col


def mirror_moves(moves: list, cols: int) -> list:
    return [mirror_col(c, cols) for c in moves]


def canonical_moves_key(moves: list, cols: int) -> tuple[str, bool]:
    original = list(moves)
    mirrored = mirror_moves(original, cols)

    if tuple(original) <= tuple(mirrored):
        return ",".join(map(str, original)), False
    return ",".join(map(str, mirrored)), True


def _normalize_opening_book_payload(data):
    if isinstance(data, dict) and "book" in data and isinstance(data["book"], dict):
        data = data["book"]

    if not isinstance(data, dict):
        return {}

    out = {}
    for k, v in data.items():
        if isinstance(v, int):
            out[str(k)] = {"col": int(v), "weight": 1}
        elif isinstance(v, dict) and "col" in v:
            try:
                out[str(k)] = {
                    "col": int(v["col"]),
                    "weight": int(v.get("weight", 1)),
                }
            except Exception:
                continue
    return out


def _default_opening_book():
    return {
        "": {"col": 4, "weight": 100},
        "4": {"col": 4, "weight": 80},
        "3": {"col": 4, "weight": 80},
        "5": {"col": 4, "weight": 80},
        "4,4": {"col": 3, "weight": 60},
        "4,3": {"col": 4, "weight": 60},
        "4,5": {"col": 4, "weight": 60},
        "3,4": {"col": 4, "weight": 60},
        "5,4": {"col": 4, "weight": 60},
        "4,4,3": {"col": 5, "weight": 50},
        "4,4,5": {"col": 3, "weight": 50},
        "4,3,4": {"col": 5, "weight": 50},
        "4,5,4": {"col": 3, "weight": 50},
    }


def _load_opening_book():
    global _cached_opening_book
    if _cached_opening_book is not None:
        return _cached_opening_book

    book = {}

    if os.path.exists(OPENING_BOOK_PATH):
        try:
            with open(OPENING_BOOK_PATH, "r", encoding="utf-8") as f:
                payload = json.load(f)
            book = _normalize_opening_book_payload(payload)
            print(f"[ia_engine] Opening book chargé : {OPENING_BOOK_PATH}")
        except Exception as e:
            print(f"[ia_engine] Erreur chargement opening book : {e}")
            book = {}

    if not book:
        book = _default_opening_book()
        print("[ia_engine] Opening book par défaut utilisé")

    _cached_opening_book = book
    return _cached_opening_book


def opening_book_move(board: list, player: str) -> Optional[int]:
    rows = len(board)
    cols = len(board[0])

    if rows < 4 or cols < 4:
        return None

    moves = board_to_moves(board, starting_player=RED)
    key, mirrored = canonical_moves_key(moves, cols)

    book = _load_opening_book()
    entry = book.get(key)

    if not entry:
        return None

    col = int(entry["col"])
    if mirrored:
        col = mirror_col(col, cols)

    if col in valid_columns(board):
        return col

    return None


# ══════════════════════════════════════════════════════════════
# FEATURES MODELE
# ══════════════════════════════════════════════════════════════


def extract_features(board: list, player: str) -> list:
    opp = other(player)
    current_ch, opponent_ch, empty_ch = [], [], []
    for row in board:
        for cell in row:
            current_ch.append(1 if cell == player else 0)
            opponent_ch.append(1 if cell == opp else 0)
            empty_ch.append(1 if cell == EMPTY else 0)
    return current_ch + opponent_ch + empty_ch


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


def _trained_scores(board: list, player: str, cols: list, model, scaler) -> dict:
    import numpy as np

    features = extract_features(board, player)
    expected = getattr(scaler, "n_features_in_", None)
    if expected is None or len(features) != expected:
        return {}

    X = np.array([features], dtype=np.float32)
    X_scaled = scaler.transform(X)

    proba = model.predict_proba(X_scaled)[0]
    classes = list(model.classes_)

    col_scores = {}
    for col in cols:
        if col in classes:
            idx = classes.index(col)
            col_scores[col] = float(proba[idx])
        else:
            col_scores[col] = 0.0

    return col_scores


def _trained_best_move(board: list, player: str, cols: list, model, scaler) -> dict:
    win_col = _find_immediate_winning_move(board, player)
    if win_col is not None:
        return {
            "col": win_col,
            "scores": {win_col: 1_000_000},
            "source": "winning_move",
        }

    block_col = _find_immediate_blocking_move(board, player)
    if block_col is not None:
        return {
            "col": block_col,
            "scores": {block_col: 999_999},
            "source": "blocking_move",
        }

    safe_cols = _safe_columns(board, player)
    col_scores = _trained_scores(board, player, safe_cols, model, scaler)

    if not col_scores:
        return best_move(board, player, depth=6, ai_mode="minimax")

    best_col = max(
        safe_cols,
        key=lambda c: (col_scores.get(c, 0.0), -abs(c - (len(board[0]) // 2))),
    )
    nice_scores = {c: round(col_scores.get(c, 0.0), 4) for c in safe_cols}
    return {"col": best_col, "scores": nice_scores, "source": "trained"}


# ══════════════════════════════════════════════════════════════
# ORDERING
# ══════════════════════════════════════════════════════════════


def _board_key(board: list) -> tuple:
    return tuple(tuple(r) for r in board)


def _history_score(player: str, col: int) -> int:
    return _history_heuristic.get((player, col), 0)


def _record_killer(ply: int, col: int):
    if ply >= _MAX_PLY:
        return
    if _killer_moves[ply][0] == col:
        return
    _killer_moves[ply][1] = _killer_moves[ply][0]
    _killer_moves[ply][0] = col


def _record_history(player: str, col: int, depth: int):
    key = (player, col)
    _history_heuristic[key] = _history_heuristic.get(key, 0) + depth * depth


def _hybrid_ordered_cols(
    board: list, player: str, cols: list, ply: int = 0
) -> tuple[list, dict]:
    ordered_base = _order_columns_center_first(cols, len(board[0]))
    trained_scores = {}

    model, scaler = _load_trained_model()
    if model is not None and scaler is not None:
        try:
            trained_scores = _trained_scores(board, player, ordered_base, model, scaler)
        except Exception:
            trained_scores = {}

    center = len(board[0]) // 2
    scored = []

    immediate_win = _find_immediate_winning_move(board, player)
    immediate_block = _find_immediate_blocking_move(board, player)

    for c in ordered_base:
        score = 0

        if immediate_win == c:
            score += 10_000_000

        if immediate_block == c:
            score += 9_000_000

        if _creates_double_threat(board, player, c):
            score += 700_000

        if _gives_opponent_immediate_win(board, player, c):
            score -= 2_500_000

        if _opponent_creates_double_threat_after(board, player, c):
            score -= 250_000

        if ply < _MAX_PLY:
            if _killer_moves[ply][0] == c:
                score += 120_000
            elif _killer_moves[ply][1] == c:
                score += 90_000

        score += _history_score(player, c)
        score += (len(board[0]) - abs(c - center)) * 20
        score += trained_scores.get(c, 0.0) * 120.0

        scored.append((c, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in scored], trained_scores


# ══════════════════════════════════════════════════════════════
# NEGAMAX + ALPHA-BETA + TT
# ══════════════════════════════════════════════════════════════


def _evaluate_for_side_to_move(
    grid: list, current_player: str, root_player: str
) -> int:
    val = heuristic_score(grid, root_player)
    return val if current_player == root_player else -val


def _negamax(
    grid: list,
    depth: int,
    alpha: int,
    beta: int,
    current_player: str,
    root_player: str,
    ply: int,
) -> tuple[int, Optional[int], Optional[int]]:
    """
    Retourne (score, best_move, mate_distance)

    - score : du point de vue du joueur current_player
    - best_move : meilleure colonne
    - mate_distance :
        * nombre de demi-coups avant la fin forcée si trouvé
        * None sinon
    """
    global _tt

    if _time_up():
        return 0, None, None

    alpha_orig = alpha
    beta_orig = beta

    term, winner = terminal_state(grid)
    if term:
        if winner is None:
            return 0, None, 0
        if winner == current_player:
            return WIN_SCORE - ply, None, 0
        return -WIN_SCORE + ply, None, 0

    if depth <= 0:
        return _evaluate_for_side_to_move(grid, current_player, root_player), None, None

    key = (_board_key(grid), depth, current_player)
    tt_entry = _tt.get(key)
    if tt_entry is not None:
        tt_score, tt_move, tt_flag, tt_dist = tt_entry
        if tt_flag == TT_EXACT:
            return tt_score, tt_move, tt_dist
        if tt_flag == TT_LOWER:
            alpha = max(alpha, tt_score)
        elif tt_flag == TT_UPPER:
            beta = min(beta, tt_score)
        if alpha >= beta:
            return tt_score, tt_move, tt_dist

    cols = valid_columns(grid)
    if not cols:
        return 0, None, 0

    safe_cols = [
        c for c in cols if not _gives_opponent_immediate_win(grid, current_player, c)
    ]
    if safe_cols:
        cols = safe_cols

    ordered_cols, _ = _hybrid_ordered_cols(grid, current_player, cols, ply=ply)
    if not ordered_cols:
        ordered_cols = _order_columns_center_first(cols, len(grid[0]))

    best_score = -INF
    best_move = ordered_cols[0]
    best_dist = None

    opp = other(current_player)

    for col in ordered_cols:
        pos = drop_in_grid(grid, col, current_player)
        if not pos:
            continue

        if check_win_cells(grid, pos[0], pos[1], current_player):
            score = WIN_SCORE - (ply + 1)
            move_dist = 1
        else:
            child_score, _, child_dist = _negamax(
                grid=grid,
                depth=depth - 1,
                alpha=-beta,
                beta=-alpha,
                current_player=opp,
                root_player=root_player,
                ply=ply + 1,
            )
            score = -child_score
            move_dist = None if child_dist is None else child_dist + 1

        undo_in_grid(grid, col)

        if _time_up():
            return 0, None, None

        better = False
        if score > best_score:
            better = True
        elif score == best_score:
            if score > 900_000:
                if best_dist is None or (
                    move_dist is not None and move_dist < best_dist
                ):
                    better = True
            elif score < -900_000:
                if best_dist is None or (
                    move_dist is not None and move_dist > best_dist
                ):
                    better = True
            else:
                center = len(grid[0]) // 2
                if abs(col - center) < abs(best_move - center):
                    better = True

        if better:
            best_score = score
            best_move = col
            best_dist = move_dist

        if best_score > alpha:
            alpha = best_score

        if alpha >= beta:
            _record_killer(ply, col)
            _record_history(current_player, col, depth)
            break

    if best_score <= alpha_orig:
        flag = TT_UPPER
    elif best_score >= beta_orig:
        flag = TT_LOWER
    else:
        flag = TT_EXACT

    _tt[key] = (best_score, best_move, flag, best_dist)
    return best_score, best_move, best_dist


def minimax(
    grid: list,
    depth: int,
    alpha: float,
    beta: float,
    maximizing_player: bool,
    ai_player: str,
    human_player: str,
    ply: int = 0,
) -> dict:
    current = ai_player if maximizing_player else human_player
    score, move, dist = _negamax(
        grid=grid,
        depth=depth,
        alpha=int(alpha),
        beta=int(beta),
        current_player=current,
        root_player=ai_player,
        ply=ply,
    )

    if current != ai_player:
        score = -score

    return {"score": int(score), "move": move, "distance": dist}


# ══════════════════════════════════════════════════════════════
# RECHERCHE ITÉRATIVE
# ══════════════════════════════════════════════════════════════


def _iterative_search(
    board: list,
    player: str,
    depth: int,
    time_limit_ms: Optional[int] = None,
) -> dict:
    global _search_deadline, _stop_search, _tt

    if time_limit_ms is None:
        time_limit_ms = DEFAULT_TIME_LIMIT_MS

    _stop_search = False
    _search_deadline = time.perf_counter() + (time_limit_ms / 1000.0)
    _tt = {}

    cols = valid_columns(board)
    if not cols:
        _search_deadline = None
        _stop_search = False
        return {
            "col": None,
            "scores": {},
            "source": "iterative",
            "depth_reached": 0,
            "distance": None,
            "distances": {},
        }

    win_col = _find_immediate_winning_move(board, player)
    if win_col is not None:
        _search_deadline = None
        _stop_search = False
        return {
            "col": win_col,
            "scores": {win_col: WIN_SCORE},
            "source": "winning_move",
            "depth_reached": 1,
            "distance": 1,
            "distances": {win_col: 1},
        }

    block_col = _find_immediate_blocking_move(board, player)
    if block_col is not None:
        _search_deadline = None
        _stop_search = False
        return {
            "col": block_col,
            "scores": {block_col: WIN_SCORE - 1},
            "source": "blocking_move",
            "depth_reached": 1,
            "distance": None,
            "distances": {},
        }

    moves_played = count_moves(board)
    if OPENING_BOOK_MAX_MOVES >= 0 and moves_played <= OPENING_BOOK_MAX_MOVES:
        book_col = opening_book_move(board, player)
        if book_col is not None:
            _search_deadline = None
            _stop_search = False
            return {
                "col": book_col,
                "scores": {},
                "source": "opening",
                "depth_reached": 0,
                "distance": None,
                "distances": {},
            }

    ordered_default = _order_columns_center_first(cols, len(board[0]))
    safe_cols = _safe_columns(board, player)

    best_result = {
        "col": ordered_default[0] if ordered_default else None,
        "scores": {},
        "source": "iterative_depth_0",
        "depth_reached": 0,
        "distance": None,
        "distances": {},
    }

    for current_depth in range(1, max(1, depth) + 1):
        if _time_up():
            break

        scores = {}
        distances = {}
        best_val = -INF
        best_col = None
        best_dist = None
        opp = other(player)

        ordered_root, trained_scores = _hybrid_ordered_cols(
            board, player, safe_cols, ply=0
        )
        if not ordered_root:
            ordered_root = _order_columns_center_first(safe_cols, len(board[0]))
            trained_scores = {}

        for col in ordered_root:
            if _time_up():
                break

            next_board = copy_grid(board)
            pos = drop_in_grid(next_board, col, player)
            if not pos:
                continue

            if check_win_cells(next_board, pos[0], pos[1], player):
                val = WIN_SCORE - 1
                dist = 1
            else:
                child_score, _, child_dist = _negamax(
                    grid=next_board,
                    depth=current_depth - 1,
                    alpha=-INF,
                    beta=INF,
                    current_player=opp,
                    root_player=player,
                    ply=1,
                )
                val = -child_score
                dist = None if child_dist is None else child_dist + 1

            val += trained_scores.get(col, 0.0) * 100.0
            if _creates_double_threat(board, player, col):
                val += 90_000
            if _opponent_creates_double_threat_after(board, player, col):
                val -= 40_000

            scores[col] = int(val)
            distances[col] = dist

            better = False
            if best_col is None or val > best_val:
                better = True
            elif val == best_val:
                if val > 900_000:
                    if best_dist is None or (dist is not None and dist < best_dist):
                        better = True
                elif val < -900_000:
                    if best_dist is None or (dist is not None and dist > best_dist):
                        better = True
                else:
                    center = len(board[0]) // 2
                    if abs(col - center) < abs(best_col - center):
                        better = True

            if better:
                best_val = val
                best_col = col
                best_dist = dist

        if not _time_up() and best_col is not None:
            best_result = {
                "col": best_col,
                "scores": scores,
                "source": f"iterative_depth_{current_depth}",
                "depth_reached": current_depth,
                "distance": best_dist,
                "distances": distances,
            }

            if best_val > WIN_SCORE - 10_000:
                break

    _search_deadline = None
    _stop_search = False
    return best_result


# ══════════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════════


def predict_outcome(
    board: list,
    player: str,
    depth: int = 12,
    time_limit_ms: int = 1800,
) -> dict:
    term, winner = terminal_state(board)
    if term:
        return {
            "winner": winner,
            "moves": 0,
            "score": 0,
            "depth_reached": 0,
            "best_col": None,
            "source": "terminal",
        }

    immediate = _find_immediate_winning_move(board, player)
    if immediate is not None:
        return {
            "winner": player,
            "moves": 1,
            "score": WIN_SCORE - 1,
            "depth_reached": 1,
            "best_col": immediate,
            "source": "winning_move",
        }

    opp = other(player)
    opp_immediate = _find_immediate_winning_move(board, opp)
    if opp_immediate is not None:
        safe = _safe_columns(board, player)
        if not safe:
            return {
                "winner": opp,
                "moves": 1,
                "score": -WIN_SCORE + 1,
                "depth_reached": 1,
                "best_col": None,
                "source": "opponent_winning_move",
            }

    remaining = sum(1 for row in board for cell in row if cell == EMPTY)

    if remaining <= 20:
        result = _iterative_search(
            board=copy_grid(board),
            player=player,
            depth=42,
            time_limit_ms=10000,
        )
    else:
        result = _iterative_search(
            board=copy_grid(board),
            player=player,
            depth=max(1, min(depth, 16)),
            time_limit_ms=time_limit_ms,
        )

    col = result.get("col")
    scores = result.get("scores", {})
    distances = result.get("distances", {})
    score = int(scores.get(col, 0)) if col is not None else 0
    dist = distances.get(col)
    depth_reached = int(result.get("depth_reached", 0))

    if score > 900_000:
        return {
            "winner": player,
            "moves": dist,
            "score": score,
            "depth_reached": depth_reached,
            "best_col": col,
            "source": result.get("source", "iterative"),
        }

    if score < -900_000:
        return {
            "winner": opp,
            "moves": dist,
            "score": score,
            "depth_reached": depth_reached,
            "best_col": col,
            "source": result.get("source", "iterative"),
        }

    return {
        "winner": None,
        "moves": dist,
        "score": score,
        "depth_reached": depth_reached,
        "best_col": col,
        "source": result.get("source", "iterative"),
    }


# ══════════════════════════════════════════════════════════════
# HYBRID
# ══════════════════════════════════════════════════════════════


def _hybrid_best_move(
    board: list, player: str, depth: int, time_limit_ms: Optional[int] = None
) -> dict:
    return _iterative_search(board, player, depth, time_limit_ms=time_limit_ms)


# ══════════════════════════════════════════════════════════════
# BEST MOVE
# ══════════════════════════════════════════════════════════════


def best_move(
    board: list,
    player: str,
    depth: int,
    ai_mode: str,
    time_limit_ms: Optional[int] = None,
) -> dict:
    cols = valid_columns(board)
    if not cols:
        return {
            "col": None,
            "scores": {},
            "source": "none",
            "depth_reached": 0,
            "distance": None,
            "distances": {},
        }

    ai_mode = (ai_mode or "minimax").lower()
    depth = max(1, min(16, depth))

    if ai_mode == "random":
        c = random.choice(cols)
        return {
            "col": c,
            "scores": {},
            "source": "random",
            "depth_reached": 0,
            "distance": None,
            "distances": {},
        }

    if ai_mode == "lose":
        c = random.choice(cols)
        return {
            "col": c,
            "scores": {},
            "source": "lose",
            "depth_reached": 0,
            "distance": None,
            "distances": {},
        }

    win_col = _find_immediate_winning_move(board, player)
    if win_col is not None:
        return {
            "col": win_col,
            "scores": {win_col: 1_000_000},
            "source": "winning_move",
            "depth_reached": 1,
            "distance": 1,
            "distances": {win_col: 1},
        }

    block_col = _find_immediate_blocking_move(board, player)
    if block_col is not None:
        return {
            "col": block_col,
            "scores": {block_col: 999_999},
            "source": "blocking_move",
            "depth_reached": 1,
            "distance": None,
            "distances": {},
        }

    moves_played = count_moves(board)
    if OPENING_BOOK_MAX_MOVES >= 0 and moves_played <= OPENING_BOOK_MAX_MOVES:
        book_col = opening_book_move(board, player)
        if book_col is not None:
            return {
                "col": book_col,
                "scores": {},
                "source": "opening",
                "depth_reached": 0,
                "distance": None,
                "distances": {},
            }

    if ai_mode == "trained":
        model, scaler = _load_trained_model()
        if model is not None and scaler is not None:
            r = _trained_best_move(board, player, cols, model, scaler)
            r.setdefault("depth_reached", 0)
            r.setdefault("distance", None)
            r.setdefault("distances", {})
            return r
        ai_mode = "minimax"

    if ai_mode == "hybrid":
        return _hybrid_best_move(board, player, depth, time_limit_ms=time_limit_ms)

    return _iterative_search(
        board=board,
        player=player,
        depth=depth,
        time_limit_ms=(
            time_limit_ms if time_limit_ms is not None else DEFAULT_TIME_LIMIT_MS
        ),
    )


def reload_model():
    global _cached_model, _cached_scaler, _cached_opening_book, _tt
    _cached_model = None
    _cached_scaler = None
    _cached_opening_book = None
    _tt = {}
    m, s = _load_trained_model()
    _load_opening_book()
    return m is not None
