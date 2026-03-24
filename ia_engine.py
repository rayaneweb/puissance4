# ia_engine.py — Moteur IA Puissance 4
# Partagé entre game.py (desktop) et app.py (web)
# Ajout : opening book + prédiction de résultat

import random
import os
import json
import joblib
from typing import Optional

EMPTY = "."
RED = "R"
YELLOW = "Y"
CONNECT_N = 4

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_file(filename):
    candidates = [
        os.path.join(_BASE_DIR, filename),
        os.path.join(_BASE_DIR, "desktop", filename),
        os.path.join(_BASE_DIR, "donnees", filename),
        os.path.join(os.path.dirname(_BASE_DIR), filename),
        os.path.join(os.getcwd(), filename),
        os.path.join(os.getcwd(), "desktop", filename),
        os.path.join(os.getcwd(), "donnees", filename),
        os.path.join(os.path.dirname(os.getcwd()), filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return os.path.join(_BASE_DIR, filename)


MODEL_PATH = _find_file("connect4_policy_9x9.pkl")
SCALER_PATH = _find_file("connect4_policy_9x9_scaler.pkl")
OPENING_BOOK_PATH = _find_file("opening_book.json")


# ══════════════════════════════════════════════════════════════
# FONCTIONS DE BASE
# ══════════════════════════════════════════════════════════════


def other(token: str) -> str:
    return YELLOW if token == RED else RED


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


def copy_grid(grid: list) -> list:
    return [row[:] for row in grid]


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


# ══════════════════════════════════════════════════════════════
# FIN DE PARTIE
# ══════════════════════════════════════════════════════════════


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
    rows, cols = len(grid), len(grid[0])
    score = 0

    center = cols // 2
    score += sum(3 for r in range(rows) if grid[r][center] == player)

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

    return score


# ══════════════════════════════════════════════════════════════
# MINIMAX
# ══════════════════════════════════════════════════════════════


def minimax(
    grid: list,
    depth: int,
    alpha: float,
    beta: float,
    maximizing_player: bool,
    ai_player: str,
    human_player: str,
) -> dict:
    terminal_over, terminal_winner = terminal_state(grid)

    if terminal_over:
        if terminal_winner == ai_player:
            return {"score": 1_000_000_000 + depth, "move": None}
        if terminal_winner == human_player:
            return {"score": -1_000_000_000 - depth, "move": None}
        return {"score": 0, "move": None}

    if depth == 0:
        return {"score": heuristic_score(grid, ai_player), "move": None}

    moves = valid_columns(grid)
    center = len(grid[0]) // 2
    moves.sort(key=lambda c: abs(c - center))

    best_move = moves[0] if moves else None

    if maximizing_player:
        max_eval = -float("inf")

        for col in moves:
            next_grid = copy_grid(grid)
            pos = drop_in_grid(next_grid, col, ai_player)
            if not pos:
                continue

            ai_win = check_win_cells(next_grid, pos[0], pos[1], ai_player)
            if ai_win:
                result = {"score": 1_000_000_000 + depth, "move": col}
            else:
                opp_can_win = False
                opp_moves = valid_columns(next_grid)

                for opp_col in opp_moves:
                    test_grid = copy_grid(next_grid)
                    opp_pos = drop_in_grid(test_grid, opp_col, human_player)
                    if not opp_pos:
                        continue
                    opp_win = check_win_cells(
                        test_grid, opp_pos[0], opp_pos[1], human_player
                    )
                    if opp_win:
                        opp_can_win = True
                        break

                if opp_can_win:
                    result = {"score": -999_999_999, "move": col}
                else:
                    result = minimax(
                        next_grid,
                        depth - 1,
                        alpha,
                        beta,
                        False,
                        ai_player,
                        human_player,
                    )

            if result["score"] > max_eval:
                max_eval = result["score"]
                best_move = col

            alpha = max(alpha, max_eval)
            if beta <= alpha:
                break

        return {"score": max_eval, "move": best_move}

    min_eval = float("inf")

    for col in moves:
        next_grid = copy_grid(grid)
        pos = drop_in_grid(next_grid, col, human_player)
        if not pos:
            continue

        human_win = check_win_cells(next_grid, pos[0], pos[1], human_player)
        if human_win:
            result = {"score": -1_000_000_000 - depth, "move": col}
        else:
            ai_can_win = False
            ai_moves = valid_columns(next_grid)

            for ai_col in ai_moves:
                test_grid = copy_grid(next_grid)
                ai_pos = drop_in_grid(test_grid, ai_col, ai_player)
                if not ai_pos:
                    continue
                ai_win = check_win_cells(test_grid, ai_pos[0], ai_pos[1], ai_player)
                if ai_win:
                    ai_can_win = True
                    break

            if ai_can_win:
                result = {"score": 999_999_999, "move": col}
            else:
                result = minimax(
                    next_grid,
                    depth - 1,
                    alpha,
                    beta,
                    True,
                    ai_player,
                    human_player,
                )

        if result["score"] < min_eval:
            min_eval = result["score"]
            best_move = col

        beta = min(beta, min_eval)
        if beta <= alpha:
            break

    return {"score": min_eval, "move": best_move}


# ══════════════════════════════════════════════════════════════
# OPENING BOOK
# ══════════════════════════════════════════════════════════════

_cached_opening_book = None


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
# FEATURES
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
    import numpy as np

    features = extract_features(board, player)

    expected = scaler.n_features_in_
    if len(features) != expected:
        print(
            f"[ia_engine] ⚠️ Grille {len(board)}x{len(board[0])} incompatible avec le modèle ({expected} features attendues) — fallback minimax"
        )
        return best_move(board, player, depth=4, ai_mode="minimax")

    X = np.array([features], dtype=np.float32)
    X_scaled = scaler.transform(X)

    proba = model.predict_proba(X_scaled)[0]
    classes = list(model.classes_)

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

    if best_col is None:
        best_col = cols[len(cols) // 2]

    return {"col": best_col, "scores": col_scores, "source": "trained"}


# ══════════════════════════════════════════════════════════════
# PRÉDICTION
# ══════════════════════════════════════════════════════════════


def _minimax_with_distance(
    grid: list,
    depth: int,
    alpha: float,
    beta: float,
    maximizing: bool,
    root_player: str,
    ply: int = 0,
):
    term, winner = terminal_state(grid)
    if term:
        if winner == root_player:
            return 1_000_000 - ply, ply
        if winner == other(root_player):
            return -1_000_000 + ply, ply
        return 0, ply

    if depth == 0:
        return heuristic_score(grid, root_player), ply

    cols = valid_columns(grid)
    if not cols:
        return 0, ply

    center = len(grid[0]) // 2
    cols.sort(key=lambda c: abs(c - center))

    current = root_player if maximizing else other(root_player)

    if maximizing:
        best_score = -(10**18)
        best_dist = 10**9

        for col in cols:
            pos = drop_in_grid(grid, col, current)
            if not pos:
                continue

            score, dist = _minimax_with_distance(
                grid, depth - 1, alpha, beta, False, root_player, ply + 1
            )
            undo_in_grid(grid, col)

            if score > best_score:
                best_score = score
                best_dist = dist
            elif score == best_score:
                if score > 900000:
                    best_dist = min(best_dist, dist)
                elif score < -900000:
                    best_dist = max(best_dist, dist)

            alpha = max(alpha, best_score)
            if alpha >= beta:
                break

        return best_score, best_dist

    best_score = 10**18
    best_dist = 10**9

    for col in cols:
        pos = drop_in_grid(grid, col, current)
        if not pos:
            continue

        score, dist = _minimax_with_distance(
            grid, depth - 1, alpha, beta, True, root_player, ply + 1
        )
        undo_in_grid(grid, col)

        if score < best_score:
            best_score = score
            best_dist = dist
        elif score == best_score:
            if score > 900000:
                best_dist = max(best_dist, dist)
            elif score < -900000:
                best_dist = min(best_dist, dist)

        beta = min(beta, best_score)
        if alpha >= beta:
            break

    return best_score, best_dist


def _plies_to_turns(ply_distance: int):
    if ply_distance is None:
        return None
    return max(1, (ply_distance + 1) // 2)


def predict_outcome(board: list, player: str, depth: int = 8) -> dict:
    grid = copy_grid(board)

    term, winner = terminal_state(grid)
    if term:
        return {
            "winner": winner,
            "moves": 0 if winner is not None else None,
            "score": (
                0 if winner is None else (1_000_000 if winner == player else -1_000_000)
            ),
        }

    cols = valid_columns(grid)
    if not cols:
        return {"winner": None, "moves": None, "score": 0}

    # 1) victoire immédiate du joueur courant
    for col in cols:
        pos = drop_in_grid(grid, col, player)
        if pos:
            if check_win_cells(grid, pos[0], pos[1], player):
                undo_in_grid(grid, col)
                return {"winner": player, "moves": 1, "score": 1_000_000}
            undo_in_grid(grid, col)

    # 2) victoire immédiate adverse au prochain coup
    opp = other(player)
    for col in cols:
        pos = drop_in_grid(grid, col, opp)
        if pos:
            if check_win_cells(grid, pos[0], pos[1], opp):
                undo_in_grid(grid, col)
                return {"winner": opp, "moves": 1, "score": -1_000_000}
            undo_in_grid(grid, col)

    score, ply_dist = _minimax_with_distance(
        grid, depth, -(10**18), 10**18, True, player, 0
    )

    if score >= 900000:
        return {
            "winner": player,
            "moves": _plies_to_turns(ply_dist),
            "score": int(score),
        }

    if score <= -900000:
        return {
            "winner": opp,
            "moves": _plies_to_turns(ply_dist),
            "score": int(score),
        }

    return {
        "winner": None,
        "moves": None,
        "score": int(score),
    }


# ══════════════════════════════════════════════════════════════
# BEST MOVE
# ══════════════════════════════════════════════════════════════


def best_move(board: list, player: str, depth: int, ai_mode: str) -> dict:
    cols = valid_columns(board)
    if not cols:
        return {"col": None, "scores": {}, "source": "none"}

    ai_mode = (ai_mode or "minimax").lower()

    if ai_mode == "random":
        return {"col": random.choice(cols), "scores": {}, "source": "random"}

    if ai_mode == "lose":
        return {"col": random.choice(cols), "scores": {}, "source": "lose"}

    moves_played = count_moves(board)
    if moves_played <= 8:
        book_col = opening_book_move(board, player)
        if book_col is not None:
            return {"col": book_col, "scores": {}, "source": "opening"}

    if ai_mode == "trained":
        model, scaler = _load_trained_model()
        if model is not None and scaler is not None:
            return _trained_best_move(board, player, cols, model, scaler)
        ai_mode = "minimax"

    depth = max(1, min(8, depth))

    center = len(board[0]) // 2
    cols.sort(key=lambda c: abs(c - center))

    for col in cols:
        drop_in_grid(board, col, player)
        term, winner = terminal_state(board)
        undo_in_grid(board, col)
        if term and winner == player:
            return {"col": col, "scores": {col: 1_000_000}, "source": "winning_move"}

    opp = other(player)
    for col in cols:
        drop_in_grid(board, col, opp)
        term, winner = terminal_state(board)
        undo_in_grid(board, col)
        if term and winner == opp:
            return {"col": col, "scores": {col: 999_999}, "source": "blocking_move"}

    scores = {}
    best_val = -float("inf")
    best_col = cols[0]

    for col in cols:
        next_board = copy_grid(board)
        pos = drop_in_grid(next_board, col, player)
        if not pos:
            continue

        ai_win = check_win_cells(next_board, pos[0], pos[1], player)
        if ai_win:
            val = 1_000_000_000 + depth
        else:
            result = minimax(
                next_board,
                depth - 1,
                -float("inf"),
                float("inf"),
                False,
                player,
                opp,
            )
            val = result["score"]

        scores[col] = val
        if val > best_val:
            best_val = val
            best_col = col

    return {"col": best_col, "scores": scores, "source": "minimax"}


def reload_model():
    global _cached_model, _cached_scaler, _cached_opening_book
    _cached_model = None
    _cached_scaler = None
    _cached_opening_book = None
    m, s = _load_trained_model()
    _load_opening_book()
    return m is not None
