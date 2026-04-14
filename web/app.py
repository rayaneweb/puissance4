import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

import json
import secrets
from typing import Optional, List

import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

import ia_engine
from ia_engine import (
    best_move,
    check_win_cells,
    valid_columns,
    drop_in_grid,
    copy_grid,
    EMPTY,
    RED,
    YELLOW,
    other,
    reload_model,
    predict_outcome,
)

load_dotenv(os.path.join(BASE_DIR, ".env"))
load_dotenv(os.path.join(PARENT_DIR, ".env"))
# ══════════════════════════════════════════════════════════════
# PATHS / ENV
# ══════════════════════════════════════════════════════════════

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

load_dotenv(os.path.join(BASE_DIR, ".env"))
load_dotenv(os.path.join(PARENT_DIR, ".env"))

from ia_engine import (
    best_move,
    check_win_cells,
    valid_columns,
    drop_in_grid,
    copy_grid,
    EMPTY,
    RED,
    YELLOW,
    other,
    reload_model,
    predict_outcome,
)

# ══════════════════════════════════════════════════════════════
# APP
# ══════════════════════════════════════════════════════════════

app = FastAPI(title="Puissance 4 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════════
# DB
# ══════════════════════════════════════════════════════════════


def db_conn():
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        return psycopg2.connect(database_url, cursor_factory=RealDictCursor)

    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "5432")
    name = os.environ.get("DB_NAME", "puissance4_db")
    user = os.environ.get("DB_USER", "postgres")
    password = os.environ.get("DB_PASSWORD", "")

    return psycopg2.connect(
        host=host,
        port=port,
        database=name,
        user=user,
        password=password,
        cursor_factory=RealDictCursor,
    )


def init_db():
    try:
        conn = db_conn()
        cur = conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS online_rooms (
                id SERIAL PRIMARY KEY,
                room_code VARCHAR(32) UNIQUE NOT NULL,
                board_json TEXT NOT NULL,
                current_player VARCHAR(1) NOT NULL DEFAULT 'R',
                winner VARCHAR(16),
                status VARCHAR(16) NOT NULL DEFAULT 'waiting',
                player_red_name VARCHAR(100),
                player_yellow_name VARCHAR(100),
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS online_moves (
                id SERIAL PRIMARY KEY,
                room_code VARCHAR(32) NOT NULL,
                player VARCHAR(1) NOT NULL,
                col INTEGER NOT NULL,
                played_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )

        conn.commit()
        cur.close()
        conn.close()
        print("[DB] OK")
    except Exception as e:
        print(f"[DB] Init ignorée: {e}")


@app.on_event("startup")
def startup_event():
    try:
        reload_model()
        print("[IA] reload_model OK")
    except Exception as e:
        print(f"[IA] erreur reload_model: {e}")

    init_db()


# ══════════════════════════════════════════════════════════════
# UTILS GAME
# ══════════════════════════════════════════════════════════════


def normalize_player(player: str) -> str:
    p = str(player).strip().upper()
    if p not in (RED, YELLOW):
        raise HTTPException(status_code=400, detail="player invalide")
    return p


def normalize_mode(mode: Optional[str]) -> str:
    if not mode:
        return "minimax"

    m = str(mode).strip().lower()
    aliases = {
        "random": "random",
        "aleatoire": "random",
        "minimax": "minimax",
        "ia": "minimax",
        "smart": "minimax",
        "trained": "trained",
        "train": "trained",
        "treined": "trained",
        "ml": "trained",
        "model": "trained",
        "hybrid": "hybrid",
        "hybride": "hybrid",
        "hybid": "hybrid",
    }
    return aliases.get(m, "minimax")


def make_empty_board(rows: int = 9, cols: int = 9):
    return [[EMPTY for _ in range(cols)] for _ in range(rows)]


def validate_board(board):
    if not isinstance(board, list) or not board:
        raise HTTPException(status_code=400, detail="grid invalide")

    row_len = None
    for row in board:
        if not isinstance(row, list):
            raise HTTPException(status_code=400, detail="grid invalide")

        if row_len is None:
            row_len = len(row)
        elif len(row) != row_len:
            raise HTTPException(status_code=400, detail="grid non rectangulaire")

        for cell in row:
            if cell not in (EMPTY, RED, YELLOW):
                raise HTTPException(status_code=400, detail="case invalide")

    if row_len is None or row_len == 0:
        raise HTTPException(status_code=400, detail="grid invalide")

    if len(board) < 4 or row_len < 4:
        raise HTTPException(status_code=400, detail="grille trop petite")

    return board


def count_tokens(board):
    red_count = 0
    yellow_count = 0
    for row in board:
        for cell in row:
            if cell == RED:
                red_count += 1
            elif cell == YELLOW:
                yellow_count += 1
    return red_count, yellow_count


def infer_current_player_from_board(board):
    red_count, yellow_count = count_tokens(board)

    # Rouge commence
    if red_count == yellow_count:
        return RED

    if red_count == yellow_count + 1:
        return YELLOW

    return None


def board_has_valid_turn_counts(board):
    red_count, yellow_count = count_tokens(board)
    return red_count == yellow_count or red_count == yellow_count + 1


def winner_of_board(board):
    rows = len(board)
    cols = len(board[0])

    for r in range(rows):
        for c in range(cols):
            token = board[r][c]
            if token == EMPTY:
                continue
            cells = check_win_cells(board, r, c, token)
            if cells:
                return token
    return None


def board_is_draw(board):
    return len(valid_columns(board)) == 0 and winner_of_board(board) is None


def terminal_board_info(board):
    winner = winner_of_board(board)
    draw = board_is_draw(board)
    over = winner is not None or draw
    return {
        "over": over,
        "winner": winner,
        "draw": draw,
    }


def pluralize_coup(n: Optional[int]) -> str:
    if n is None:
        return "coups"
    return "coup" if int(n) == 1 else "coups"


def prediction_text_from_result(winner, moves_to_win, draw=False):
    if draw:
        return "Prédiction : Match nul"

    if winner == RED:
        if moves_to_win == 0:
            return "Prédiction : Rouge a déjà gagné"
        if moves_to_win is not None:
            return f"Prédiction : Rouge gagne dans {moves_to_win} {pluralize_coup(moves_to_win)}"
        return "Prédiction : Rouge va gagner"

    if winner == YELLOW:
        if moves_to_win == 0:
            return "Prédiction : Jaune a déjà gagné"
        if moves_to_win is not None:
            return f"Prédiction : Jaune gagne dans {moves_to_win} {pluralize_coup(moves_to_win)}"
        return "Prédiction : Jaune va gagner"

    return "Prédiction : Match nul ou position incertaine"


def normalize_prediction_result(raw, board=None, requested_player=None):
    """
    Uniformise la sortie de predict_outcome().
    moves_to_win = nombre de demi-coups renvoyés par ia_engine
    """

    if board is not None:
        term = terminal_board_info(board)
        if term["winner"] == RED:
            return {
                "winner": RED,
                "moves_to_win": 0,
                "score": None,
                "text": "Prédiction : Rouge a déjà gagné",
                "terminal": True,
                "draw": False,
            }
        if term["winner"] == YELLOW:
            return {
                "winner": YELLOW,
                "moves_to_win": 0,
                "score": None,
                "text": "Prédiction : Jaune a déjà gagné",
                "terminal": True,
                "draw": False,
            }
        if term["draw"]:
            return {
                "winner": None,
                "moves_to_win": 0,
                "score": 0,
                "text": "Prédiction : Match nul",
                "terminal": True,
                "draw": True,
            }

    if raw is None:
        return {
            "winner": None,
            "moves_to_win": None,
            "score": None,
            "text": "Prédiction : indisponible",
            "terminal": False,
            "draw": False,
        }

    winner = None
    moves_to_win = None
    score = None

    if isinstance(raw, dict):
        winner = raw.get("winner")
        moves_to_win = raw.get("moves_to_win")
        if moves_to_win is None:
            moves_to_win = raw.get("moves")
        if moves_to_win is None:
            moves_to_win = raw.get("plies_to_win")
        score = raw.get("score")

    elif isinstance(raw, (tuple, list)):
        winner = raw[0] if len(raw) > 0 else None
        moves_to_win = raw[1] if len(raw) > 1 else None
        score = raw[2] if len(raw) > 2 else None

    elif isinstance(raw, str):
        return {
            "winner": None,
            "moves_to_win": None,
            "score": None,
            "text": raw,
            "terminal": False,
            "draw": False,
        }

    text = prediction_text_from_result(winner, moves_to_win, draw=False)

    return {
        "winner": winner,
        "moves_to_win": moves_to_win,
        "score": score,
        "text": text,
        "terminal": False,
        "draw": False,
    }


def call_prediction_engine(board, player, depth=8, ai_mode="minimax"):
    """
    Appelle le moteur de prédiction de manière plus sûre :
    - valide la grille
    - vérifie si la partie est déjà finie
    - vérifie si le joueur demandé correspond au tour estimé
    """
    try:
        validate_board(board)

        if not board_has_valid_turn_counts(board):
            return {
                "winner": None,
                "moves_to_win": None,
                "score": None,
                "text": "Prédiction : grille incohérente",
                "terminal": False,
                "draw": False,
            }

        term = terminal_board_info(board)
        if term["over"]:
            return normalize_prediction_result(
                None, board=board, requested_player=player
            )

        inferred = infer_current_player_from_board(board)
        effective_player = player

        # Si le player envoyé ne correspond pas à la grille,
        # on bascule sur celui déduit de la position
        if inferred in (RED, YELLOW) and inferred != player:
            effective_player = inferred

        raw = predict_outcome(board, effective_player, max(1, min(int(depth or 8), 12)))
        normalized = normalize_prediction_result(
            raw, board=board, requested_player=effective_player
        )
        normalized["player_used"] = effective_player
        normalized["requested_player"] = player
        normalized["ai_mode"] = ai_mode
        return normalized

    except Exception as e:
        print(f"[PREDICT] erreur: {e}")
        return {
            "winner": None,
            "moves_to_win": None,
            "score": None,
            "text": "Prédiction : indisponible",
            "terminal": False,
            "draw": False,
        }


# ══════════════════════════════════════════════════════════════
# MODELS API
# ══════════════════════════════════════════════════════════════


class MoveRequest(BaseModel):
    grid: List[List[str]]
    player: str
    ai_mode: Optional[str] = "minimax"
    depth: Optional[int] = Field(default=6, ge=1, le=12)


class PredictRequest(BaseModel):
    grid: List[List[str]]
    player: str
    ai_mode: Optional[str] = "minimax"
    depth: Optional[int] = Field(default=8, ge=1, le=12)


class PlayRequest(BaseModel):
    grid: List[List[str]]
    col: int
    player: str


class NewRoomRequest(BaseModel):
    player_name: Optional[str] = "Joueur"
    rows: Optional[int] = 9
    cols: Optional[int] = 9
    starting_color: Optional[str] = "R"


class JoinRoomRequest(BaseModel):
    room_code: str
    player_name: Optional[str] = "Joueur"


class RoomMoveRequest(BaseModel):
    room_code: str
    col: int
    player: str


class RoomAIMoveRequest(BaseModel):
    room_code: str
    player: str
    ai_mode: Optional[str] = "minimax"
    depth: Optional[int] = Field(default=6, ge=1, le=12)


# ══════════════════════════════════════════════════════════════
# ROOT / DEBUG
# ══════════════════════════════════════════════════════════════


@app.get("/api/health")
def health():
    return {"ok": True, "message": "API OK"}


@app.post("/api/reload_model")
def api_reload_model():
    try:
        reload_model()
        return {"ok": True, "message": "Modèle rechargé"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur reload_model: {e}")


# ══════════════════════════════════════════════════════════════
# LOCAL GAME API
# ══════════════════════════════════════════════════════════════


@app.post("/api/best_move")
def api_best_move(req: MoveRequest):
    try:
        board = validate_board(req.grid)
        player = normalize_player(req.player)
        ai_mode = normalize_mode(req.ai_mode)
        depth = req.depth or 6

        term = terminal_board_info(board)
        if term["over"]:
            return {
                "ok": True,
                "col": None,
                "mode": "finished",
                "reason": "finished",
                "scores": {},
                "winner": term["winner"],
                "draw": term["draw"],
            }

        # Optionnel mais plus robuste
        inferred = infer_current_player_from_board(board)
        if inferred in (RED, YELLOW):
            player = inferred

        result = best_move(board, player, depth, ai_mode)

        return {
            "ok": True,
            "col": result.get("col"),
            "mode": result.get("source", ai_mode),
            "reason": result.get("source"),
            "scores": result.get("scores"),
            "player_used": player,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur IA: {e}")


@app.post("/api/predict")
def api_predict(req: PredictRequest):
    try:
        board = validate_board(req.grid)
        player = normalize_player(req.player)
        ai_mode = normalize_mode(req.ai_mode)
        depth = req.depth or 8

        prediction = call_prediction_engine(
            board=board,
            player=player,
            depth=depth,
            ai_mode=ai_mode,
        )

        return {
            "ok": True,
            "winner": prediction.get("winner"),
            "moves_to_win": prediction.get("moves_to_win"),
            "score": prediction.get("score"),
            "text": prediction.get("text"),
            "terminal": prediction.get("terminal"),
            "draw": prediction.get("draw"),
            "player_used": prediction.get("player_used"),
            "requested_player": prediction.get("requested_player"),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur prédiction: {e}")


@app.post("/api/play")
def api_play(req: PlayRequest):
    try:
        board = validate_board(req.grid)
        player = normalize_player(req.player)

        term = terminal_board_info(board)
        if term["over"]:
            raise HTTPException(status_code=400, detail="Partie déjà terminée")

        inferred = infer_current_player_from_board(board)
        if inferred in (RED, YELLOW) and inferred != player:
            raise HTTPException(
                status_code=400, detail="Ce n'est pas le bon joueur pour cette grille"
            )

        if req.col not in valid_columns(board):
            raise HTTPException(status_code=400, detail="Colonne invalide")

        board2 = copy_grid(board)
        pos = drop_in_grid(board2, req.col, player)
        win_cells = check_win_cells(board2, pos[0], pos[1], player) if pos else []
        winner = player if win_cells else None
        draw = board_is_draw(board2)

        return {
            "ok": True,
            "grid": board2,
            "row": pos[0] if pos else None,
            "col": req.col,
            "player": player,
            "winner": winner,
            "win_cells": win_cells,
            "draw": draw,
            "next_player": None if winner or draw else other(player),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur play: {e}")


@app.get("/api/new_board")
def api_new_board(rows: int = 9, cols: int = 9):
    if rows < 4 or cols < 4 or rows > 12 or cols > 12:
        raise HTTPException(status_code=400, detail="Taille de grille invalide")

    return {
        "ok": True,
        "grid": make_empty_board(rows, cols),
        "current_player": RED,
    }


# ══════════════════════════════════════════════════════════════
# ONLINE ROOMS
# ══════════════════════════════════════════════════════════════


def make_room_code(length: int = 6) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(secrets.choice(alphabet) for _ in range(length))


def get_room(room_code: str):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM online_rooms WHERE room_code = %s", (room_code,))
    room = cur.fetchone()
    cur.close()
    conn.close()
    return room


@app.post("/api/online/create_room")
@app.post("/api/online/create")
def api_create_room(req: NewRoomRequest):
    try:
        rows = req.rows or 9
        cols = req.cols or 9
        if rows < 4 or cols < 4 or rows > 12 or cols > 12:
            raise HTTPException(status_code=400, detail="Taille invalide")

        board = make_empty_board(rows, cols)

        starting_color = str(req.starting_color or RED).strip().upper()
        if starting_color not in (RED, YELLOW):
            starting_color = RED

        conn = db_conn()
        cur = conn.cursor()

        room_code = None
        for _ in range(10):
            try_code = make_room_code()
            try:
                red_name = req.player_name or "Joueur 1"
                yellow_name = None

                if starting_color == YELLOW:
                    yellow_name = red_name
                    red_name = None

                cur.execute(
                    """
                    INSERT INTO online_rooms
                    (room_code, board_json, current_player, winner, status, player_red_name, player_yellow_name, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                    RETURNING room_code
                    """,
                    (
                        try_code,
                        json.dumps(board),
                        starting_color,
                        None,
                        "waiting",
                        red_name,
                        yellow_name,
                    ),
                )
                created = cur.fetchone()
                conn.commit()
                room_code = created["room_code"]
                break
            except psycopg2.Error:
                conn.rollback()
                continue

        cur.close()
        conn.close()

        if not room_code:
            raise HTTPException(status_code=500, detail="Impossible de créer la room")

        return {
            "ok": True,
            "room_code": room_code,
            "code": room_code,
            "player": starting_color,
            "your_token": starting_color,
            "player_secret": room_code
            + ("_RED" if starting_color == RED else "_YELLOW"),
            "grid": board,
            "rows": rows,
            "cols": cols,
            "starting_color": starting_color,
            "current_player": starting_color,
            "current_turn": starting_color,
            "status": "waiting",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur create_room: {e}")


@app.post("/api/online/join_room")
@app.post("/api/online/join")
def api_join_room(req: JoinRoomRequest):
    try:
        room_code = req.room_code.strip().upper()

        conn = db_conn()
        cur = conn.cursor()

        cur.execute("SELECT * FROM online_rooms WHERE room_code = %s", (room_code,))
        room = cur.fetchone()

        if not room:
            cur.close()
            conn.close()
            raise HTTPException(status_code=404, detail="Salle introuvable")

        # Détermine quelle couleur est libre
        free_token = None
        if not room["player_red_name"]:
            free_token = RED
        elif not room["player_yellow_name"]:
            free_token = YELLOW

        if free_token is None:
            cur.close()
            conn.close()
            raise HTTPException(status_code=400, detail="Salle déjà pleine")

        if free_token == RED:
            cur.execute(
                """
                UPDATE online_rooms
                SET player_red_name = %s,
                    status = 'playing',
                    updated_at = NOW()
                WHERE room_code = %s
                """,
                (req.player_name or "Joueur", room_code),
            )
        else:
            cur.execute(
                """
                UPDATE online_rooms
                SET player_yellow_name = %s,
                    status = 'playing',
                    updated_at = NOW()
                WHERE room_code = %s
                """,
                (req.player_name or "Joueur", room_code),
            )

        conn.commit()
        cur.close()
        conn.close()

        return {
            "ok": True,
            "room_code": room_code,
            "code": room_code,
            "player": free_token,
            "your_token": free_token,
            "player_secret": room_code + ("_RED" if free_token == RED else "_YELLOW"),
            "status": "playing",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur join_room: {e}")


@app.get("/api/online/state/{room_code}")
@app.get("/api/online/{room_code}/state")
def api_room_state(room_code: str):
    try:
        room = get_room(room_code.strip().upper())
        if not room:
            raise HTTPException(status_code=404, detail="Salle introuvable")

        board = json.loads(room["board_json"])
        board = validate_board(board)

        moves = []
        conn = db_conn()
        cur = conn.cursor()
        cur.execute(
            "SELECT player, col, played_at FROM online_moves WHERE room_code = %s ORDER BY id ASC",
            (room_code.strip().upper(),),
        )
        rows_moves = cur.fetchall()
        cur.close()
        conn.close()

        for m in rows_moves:
            moves.append(
                {
                    "player": m["player"],
                    "col": m["col"],
                    "played_at": m["played_at"].isoformat() if m["played_at"] else None,
                }
            )

        players = []
        if room["player_red_name"]:
            players.append({"token": RED, "player_name": room["player_red_name"]})
        if room["player_yellow_name"]:
            players.append({"token": YELLOW, "player_name": room["player_yellow_name"]})

        prediction = None
        if room["winner"] is None and not board_is_draw(board):
            prediction = call_prediction_engine(
                board=board,
                player=room["current_player"],
                depth=8,
                ai_mode="minimax",
            )

        return {
            "ok": True,
            "room_code": room["room_code"],
            "code": room["room_code"],
            "grid": board,
            "rows": len(board),
            "cols": len(board[0]),
            "current_player": room["current_player"],
            "current_turn": room["current_player"],
            "winner": room["winner"],
            "status": room["status"],
            "player_red_name": room["player_red_name"],
            "player_yellow_name": room["player_yellow_name"],
            "players": players,
            "moves": moves,
            "starting_color": RED,
            "prediction": prediction,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur state: {e}")


@app.post("/api/online/play")
@app.post("/api/online/{room_code}/move")
def api_online_play(req: RoomMoveRequest, room_code: Optional[str] = None):
    try:
        room_code = (room_code or req.room_code).strip().upper()
        player = normalize_player(req.player)

        conn = db_conn()
        cur = conn.cursor()

        cur.execute("SELECT * FROM online_rooms WHERE room_code = %s", (room_code,))
        room = cur.fetchone()

        if not room:
            cur.close()
            conn.close()
            raise HTTPException(status_code=404, detail="Salle introuvable")

        if room["winner"]:
            cur.close()
            conn.close()
            raise HTTPException(status_code=400, detail="Partie terminée")

        if room["current_player"] != player:
            cur.close()
            conn.close()
            raise HTTPException(status_code=400, detail="Ce n'est pas ton tour")

        board = json.loads(room["board_json"])
        board = validate_board(board)

        if req.col not in valid_columns(board):
            cur.close()
            conn.close()
            raise HTTPException(status_code=400, detail="Colonne invalide")

        board2 = copy_grid(board)
        pos = drop_in_grid(board2, req.col, player)
        win_cells = check_win_cells(board2, pos[0], pos[1], player) if pos else []
        winner = player if win_cells else None
        draw = board_is_draw(board2)
        next_player = None if winner or draw else other(player)
        status = "finished" if winner or draw else "playing"

        cur.execute(
            """
            UPDATE online_rooms
            SET board_json = %s,
                current_player = %s,
                winner = %s,
                status = %s,
                updated_at = NOW()
            WHERE room_code = %s
            """,
            (
                json.dumps(board2),
                next_player if next_player else room["current_player"],
                winner,
                status,
                room_code,
            ),
        )

        cur.execute(
            """
            INSERT INTO online_moves (room_code, player, col)
            VALUES (%s, %s, %s)
            """,
            (room_code, player, req.col),
        )

        conn.commit()
        cur.close()
        conn.close()

        prediction = None
        if not winner and not draw and next_player:
            prediction = call_prediction_engine(
                board=board2,
                player=next_player,
                depth=8,
                ai_mode="minimax",
            )

        return {
            "ok": True,
            "grid": board2,
            "row": pos[0] if pos else None,
            "col": req.col,
            "player": player,
            "winner": winner,
            "win_cells": win_cells,
            "draw": draw,
            "next_player": next_player,
            "status": status,
            "prediction": prediction,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur online play: {e}")


@app.post("/api/online/ai_move")
def api_online_ai_move(req: RoomAIMoveRequest):
    try:
        room_code = req.room_code.strip().upper()
        player = normalize_player(req.player)
        ai_mode = normalize_mode(req.ai_mode)
        depth = req.depth or 6

        conn = db_conn()
        cur = conn.cursor()

        cur.execute("SELECT * FROM online_rooms WHERE room_code = %s", (room_code,))
        room = cur.fetchone()

        if not room:
            cur.close()
            conn.close()
            raise HTTPException(status_code=404, detail="Salle introuvable")

        if room["winner"]:
            cur.close()
            conn.close()
            raise HTTPException(status_code=400, detail="Partie terminée")

        if room["current_player"] != player:
            cur.close()
            conn.close()
            raise HTTPException(status_code=400, detail="Ce n'est pas le tour de l'IA")

        board = json.loads(room["board_json"])
        board = validate_board(board)

        result = best_move(board, player, depth, ai_mode)

        col = result.get("col")
        if col is None:
            cur.close()
            conn.close()
            raise HTTPException(status_code=400, detail="Aucun coup possible")

        if col not in valid_columns(board):
            cur.close()
            conn.close()
            raise HTTPException(status_code=400, detail="Coup IA invalide")

        board2 = copy_grid(board)
        pos = drop_in_grid(board2, col, player)
        win_cells = check_win_cells(board2, pos[0], pos[1], player) if pos else []
        winner = player if win_cells else None
        draw = board_is_draw(board2)
        next_player = None if winner or draw else other(player)
        status = "finished" if winner or draw else "playing"

        cur.execute(
            """
            UPDATE online_rooms
            SET board_json = %s,
                current_player = %s,
                winner = %s,
                status = %s,
                updated_at = NOW()
            WHERE room_code = %s
            """,
            (
                json.dumps(board2),
                next_player if next_player else room["current_player"],
                winner,
                status,
                room_code,
            ),
        )

        cur.execute(
            """
            INSERT INTO online_moves (room_code, player, col)
            VALUES (%s, %s, %s)
            """,
            (room_code, player, col),
        )

        conn.commit()
        cur.close()
        conn.close()

        prediction = None
        if not winner and not draw and next_player:
            prediction = call_prediction_engine(
                board=board2,
                player=next_player,
                depth=8,
                ai_mode=ai_mode,
            )

        return {
            "ok": True,
            "grid": board2,
            "row": pos[0] if pos else None,
            "col": col,
            "player": player,
            "winner": winner,
            "win_cells": win_cells,
            "draw": draw,
            "next_player": next_player,
            "status": status,
            "mode": result.get("source", ai_mode),
            "reason": result.get("source"),
            "scores": result.get("scores"),
            "prediction": prediction,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur online ai_move: {e}")


@app.get("/api/online/invite/{room_code}")
def api_online_invite(room_code: str):
    room_code = room_code.strip().upper()
    room = get_room(room_code)
    if not room:
        raise HTTPException(status_code=404, detail="Salle introuvable")

    return {
        "ok": True,
        "room_code": room["room_code"],
        "status": room["status"],
        "player_red_name": room["player_red_name"],
        "player_yellow_name": room["player_yellow_name"],
    }


# ══════════════════════════════════════════════════════════════
# STATIC
# ══════════════════════════════════════════════════════════════

STATIC_DIR = os.path.join(BASE_DIR, "public")

if os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
    print(f"[STATIC] OK: {STATIC_DIR}")
else:
    print("[STATIC] dossier public introuvable")
