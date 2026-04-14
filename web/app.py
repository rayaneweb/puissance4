# app.py — Backend FastAPI pour Puissance 4
# Routes principales :
# - GET  /api/health
# - POST /api/ai/move
# - POST /api/predict
# - POST /api/ai/reload
# - GET  /api/games
# - GET  /api/games/{game_id}
# - POST /api/games
# - DELETE /api/games/{game_id}
# - GET  /
# - fichiers statiques dans ./public

import os
import json
from datetime import datetime
from typing import List, Optional, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import psycopg2
import psycopg2.extras

from ia_engine import (
    best_move,
    predict_outcome,
    check_win_cells,
    valid_columns,
    drop_in_grid,
    copy_grid,
    is_draw,
    other,
    EMPTY,
    RED,
    YELLOW,
    reload_model,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DIR = os.path.join(BASE_DIR, "public")

DATABASE_URL = os.environ.get("DATABASE_URL")

app = FastAPI(title="Puissance 4 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════
# MODELS
# ══════════════════════════════════════════════════════════════


class AIMoveRequest(BaseModel):
    board: List[List[str]]
    player: str = Field(..., pattern="^[RY]$")
    ai_mode: str = "minimax"
    depth: int = 4


class PredictRequest(BaseModel):
    board: List[List[str]]
    player: str = Field(..., pattern="^[RY]$")
    depth: int = 4


class SaveGameRequest(BaseModel):
    user_id: Optional[int] = 1
    save_name: str = "partie"
    game_index: int = 1
    rows_count: int
    cols_count: int
    starting_color: str = Field(..., pattern="^[RY]$")
    control_red: str = "human"
    control_yellow: str = "ai"
    ai_mode: str = "random"
    ai_depth: int = 4
    game_mode: int = 1
    status: str = "in_progress"
    winner: Optional[str] = None
    view_index: int = 0
    moves: List[int] = []
    confidence: int = 1
    distinct_cols: int = 0
    player_red: Optional[str] = "Joueur Rouge"
    player_yellow: Optional[str] = "Joueur Jaune"


# ══════════════════════════════════════════════════════════════
# DB
# ══════════════════════════════════════════════════════════════


def get_db_conn():
    if not DATABASE_URL:
        return None
    return psycopg2.connect(DATABASE_URL, sslmode="require")


def ensure_db():
    conn = get_db_conn()
    if conn is None:
        print("[app] DATABASE_URL absent -> stockage DB désactivé")
        return

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS saved_games (
                        game_id SERIAL PRIMARY KEY,
                        user_id INTEGER DEFAULT 1,
                        save_name TEXT NOT NULL DEFAULT 'partie',
                        game_index INTEGER NOT NULL DEFAULT 1,
                        rows_count INTEGER NOT NULL CHECK (rows_count BETWEEN 4 AND 20),
                        cols_count INTEGER NOT NULL CHECK (cols_count BETWEEN 4 AND 20),
                        starting_color CHAR(1) NOT NULL CHECK (starting_color IN ('R','Y')),
                        control_red TEXT NOT NULL DEFAULT 'human',
                        control_yellow TEXT NOT NULL DEFAULT 'ai',
                        ai_mode TEXT NOT NULL DEFAULT 'random',
                        ai_depth INTEGER NOT NULL DEFAULT 4,
                        game_mode INTEGER NOT NULL DEFAULT 1,
                        status TEXT NOT NULL DEFAULT 'in_progress',
                        winner CHAR(1),
                        view_index INTEGER NOT NULL DEFAULT 0,
                        moves JSONB NOT NULL DEFAULT '[]'::jsonb,
                        confidence INTEGER NOT NULL DEFAULT 1,
                        distinct_cols INTEGER NOT NULL DEFAULT 0,
                        player_red TEXT,
                        player_yellow TEXT,
                        created_at TIMESTAMP NOT NULL DEFAULT NOW()
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_saved_games_created_at
                    ON saved_games(created_at DESC);
                    """
                )
        print("[app] Base prête")
    finally:
        conn.close()


@app.on_event("startup")
def startup():
    ensure_db()


# ══════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════


def validate_board(board: List[List[str]]) -> tuple[int, int]:
    if not board or not isinstance(board, list):
        raise HTTPException(status_code=400, detail="board invalide")
    if not all(isinstance(row, list) for row in board):
        raise HTTPException(status_code=400, detail="board invalide")

    rows = len(board)
    cols = len(board[0]) if rows else 0

    if rows < 4 or cols < 4:
        raise HTTPException(status_code=400, detail="taille de plateau invalide")

    for row in board:
        if len(row) != cols:
            raise HTTPException(status_code=400, detail="board non rectangulaire")
        for cell in row:
            if cell not in (EMPTY, RED, YELLOW):
                raise HTTPException(status_code=400, detail="cellule invalide")

    return rows, cols


# ══════════════════════════════════════════════════════════════
# ROUTES API
# ══════════════════════════════════════════════════════════════


@app.get("/api/health")
def api_health():
    return {
        "ok": True,
        "time": datetime.utcnow().isoformat() + "Z",
        "database": bool(DATABASE_URL),
    }


@app.post("/api/ai/move")
def api_ai_move(payload: AIMoveRequest):
    validate_board(payload.board)

    depth = max(1, min(int(payload.depth), 12))
    ai_mode = (payload.ai_mode or "minimax").lower()

    try:
        result = best_move(
            board=payload.board,
            player=payload.player,
            depth=depth,
            ai_mode=ai_mode,
            time_limit_ms=1800,
        )

        return {
            "col": result.get("col"),
            "scores": result.get("scores", {}),
            "player": payload.player,
            "ai_mode": ai_mode,
            "depth": depth,
            "source": result.get("source", "unknown"),
            "depth_reached": result.get("depth_reached", 0),
            "distance": result.get("distance"),
            "distances": result.get("distances", {}),
        }
    except Exception as e:
        print(f"[app] /api/ai/move error: {e}")
        return JSONResponse(
            status_code=200,
            content={
                "col": None,
                "scores": {},
                "player": payload.player,
                "ai_mode": ai_mode,
                "depth": depth,
                "source": f"error:{type(e).__name__}",
                "depth_reached": 0,
                "distance": None,
                "distances": {},
            },
        )


@app.post("/api/predict")
def api_predict(payload: PredictRequest):
    validate_board(payload.board)

    depth = max(1, min(int(payload.depth), 12))

    try:
        result = predict_outcome(
            board=payload.board,
            player=payload.player,
            depth=depth,
            time_limit_ms=1800,
        )

        return {
            "winner": result.get("winner"),
            "moves": result.get("moves"),
            "mateIn": result.get("mateIn"),
            "score": result.get("score", 0),
            "depth_reached": result.get("depth_reached", 0),
            "best_col": result.get("best_col"),
            "source": result.get("source", "predict"),
            "exact": bool(result.get("exact", False)),
        }
    except Exception as e:
        print(f"[app] /api/predict error: {e}")
        # Très important : on renvoie 200 avec un objet propre
        # pour éviter le HTTP 502 dans le front.
        return {
            "winner": None,
            "moves": None,
            "mateIn": None,
            "score": 0,
            "depth_reached": 0,
            "best_col": None,
            "source": f"error:{type(e).__name__}",
            "exact": False,
        }


@app.post("/api/ai/reload")
def api_ai_reload():
    ok = reload_model()
    return {"ok": ok}


@app.get("/api/games")
def api_list_games():
    conn = get_db_conn()
    if conn is None:
        return []

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    game_id,
                    user_id,
                    save_name,
                    game_index,
                    rows_count,
                    cols_count,
                    starting_color,
                    control_red,
                    control_yellow,
                    ai_mode,
                    ai_depth,
                    game_mode,
                    status,
                    winner,
                    view_index,
                    moves,
                    confidence,
                    distinct_cols,
                    player_red,
                    player_yellow,
                    created_at
                FROM saved_games
                ORDER BY created_at DESC
                LIMIT 500
                """
            )
            rows = cur.fetchall()

        out = []
        for row in rows:
            item = dict(row)
            if isinstance(item.get("moves"), str):
                try:
                    item["moves"] = json.loads(item["moves"])
                except Exception:
                    item["moves"] = []
            item["total_moves"] = len(item.get("moves") or [])
            out.append(item)

        return out
    finally:
        conn.close()


@app.get("/api/games/{game_id}")
def api_get_game(game_id: int):
    conn = get_db_conn()
    if conn is None:
        raise HTTPException(status_code=404, detail="base indisponible")

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    game_id,
                    user_id,
                    save_name,
                    game_index,
                    rows_count,
                    cols_count,
                    starting_color,
                    control_red,
                    control_yellow,
                    ai_mode,
                    ai_depth,
                    game_mode,
                    status,
                    winner,
                    view_index,
                    moves,
                    confidence,
                    distinct_cols,
                    player_red,
                    player_yellow,
                    created_at
                FROM saved_games
                WHERE game_id = %s
                """,
                (game_id,),
            )
            row = cur.fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="partie introuvable")

        item = dict(row)
        if isinstance(item.get("moves"), str):
            try:
                item["moves"] = json.loads(item["moves"])
            except Exception:
                item["moves"] = []
        return item
    finally:
        conn.close()


@app.post("/api/games")
def api_save_game(payload: SaveGameRequest):
    conn = get_db_conn()
    if conn is None:
        return {"ok": False, "message": "base indisponible"}

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO saved_games (
                        user_id,
                        save_name,
                        game_index,
                        rows_count,
                        cols_count,
                        starting_color,
                        control_red,
                        control_yellow,
                        ai_mode,
                        ai_depth,
                        game_mode,
                        status,
                        winner,
                        view_index,
                        moves,
                        confidence,
                        distinct_cols,
                        player_red,
                        player_yellow
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s
                    )
                    RETURNING game_id
                    """,
                    (
                        payload.user_id,
                        payload.save_name,
                        payload.game_index,
                        payload.rows_count,
                        payload.cols_count,
                        payload.starting_color,
                        payload.control_red,
                        payload.control_yellow,
                        payload.ai_mode,
                        payload.ai_depth,
                        payload.game_mode,
                        payload.status,
                        payload.winner,
                        payload.view_index,
                        json.dumps(payload.moves),
                        payload.confidence,
                        payload.distinct_cols,
                        payload.player_red,
                        payload.player_yellow,
                    ),
                )
                game_id = cur.fetchone()[0]

        return {"ok": True, "game_id": game_id}
    finally:
        conn.close()


@app.delete("/api/games/{game_id}")
def api_delete_game(game_id: int):
    conn = get_db_conn()
    if conn is None:
        raise HTTPException(status_code=404, detail="base indisponible")

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM saved_games WHERE game_id = %s", (game_id,))
                deleted = cur.rowcount

        if deleted == 0:
            raise HTTPException(status_code=404, detail="partie introuvable")

        return {"ok": True, "deleted": game_id}
    finally:
        conn.close()


# ══════════════════════════════════════════════════════════════
# FICHIERS STATIQUES
# ══════════════════════════════════════════════════════════════

if os.path.isdir(PUBLIC_DIR):
    app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="public")


@app.get("/")
def root():
    index_path = os.path.join(PUBLIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"ok": True, "message": "public/index.html introuvable"}
