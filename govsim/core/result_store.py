"""
ResultStore — the queryable ``runs`` table + per-run series artifacts + one generic plotter.

This is the research harness doc-08 §3.2 flagged as missing: a row per ``RunRecord`` keyed by
``(experiment, system, action_interface, regent_specs, objective, seed, git_commit, hypothesis)``
so results can be aggregated across runs (paired-seed stats, ablation curves), with the heavy
per-step series + raw LLM I/O written as side artifacts.

Storage: sqlite (stdlib, queryable) for the table; CSV for the metric series and JSON for the
LLM I/O (dependency-free — pandas is already a dep, parquet/pyarrow is deliberately avoided to
keep CI light; see govsim/docs_gates/decisions.md). One ``plot`` method renders any series keys.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from govsim.core.experiment import RunRecord

_COLUMNS = [
    "experiment", "system_id", "action_interface_id", "schedule_id", "hypothesis_id",
    "seed", "horizon", "git_commit", "regent_specs", "objective_ids", "score", "components",
    "creativity_metric", "token_cost", "terminated_at_step", "created_at", "series_path", "llm_io_path",
]
_JSON_COLUMNS = {"regent_specs", "objective_ids", "score", "components"}


class ResultStore:
    def __init__(self, root: str | Path = "logs/runs") -> None:
        self.root = Path(root)
        self.artifacts = self.root / "artifacts"
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "runs.db"
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        cols = ", ".join(f"{c} TEXT" if c in _JSON_COLUMNS or c.endswith("_path") else f"{c}" for c in _COLUMNS)
        self._conn.execute(f"CREATE TABLE IF NOT EXISTS runs (run_id INTEGER PRIMARY KEY AUTOINCREMENT, {cols})")
        self._conn.commit()

    def add(self, record: RunRecord) -> int:
        """Insert a run row + write its series/LLM-I/O artifacts. Returns the run_id."""
        row = {
            "experiment": record.experiment,
            "system_id": record.system_id,
            "action_interface_id": record.action_interface_id,
            "schedule_id": record.schedule_id,
            "hypothesis_id": record.hypothesis_id,
            "seed": record.seed,
            "horizon": record.horizon,
            "git_commit": record.git_commit,
            "regent_specs": json.dumps(record.regent_specs),
            "objective_ids": json.dumps(record.objective_ids),
            "score": json.dumps(record.score),
            "components": json.dumps(record.components),
            "creativity_metric": record.creativity_metric,
            "token_cost": record.token_cost,
            "terminated_at_step": record.terminated_at_step,
            "created_at": record.created_at,
            "series_path": None,
            "llm_io_path": None,
        }
        placeholders = ", ".join("?" for _ in _COLUMNS)
        cur = self._conn.execute(
            f"INSERT INTO runs ({', '.join(_COLUMNS)}) VALUES ({placeholders})",
            [row[c] for c in _COLUMNS],
        )
        run_id = int(cur.lastrowid)

        series_path = self._write_series(run_id, record.metrics_series)
        llm_path = self._write_llm_io(run_id, record.llm_io)
        self._conn.execute(
            "UPDATE runs SET series_path=?, llm_io_path=? WHERE run_id=?",
            [str(series_path) if series_path else None, str(llm_path) if llm_path else None, run_id],
        )
        self._conn.commit()
        return run_id

    def _write_series(self, run_id: int, series: list[dict[str, float]]) -> Path | None:
        if not series:
            return None
        import pandas as pd

        path = self.artifacts / f"run_{run_id}_series.csv"
        pd.DataFrame(series).to_csv(path, index=False)
        return path

    def _write_llm_io(self, run_id: int, llm_io: list[dict[str, Any]]) -> Path | None:
        if not llm_io:
            return None
        path = self.artifacts / f"run_{run_id}_llm_io.json"
        path.write_text(json.dumps(llm_io, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def query(self, **filters: Any) -> list[dict[str, Any]]:
        """Return run rows matching equality ``filters`` (e.g. ``experiment=...``, ``seed=...``)."""
        where = " AND ".join(f"{k}=?" for k in filters)
        sql = "SELECT * FROM runs" + (f" WHERE {where}" if where else "") + " ORDER BY run_id"
        rows = self._conn.execute(sql, list(filters.values())).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            for c in _JSON_COLUMNS:
                if d.get(c):
                    d[c] = json.loads(d[c])
            out.append(d)
        return out

    def load_series(self, run_id: int) -> list[dict[str, float]]:
        row = self._conn.execute("SELECT series_path FROM runs WHERE run_id=?", [run_id]).fetchone()
        if not row or not row["series_path"]:
            return []
        import pandas as pd

        return pd.read_csv(row["series_path"]).to_dict(orient="records")

    def plot(self, run_id: int, keys: list[str] | None = None, out_path: str | Path | None = None) -> Path:
        """Schema-driven line plot of a run's series (one axis per key vs step)."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        series = self.load_series(run_id)
        if not series:
            raise ValueError(f"run {run_id} has no series to plot")
        if keys is None:
            keys = [k for k in series[0] if k != "step"]
        steps = [row.get("step", i) for i, row in enumerate(series)]
        fig, ax = plt.subplots(figsize=(9, 5))
        for k in keys:
            ax.plot(steps, [row.get(k) for row in series], label=k)
        ax.set_xlabel("step")
        ax.legend()
        ax.set_title(f"run {run_id}")
        out_path = Path(out_path) if out_path else (self.artifacts / f"run_{run_id}_plot.png")
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def close(self) -> None:
        self._conn.close()
