import json
import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal


JobType = Literal["analyze", "scan_etf", "scan_stock"]
JobStatus = Literal["queued", "running", "succeeded", "failed"]


@dataclass
class Job:
    job_id: str
    job_type: JobType
    created_at: float
    started_at: float | None
    ended_at: float | None
    status: JobStatus
    cmd: list[str]
    out_dir: Path
    log_path: Path
    process: subprocess.Popen | None
    return_code: int | None
    error: str | None


class JobManager:
    def __init__(self, *, outputs_root: Path):
        self._lock = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._outputs_root = outputs_root

    def create_job_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def _safe_out_dir(self, job_id: str) -> Path:
        return (self._outputs_root / f"web_{job_id}").resolve()

    def start_subprocess_job(
        self,
        *,
        job_type: JobType,
        cmd: list[str],
        job_id: str | None = None,
        out_dir: Path | None = None,
    ) -> Job:
        job_id2 = job_id or self.create_job_id()
        out_dir2 = out_dir.resolve() if out_dir is not None else self._safe_out_dir(job_id2)
        if self._outputs_root not in out_dir2.parents and out_dir2 != self._outputs_root:
            raise ValueError("out_dir 必须在 outputs/ 目录下")
        out_dir2.mkdir(parents=True, exist_ok=True)
        log_path = out_dir2 / "job.log"

        job = Job(
            job_id=job_id2,
            job_type=job_type,
            created_at=time.time(),
            started_at=None,
            ended_at=None,
            status="queued",
            cmd=cmd,
            out_dir=out_dir2,
            log_path=log_path,
            process=None,
            return_code=None,
            error=None,
        )

        with self._lock:
            self._jobs[job_id2] = job

        # 直接起子进程，别在 FastAPI worker 里硬跑，省得阻塞
        def runner() -> None:
            with self._lock:
                job.status = "running"
                job.started_at = time.time()

            try:
                with log_path.open("wb") as f:
                    proc = subprocess.Popen(
                        cmd,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                        cwd=str(self._outputs_root.parent),
                        env=os.environ.copy(),
                    )
                    with self._lock:
                        job.process = proc

                    rc = proc.wait()
                    with self._lock:
                        job.return_code = int(rc)
                        job.ended_at = time.time()
                        job.status = "succeeded" if rc == 0 else "failed"
                        if rc != 0:
                            job.error = f"子进程退出码={rc}"
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    job.return_code = -1
                    job.ended_at = time.time()
                    job.status = "failed"
                    job.error = str(exc)

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        return job

    def get_job(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            return job

    def list_artifacts(self, out_dir: Path) -> dict[str, list[str]]:
        if not out_dir.exists():
            return {"png": [], "json": [], "other": []}
        png: list[str] = []
        js: list[str] = []
        other: list[str] = []
        for p in out_dir.rglob("*"):
            if not p.is_file():
                continue
            rel = str(p.relative_to(out_dir)).replace("\\", "/")
            if rel == "job.log":
                continue
            if p.suffix.lower() == ".png":
                png.append(rel)
            elif p.suffix.lower() == ".json":
                js.append(rel)
            else:
                other.append(rel)
        png.sort()
        js.sort()
        other.sort()
        return {"png": png, "json": js, "other": other}

    def tail_log(self, log_path: Path, *, max_bytes: int = 6000) -> str:
        if not log_path.exists():
            return ""
        try:
            data = log_path.read_bytes()
            if len(data) > max_bytes:
                data = data[-max_bytes:]
            return data.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return ""


def create_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field

    from .json_utils import sanitize_for_json

    root = Path(__file__).resolve().parents[1]
    outputs_root = (root / "outputs").resolve()
    outputs_root.mkdir(parents=True, exist_ok=True)

    static_root = (root / "llm_trading" / "web_static").resolve()
    if not static_root.exists():
        raise RuntimeError(f"缺少前端静态文件目录：{static_root}")

    jobs = JobManager(outputs_root=outputs_root)
    app = FastAPI(title="LLM辅助交易 - Web", version="0.1.0")

    app.mount("/static", StaticFiles(directory=str(static_root)), name="static")
    app.mount("/outputs", StaticFiles(directory=str(outputs_root)), name="outputs")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(str(static_root / "index.html"), headers={"Cache-Control": "no-store"})

    @app.get("/api/health")
    def health():
        return {
            "ok": True,
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "cwd": str(root),
            "time": datetime.now().isoformat(),
        }

    class AnalyzeRequest(BaseModel):
        asset: Literal["stock", "etf", "index"] = "stock"
        symbol: str = Field(min_length=1)
        freq: Literal["daily", "weekly"] = "weekly"
        method: Literal["wyckoff", "chan", "ichimoku", "turtle", "momentum", "dow", "vsa", "institution", "both", "all"] = "all"
        window: int = 500
        start_date: str | None = None
        end_date: str | None = None
        adjust: str | None = None
        narrate: bool = False
        narrate_provider: Literal["gemini", "openai"] = "openai"
        narrate_schools: str = "chan,wyckoff,ichimoku,turtle,momentum"
        narrate_temperature: float = 0.2
        narrate_max_output_tokens: int = 1200

    class ScanEtfRequest(BaseModel):
        freq: Literal["daily", "weekly"] = "weekly"
        window: int = 400
        min_weeks: int = 60
        min_amount: float = 0.0
        min_amount_avg20: float = 0.0
        limit: int = 0
        top_k: int = 30
        workers: int = 8

    class ScanStockRequest(BaseModel):
        freq: Literal["daily", "weekly"] = "weekly"
        window: int = 500
        start_date: str = "20100101"
        end_date: str | None = None
        adjust: str | None = None
        daily_filter: Literal["none", "ma20", "macd"] = "macd"
        base_filters: str = "trend_template"
        tt_near_high: float = 0.25
        tt_above_low: float = 0.30
        tt_slope_weeks: int = 4
        horizons: str = "4,8,12"
        rank_horizon: int = 8
        min_weeks: int = 120
        min_trades: int = 12
        min_price: float = 0.0
        max_price: float = 0.0
        min_amount: float = 0.0
        limit: int = 0
        top_k: int = 50
        workers: int = 8
        buy_cost: float = 0.001
        sell_cost: float = 0.002
        include_st: bool = False
        exclude_bj: bool = False
        cache_ttl_hours: float = 24.0

    def _cmd_analyze(payload: AnalyzeRequest, out_dir: Path) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "llm_trading",
            "analyze",
            "--asset",
            payload.asset,
            "--symbol",
            payload.symbol,
            "--freq",
            payload.freq,
            "--method",
            payload.method,
            "--window",
            str(int(payload.window)),
            "--out-dir",
            str(out_dir),
        ]
        if payload.start_date:
            cmd += ["--start-date", payload.start_date]
        if payload.end_date:
            cmd += ["--end-date", payload.end_date]
        if payload.adjust is not None and payload.adjust != "":
            cmd += ["--adjust", payload.adjust]
        if payload.narrate:
            cmd += [
                "--narrate",
                "--narrate-provider",
                payload.narrate_provider,
                "--narrate-schools",
                payload.narrate_schools,
                "--narrate-temperature",
                str(float(payload.narrate_temperature)),
                "--narrate-max-output-tokens",
                str(int(payload.narrate_max_output_tokens)),
            ]
        return cmd

    def _cmd_scan_etf(payload: ScanEtfRequest, out_dir: Path) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "llm_trading",
            "scan-etf",
            "--freq",
            payload.freq,
            "--window",
            str(int(payload.window)),
            "--min-weeks",
            str(int(payload.min_weeks)),
            "--min-amount",
            str(float(payload.min_amount)),
            "--min-amount-avg20",
            str(float(payload.min_amount_avg20)),
            "--top-k",
            str(int(payload.top_k)),
            "--workers",
            str(int(payload.workers)),
            "--out-dir",
            str(out_dir),
            "--verbose",
        ]
        if payload.limit and int(payload.limit) > 0:
            cmd += ["--limit", str(int(payload.limit))]
        return cmd

    def _cmd_scan_stock(payload: ScanStockRequest, out_dir: Path) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "llm_trading",
            "scan-stock",
            "--freq",
            payload.freq,
            "--window",
            str(int(payload.window)),
            "--start-date",
            payload.start_date,
            "--daily-filter",
            payload.daily_filter,
            "--base-filters",
            payload.base_filters,
            "--tt-near-high",
            str(float(payload.tt_near_high)),
            "--tt-above-low",
            str(float(payload.tt_above_low)),
            "--tt-slope-weeks",
            str(int(payload.tt_slope_weeks)),
            "--horizons",
            payload.horizons,
            "--rank-horizon",
            str(int(payload.rank_horizon)),
            "--min-weeks",
            str(int(payload.min_weeks)),
            "--min-trades",
            str(int(payload.min_trades)),
            "--min-price",
            str(float(payload.min_price)),
            "--max-price",
            str(float(payload.max_price)),
            "--min-amount",
            str(float(payload.min_amount)),
            "--top-k",
            str(int(payload.top_k)),
            "--workers",
            str(int(payload.workers)),
            "--buy-cost",
            str(float(payload.buy_cost)),
            "--sell-cost",
            str(float(payload.sell_cost)),
            "--cache-ttl-hours",
            str(float(payload.cache_ttl_hours)),
            "--out-dir",
            str(out_dir),
            "--verbose",
        ]
        if payload.end_date:
            cmd += ["--end-date", payload.end_date]
        if payload.adjust is not None and payload.adjust != "":
            cmd += ["--adjust", payload.adjust]
        if payload.limit and int(payload.limit) > 0:
            cmd += ["--limit", str(int(payload.limit))]
        if payload.include_st:
            cmd += ["--include-st"]
        if payload.exclude_bj:
            cmd += ["--exclude-bj"]
        return cmd

    @app.post("/api/jobs/analyze")
    def start_analyze(payload: AnalyzeRequest):
        job_id = jobs.create_job_id()
        out_dir = (outputs_root / f"web_{job_id}").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = _cmd_analyze(payload, out_dir)

        job = jobs.start_subprocess_job(job_type="analyze", cmd=cmd, job_id=job_id, out_dir=out_dir)
        return {"job_id": job.job_id, "out_dir": f"/outputs/{job.out_dir.name}"}

    @app.post("/api/jobs/scan-etf")
    def start_scan_etf(payload: ScanEtfRequest):
        job_id = jobs.create_job_id()
        out_dir = (outputs_root / f"web_{job_id}").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = _cmd_scan_etf(payload, out_dir)

        job = jobs.start_subprocess_job(job_type="scan_etf", cmd=cmd, job_id=job_id, out_dir=out_dir)
        return {"job_id": job.job_id, "out_dir": f"/outputs/{job.out_dir.name}"}

    @app.post("/api/jobs/scan-stock")
    def start_scan_stock(payload: ScanStockRequest):
        job_id = jobs.create_job_id()
        out_dir = (outputs_root / f"web_{job_id}").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = _cmd_scan_stock(payload, out_dir)

        job = jobs.start_subprocess_job(job_type="scan_stock", cmd=cmd, job_id=job_id, out_dir=out_dir)
        return {"job_id": job.job_id, "out_dir": f"/outputs/{job.out_dir.name}"}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        job = jobs.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")

        artifacts = jobs.list_artifacts(job.out_dir) if job.status in {"succeeded", "failed"} else {"png": [], "json": [], "other": []}
        return JSONResponse(
            {
                "job_id": job.job_id,
                "type": job.job_type,
                "status": job.status,
                "return_code": job.return_code,
                "created_at": job.created_at,
                "started_at": job.started_at,
                "ended_at": job.ended_at,
                "cmd": job.cmd,
                "out_dir": f"/outputs/{job.out_dir.name}",
                "error": job.error,
                "artifacts": artifacts,
                "log_tail": jobs.tail_log(job.log_path),
            }
        )

    @app.get("/api/jobs/{job_id}/result")
    def job_result(job_id: str):
        job = jobs.get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        if job.status != "succeeded":
            raise HTTPException(status_code=409, detail=f"job not succeeded: {job.status}")

        # 读常见文件给前端（防止它一堆 fetch）
        payload: dict[str, Any] = {"artifacts": jobs.list_artifacts(job.out_dir)}
        for rel in payload["artifacts"]["json"][:20]:
            p = job.out_dir / rel
            try:
                payload.setdefault("json", {})[rel] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
        return sanitize_for_json(payload)

    return app
