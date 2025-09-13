from typing import Dict, Any
from sqlalchemy import Table, Column, Integer, String, DateTime, MetaData, create_engine, Text
from sqlalchemy.sql import select
from datetime import datetime
import json
from services.db import get_engine
from config.settings import MODEL_STORE_PATH


def _ensure_registry_table():
    engine = get_engine()
    meta = MetaData()
    registry = Table('model_registry', meta,
                     Column('id', Integer, primary_key=True, autoincrement=True),
                     Column('model_name', String(100), nullable=False),
                     Column('artifact_path', String(1024), nullable=False),
                     Column('trained_at', DateTime, nullable=False),
                     Column('metrics', Text, nullable=True),
                     Column('metadata_json', Text, nullable=True)
    )
    job_runs = Table('job_runs', meta,
                     Column('job_id', String(64), primary_key=True),
                     Column('status', String(32), nullable=False),
                     Column('details', Text, nullable=True),
                     Column('updated_at', DateTime, nullable=False)
    )
    meta.create_all(engine, checkfirst=True)
    return registry, engine, job_runs


def register_model(model_name: str, artifact_path: str, metrics: Dict[str, Any], metadata: Dict[str, Any] = None):
    registry, engine, job_runs = _ensure_registry_table()
    ins = registry.insert().values(
        model_name=model_name,
        artifact_path=artifact_path,
        trained_at=datetime.utcnow(),
        metrics=json.dumps(metrics or {}),
        metadata_json=json.dumps(metadata or {})
    )
    conn = engine.connect()
    with conn.begin():
        conn.execute(ins)
    conn.close()


def list_models():
    registry, engine, job_runs = _ensure_registry_table()
    conn = engine.connect()
    sel = select(registry.c.id, registry.c.model_name, registry.c.artifact_path, registry.c.trained_at, registry.c.metrics)
    res = conn.execute(sel).fetchall()
    conn.close()
    return [dict(r._mapping) for r in res]


def register_job(job_id: str, status: str, details: Dict[str, Any] = None):
    """Insert or update a job_runs entry for job state persistence."""
    registry, engine, job_runs = _ensure_registry_table()
    conn = engine.connect()
    try:
        # Check if exists
        sel = select(job_runs.c.job_id).where(job_runs.c.job_id == job_id)
        exists = conn.execute(sel).fetchone()
        payload = {
            'job_id': job_id,
            'status': status,
            'details': json.dumps(details or {}),
            'updated_at': datetime.utcnow()
        }
        if exists:
            upd = job_runs.update().where(job_runs.c.job_id == job_id).values(**payload)
            with conn.begin():
                conn.execute(upd)
        else:
            ins = job_runs.insert().values(**payload)
            with conn.begin():
                conn.execute(ins)
    finally:
        conn.close()


def list_jobs_db():
    """Return dict of job_id -> payload stored in DB."""
    registry, engine, job_runs = _ensure_registry_table()
    conn = engine.connect()
    sel = select(job_runs.c.job_id, job_runs.c.status, job_runs.c.details, job_runs.c.updated_at)
    res = conn.execute(sel).fetchall()
    conn.close()
    out = {}
    for r in res:
        rec = dict(r._mapping)
        try:
            rec['details'] = json.loads(rec.get('details') or '{}')
        except Exception:
            rec['details'] = rec.get('details')
        out[rec['job_id']] = rec
    return out


def get_job_db(job_id: str):
    registry, engine, job_runs = _ensure_registry_table()
    conn = engine.connect()
    sel = select(job_runs.c.job_id, job_runs.c.status, job_runs.c.details, job_runs.c.updated_at).where(job_runs.c.job_id == job_id)
    res = conn.execute(sel).fetchone()
    conn.close()
    if not res:
        return {}
    rec = dict(res._mapping)
    try:
        rec['details'] = json.loads(rec.get('details') or '{}')
    except Exception:
        rec['details'] = rec.get('details')
    return rec
