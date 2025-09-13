import os
import json
import time

import pytest

from app import server, TRANSFORMS_DATA


def test_api_kpis_returns_json(tmp_path):
    client = server.test_client()
    # Ensure ETL has had a chance to populate TRANSFORMS_DATA in this process
    # (initialize_data is executed at import-time in app.__init__)
    resp = client.get('/api/kpis')
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, dict)

    # Check snapshot file exists (may be written to cache-directory)
    snapshot_path = os.path.join(os.getcwd(), 'cache-directory', 'kpis_snapshot.json')
    # Allow slight delay if ETL is still writing
    for _ in range(5):
        if os.path.exists(snapshot_path):
            break
        time.sleep(0.2)
    assert os.path.exists(snapshot_path), 'kpis_snapshot.json not written'

    with open(snapshot_path, 'r', encoding='utf-8') as fh:
        snap = json.load(fh)
    assert isinstance(snap, dict)
    # Basic parity: keys in snapshot should be subset of returned data or vice versa
    assert set(snap.keys()).issubset(set(data.keys())) or set(data.keys()).issubset(set(snap.keys()))
