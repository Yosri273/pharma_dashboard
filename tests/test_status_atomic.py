import os
import time
import json
from services.training import enqueue_training, update_status


def tiny_job(job_id=None):
    update_status(job_id, 'running', {'percent': 10})
    time.sleep(0.1)
    update_status(job_id, 'success', {'percent': 100})
    return True


def test_status_file_written_and_valid_json(tmp_path):
    # Ensure model_store is the workspace model_store for the test
    # We rely on the existing MODEL_STORE_PATH; simply run the job and assert the file is created and contains JSON
    jid = enqueue_training(tiny_job)
    path = os.path.join('model_store', f'{jid}_status.json')

    for _ in range(50):
        if os.path.exists(path) and os.path.getsize(path) > 0:
            break
        time.sleep(0.05)
    else:
        raise AssertionError('Status file not written or still zero bytes')

    with open(path, 'r') as fh:
        data = json.load(fh)
    assert data.get('job_id') == jid
    assert 'status' in data
    assert 'updated_at' in data
