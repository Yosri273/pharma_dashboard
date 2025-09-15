import json
from app.callbacks import thresholds as th


def test_merge_thresholds_from_inputs_tmp(tmp_path, monkeypatch):
    # base JSON text with one existing kpi
    base_json = json.dumps({"existing_info": 1})
    table = [
        {"kpi": "new_kpi", "info": 10, "warning": 5, "critical": 1},
        {"kpi": "existing", "warning": 2},
    ]
    merged = th.merge_thresholds_from_inputs(base_json, table, 0.04, 0.02, 0.25, 0.1, 30.0, 20.0, 0.08)
    # friendly fields present
    assert merged["conversion_rate_warning"] == 0.04
    assert merged["aov_info"] == 20.0
    # table merged
    assert merged["new_kpi_info"] == 10
    assert merged["new_kpi_warning"] == 5
    assert merged["new_kpi_critical"] == 1
    # existing merged/won't be removed
    assert merged["existing_info"] == 1
    assert merged["existing_warning"] == 2


def test_validate_threshold_inputs():
    ok, msg = th.validate_threshold_inputs(
        table_data=[{"kpi": "x", "info": "not-a-number"}],
        conv_warn=None, conv_crit=None, gm_warn=None, gm_crit=None, aov_warn=None, aov_info=None, ret_warn=None
    )
    assert not ok
    assert "must be numeric" in msg
    assert "must be numeric" in msg
