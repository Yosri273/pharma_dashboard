# Release Notes — Pharma Dashboard Fixes

Date: 2025-09-14

Changes:
- Fixed UI recommendation rendering ordering and robustness (critical → warning → info) in `app/analysis/ui_helpers.py`.
- Implemented missing ETL helpers used by tests in `etl/transforms.py`:
  - get_kpis, get_top_products, get_sales_by_region, load_comprehensive_sample_data, get_comprehensive_kpis.
- Added `tests/conftest.py` to ensure repo root is importable during tests.
- Replaced deprecated `datetime.utcnow()` usages with timezone-aware `datetime.now(datetime.UTC)` across the codebase.
- Fixed indentation and stability in `alerting/tasks.py` state transition block.
- Added CI workflow `.github/workflows/tests.yml` to run pytest on pushes and PRs.

Verification:
- Local test suite: 25 passed, 1 warning.
- No API changes to public functions beyond adding new helpers.

Rollback Plan:
- Revert this commit hash to restore previous behavior if any regression is found.
- Disable new CI test workflow by removing `.github/workflows/tests.yml` if needed.
- For datetime changes, swapping back to `datetime.utcnow()` is trivial but not recommended.
