"""Smoke tests — make sure the package imports and the FastAPI app builds."""
from __future__ import annotations


def test_settings_load() -> None:
    from app.config import settings

    assert settings.embedding_model
    assert settings.db_path
    assert settings.sources_dir


def test_app_factory_builds() -> None:
    from app.main import create_app

    app = create_app()
    paths = {route.path for route in app.routes}
    # A handful of representative endpoints must be registered.
    assert "/" in paths
    assert "/health" in paths
    assert "/upload" in paths
    assert "/query/structured" in paths
    assert "/query/analytical" in paths
    assert "/catalog" in paths


def test_bundled_assets_present() -> None:
    from pathlib import Path

    pkg_root = Path(__file__).resolve().parent.parent / "app"
    assert (pkg_root / "db" / "schema.sql").is_file()
    assert (pkg_root / "web" / "index.html").is_file()
    assert (pkg_root / "data" / "benchmark_component_costs.csv").is_file()
