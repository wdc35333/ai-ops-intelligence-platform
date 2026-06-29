"""Runtime configuration.

All settings are read from environment variables prefixed with ``MLA_``
(e.g. ``MLA_DATA_SOURCE=db``). A local ``.env`` file is honoured for dev.

The service never writes to the kiosk database; the optional ``MLA_DATABASE_URL``
should point at a **read-only** MySQL/MariaDB user.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MLA_", env_file=".env", extra="ignore")

    # "synthetic" generates realistic series (offline/CI/portfolio demo).
    # "db" reads the live kiosk DB read-only.
    data_source: Literal["synthetic", "db"] = "synthetic"

    # SQLAlchemy URL, e.g. mysql+pymysql://ro_user:pw@host:3306/kioskdb
    database_url: Optional[str] = None

    # Default look-back window for detection (hours). 168h = 7 days.
    default_window_hours: int = 168

    # IsolationForest expected outlier fraction for the host-metrics detector.
    if_contamination: float = 0.02

    # Robust z-score threshold for the seasonal bookings detector.
    booking_z_thresh: float = 3.5

    # Deterministic seed for the synthetic source.
    synthetic_seed: int = 42


@lru_cache
def get_settings() -> Settings:
    return Settings()
