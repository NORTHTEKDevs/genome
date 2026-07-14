"""REST API server for genome.

Install with: pip install "genome[fastapi]"
Run with:     uvicorn genome.server.app:app --reload
Or:           python -m genome.server
"""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: BUSL-1.1

from genome.server.app import create_app

__all__ = ["create_app"]
