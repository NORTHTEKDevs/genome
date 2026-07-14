"""Run the genome API server: `python -m genome.server`."""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: BUSL-1.1

from __future__ import annotations


def main() -> int:
    try:
        import uvicorn
    except ImportError:
        print(
            "uvicorn not installed. Install with: pip install \"genome[fastapi]\""
        )
        return 1
    import os
    host = os.environ.get("GENOME_HOST", "0.0.0.0")
    port = int(os.environ.get("GENOME_PORT", "8080"))
    uvicorn.run("genome.server.app:app", host=host, port=port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
