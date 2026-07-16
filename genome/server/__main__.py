"""Run the genome API server: `python -m genome.server`."""

# Copyright 2026 Northtek (FrostByte Digital LLC)
# SPDX-License-Identifier: Apache-2.0

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

    host = os.environ.get("GENOME_HOST", "127.0.0.1")
    port = int(os.environ.get("GENOME_PORT", "8080"))
    api_key = os.environ.get("GENOME_API_KEY")

    # Safe by default. The API is destructive (add/update/delete/reset), so we
    # only allow binding a non-loopback interface when an API key is set;
    # otherwise refuse to start. An unauthenticated memory store on 0.0.0.0 is
    # exactly the footgun we will not ship as a default.
    loopback = {"127.0.0.1", "localhost", "::1"}
    if host not in loopback and not api_key:
        print(
            f"Refusing to start: GENOME_HOST={host!r} binds a non-loopback "
            "interface but GENOME_API_KEY is not set, which would expose an "
            "unauthenticated, destructive API. Set GENOME_API_KEY, or use "
            "GENOME_HOST=127.0.0.1 for local-only access."
        )
        return 2
    if not api_key:
        print(
            f"NOTE: GENOME_API_KEY not set; the API is unauthenticated. "
            f"Serving on {host}:{port} (loopback only)."
        )

    uvicorn.run("genome.server.app:app", host=host, port=port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
