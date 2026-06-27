#!/usr/bin/env python3
"""
ALIMA Webapp Launcher
Claude Generated - Start the FastAPI web server
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    import uvicorn
    from src.webapp.app import app

    print("\n" + "=" * 60)
    print("🚀 ALIMA Webapp Server")
    print("=" * 60)
    print("\n📝 Starting server on http://localhost:8000")
    print("📖 API docs: http://localhost:8000/docs")
    print("\n✅ Press Ctrl+C to stop the server\n")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level=os.environ.get("LOG_LEVEL", "info").lower(),
        access_log=False,
        # Honor X-Forwarded-Proto/For from the Apache reverse proxy so the app sees the
        # real client + the https/wss scheme behind TLS termination. forwarded_allow_ips
        # defaults to the proxy host; override via env when the proxy is a separate machine.
        # - Claude Generated
        proxy_headers=True,
        forwarded_allow_ips=os.environ.get("FORWARDED_ALLOW_IPS", "127.0.0.1"),
    )
