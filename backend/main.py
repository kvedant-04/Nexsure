"""
Root entry point for Uvicorn.
This file allows you to run: python -m uvicorn main:app
Instead of: python -m uvicorn app.main:app
"""

from app.main import app

__all__ = ["app"]
