#!/usr/bin/env python3
"""
FastAPI Server Runner for AI Agent Stem Diff API
"""

import logging

import uvicorn

if __name__ == "__main__":
    print("🚀 Starting AI Agent Stem Diff API Server...")
    print("📖 API Documentation will be available at: http://localhost:8000/docs")
    print("🔍 Health check: http://localhost:8000/health")

    uvicorn.run(
        "app:app",  # Import string for reload support
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
