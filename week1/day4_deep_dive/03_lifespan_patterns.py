"""
FastAPI lifespan — what it does, why it exists, three real patterns.

Run: uvicorn 03_lifespan_patterns:app --reload --port 8002
"""

import time
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException

shared_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    asynccontextmanager splits a function into two halves at 'yield':

    BEFORE yield = startup code (runs once when server starts)
        → load ML models, connect to DB, warm up caches
    
    yield
        → server is live and handling requests HERE
    
    AFTER yield = shutdown code (runs once when server stops Ctrl+C)
        → close DB connections, save state, cleanup temp files

    The 'shared_state' dict is how you share data across requests.
    It lives in memory for the entire server lifetime.
    """
    # ── STARTUP 
    print("\n[STARTUP] Server is starting...")
    print("[STARTUP] Step 1: Loading configuration")
    time.sleep(0.9)  # simulate config load

    print("[STARTUP] Step 2: Connecting to database")
    time.sleep(0.9)  # simulate DB connection

    print("[STARTUP] Step 3: Loading ML model (this is the slow part)")
    time.sleep(2.5)  # simulate model load

    shared_state["model"] = "MockMLModel(loaded=True)"
    shared_state["db_connection"] = "MockDB(connected=True)"
    shared_state["start_time"] = time.time()
    shared_state["request_count"] = 0

    print("[STARTUP] All resources ready. Server accepting requests.\n")

    yield  # ← server runs here, handling requests

    # ── SHUTDOWN 
    print("\n[SHUTDOWN] Server is shutting down...")
    print(f"[SHUTDOWN] Handled {shared_state['request_count']} requests total")
    print("[SHUTDOWN] Closing database connection")
    print("[SHUTDOWN] Releasing model from memory")
    shared_state.clear()
    print("[SHUTDOWN] Cleanup complete.")


app = FastAPI(lifespan=lifespan)


@app.get("/status")
def status():
    """
    Shows what's in shared_state — proves resources persist across requests.
    Call this multiple times and watch request_count increment.

    curl http://localhost:8002/status
    """
    shared_state["request_count"] += 1
    uptime = time.time() - shared_state["start_time"]
    return {
        "model_loaded": shared_state.get("model"),
        "db_connected": shared_state.get("db_connection"),
        "uptime_seconds": round(uptime, 2),
        "requests_served": shared_state["request_count"],
    }


@app.get("/predict")
def predict(text: str):
    """
    Simulates using the loaded model.
    Key: model is loaded ONCE at startup, reused on every request.

    curl "http://localhost:8002/predict?text=hello+world"
    """
    if "model" not in shared_state:
        raise HTTPException(status_code=503, detail="Model not loaded")

    shared_state["request_count"] += 1
    # Simulate using the model
    return {
        "input": text,
        "model_used": shared_state["model"],
        "prediction": f"processed: {text.upper()}",
        "request_number": shared_state["request_count"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# WHAT HAPPENS WITHOUT LIFESPAN (the wrong way)
# This shows you exactly why lifespan exists
# ─────────────────────────────────────────────────────────────────────────────

def fake_load_model():
    """Simulates loading a model — takes time."""
    time.sleep(3)  # 1 second load time
    return "SlowModel()"


@app.get("/predict-slow")
def predict_without_lifespan(text: str):
    """
    THE WRONG WAY: loads model on every request.

    Call this 5 times and time it:
    for i in {1..5}; do
        time curl "http://localhost:8002/predict-slow?text=test"
    done

    Compare to /predict which reuses the loaded model.
    Each call to /predict-slow takes 1+ second.
    Each call to /predict takes milliseconds.

    In production with a real LLM: 30 seconds per request vs 2 seconds.
    This is why lifespan is not optional.
    """
    model = fake_load_model()  # loads EVERY request ← never do this
    return {"model": model, "input": text, "result": "slow"}