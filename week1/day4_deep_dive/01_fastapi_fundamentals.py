import time
from enum import Enum
from typing import Optional
from fastapi import FastAPI, HTTPException, Query, Path, Depends, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

app = FastAPI(
    title="FastAPI Deep Dive",
    description="Every concept you need to understand.",
    version="1.0.0",
)


# CONCEPT 1: Basic routing
# A route = URL path + HTTP method + Python function
# FastAPI calls the function when that URL + method is hit

@app.get("/")
def root():
    """Simplest possible route. Returns a dict → FastAPI converts to JSON."""
    return {"message": "FastAPI is running"}

@app.get("/hello/{name}")
def hello(name: str):
    """
    Path parameter: {name} in URL is extracted and passed to function.
    curl http://localhost:8001/hello/Arjun
    → {"greeting": "Hello, Arjun"}
    """
    return {"greeting": f"Hello, {name}"}


# CONCEPT 2: Query parameters vs Path parameters
# Path param:  /items/42        → part of the URL structure
# Query param: /items?limit=10  → after the ? in the URL

@app.get("/items/{item_id}")
def get_item(
    item_id: int = Path(..., ge=1, description="Item ID, must be >= 1"),
    include_details: bool = Query(default=False, description="Include extra details"),
    category: Optional[str] = Query(default=None),
):
    """
    item_id is a path parameter — required, must be int >= 1
    include_details is a query parameter — optional, default False
    category is a query parameter — optional, default None

    curl http://localhost:8001/items/5
    curl http://localhost:8001/items/5?include_details=true&category=tech
    curl http://localhost:8001/items/0  ← will fail validation (ge=1)
    """
    result = {"item_id": item_id, "category": category}
    if include_details:
        result["details"] = "Extra information here"
    return result


# CONCEPT 3: Request body with Pydantic
# POST/PUT requests send data in the body (not URL)
# Pydantic validates it before your function even runs

class Priority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"

class CreateTaskRequest(BaseModel):
    title: str = Field(..., min_length=3, max_length=100)
    description: Optional[str] = Field(default=None, max_length=500)
    priority: Priority = Field(default=Priority.medium)
    tags: list[str] = Field(default_factory=list, max_length=5)

    @field_validator("title")
    @classmethod
    def tile_must_not_be_blank(cls, v:str) -> str:
        if v.strip() == "":
            raise ValueError("Title cannot be blank or whitespace only")
        return v.strip()

class TaskResponse(BaseModel):
    id: int
    title: str
    priority: Priority
    tags: list[str]
    created_at: float


@app.post("/tasks", response_model=TaskResponse, status_code=201)
def create_task(request: CreateTaskRequest):
    """
    FastAPI reads the request body, validates with CreateTaskRequest,
    then serializes the return value using TaskResponse.

    If validation fails → FastAPI returns 422 automatically (you write no error code)

    Test valid request:
    curl -X POST http://localhost:8001/tasks \
      -H "Content-Type: application/json" \
      -d '{"title": "Learn FastAPI", "priority": "high", "tags": ["python", "api"]}'

    Test invalid (title too short):
    curl -X POST http://localhost:8001/tasks \
      -H "Content-Type: application/json" \
      -d '{"title": "Hi"}'

    Test invalid (wrong priority enum):
    curl -X POST http://localhost:8001/tasks \
      -H "Content-Type: application/json" \
      -d '{"title": "Valid title", "priority": "urgent"}'
    """

    return TaskResponse(
        id = 42,
        title=request.title,
        priority=request.priority,
        tags=request.tags,
        created_at=time.time(),
    )


# CONCEPT 4: HTTP errors — when and how to raise them
#
# 400 Bad Request:   client sent invalid data (logic error, not schema error)
# 401 Unauthorized:  not logged in
# 403 Forbidden:     logged in but no permission
# 404 Not Found:     resource doesn't exist
# 422 Unprocessable: Pydantic validation failed (FastAPI raises this automatically)
# 500 Server Error:  your code crashed

# Fake database for demo
FAKE_DB = {
    1: {"name": "Alice", "role": "admin"},
    2: {"name": "Bob", "role": "user"},
}


@app.get("/user/{user_id}")
def get_user(user_id: int):
    """
    curl http://localhost:8001/users/1    ← found
    curl http://localhost:8001/users/99   ← 404
    curl http://localhost:8001/users/abc  ← 422 (not int)
    """
    if user_id not in FAKE_DB:
        raise HTTPException(
            status_code=404,
            detail=f"User {user_id} not found",
        )
    return FAKE_DB[user_id]

@app.delete("/users/{user_id}")
def delete_user(user_id: int, requester_role: str = Query(...)):
    """
    Only admins can delete. Business logic error → 403.

    curl -X DELETE "http://localhost:8001/users/2?requester_role=user"  ← 403
    curl -X DELETE "http://localhost:8001/users/2?requester_role=admin" ← 200
    """

    if requester_role != "admin":
        raise HTTPException(
            status_code=403,
            detail="Only admins can delete users",
        )
    
    if user_id not in FAKE_DB:
        raise HTTPException(status_code=404, detail="User not found")
    
    del FAKE_DB[user_id]
    return {"deleted": user_id}


# CONCEPT 5: Dependency injection
# Shared logic that multiple endpoints need → put in a dependency
# FastAPI calls it automatically before your function runs
#
# Real world uses: auth token validation, DB connection, rate limiting

def get_api_key(request: Request) -> str:
    """
    Dependency: extract and validate API key from header.
    Any endpoint that needs auth adds this as a parameter.
    FastAPI calls it automatically.
    """
    api_key = request.headers.get("X_API_Key")
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="X-API_key header is required",
        )
    if api_key != "secret-key-123":
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
        )
    return api_key

@app.get("/protected/data")
def protected_endpoint(api_key: str = Depends(get_api_key)):
    """
    This endpoint is protected. FastAPI runs get_api_key() first.
    If it raises HTTPException, your function never runs.

    curl http://localhost:8001/protected/data
    ← 401 (no API key)

    curl http://localhost:8001/protected/data -H "X-API-Key: wrong"
    ← 401 (invalid)

    curl http://localhost:8001/protected/data -H "X-API-Key: secret-key-123"
    ← 200 (success)
    """
    return {"data": "This is protected information", "authenticated_with": api_key}


# CONCEPT 6: Middleware
# Code that runs on EVERY request, before and after your endpoint function
# Use for: logging, timing, CORS, auth headers

# CORS middleware — allows browsers to call your API from different domains
# Without this, browser-based frontends get blocked by security policy
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # in production: list specific domains
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    """
    Custom middleware: measures how long every request takes.
    Adds X-Response-Time header to every response.

    Flow:
    1. Request arrives
    2. This middleware runs (before)
    3. call_next passes to your endpoint
    4. Endpoint runs
    5. This middleware continues (after)
    6. Response sent to client
    """
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
    return response


# CONCEPT 7: Global exception handler
# Catch unexpected errors — return clean JSON instead of a crash traceback

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    If any endpoint crashes with an unhandled exception,
    return a clean 500 JSON response instead of exposing internal details.

    In production you'd also log the full traceback here.
    """
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "path": str(request.url),
        },
    )

@app.get("/crash")
def crash_endpoint():
    """
    Intentionally crashes to demonstrate global exception handler.
    curl http://localhost:8001/crash
    """
    raise RuntimeError("Something went wrong internally")

