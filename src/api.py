"""REST API wrapper around RecommenderService — this is the piece that
turns "Python class with a .recommend() method" into something a
Kubernetes/AKS deployment can actually expose on a port. The Azure ML
Managed Online Endpoint path does NOT use this file — it gets a
simpler init()/run() scoring script instead, since Azure ML provides
the HTTP layer for you. Here, on raw Kubernetes, we have to provide it
ourselves, which is exactly what FastAPI + uvicorn are doing below.

Run locally with:
    uvicorn src.api:app --host 0.0.0.0 --port 8080
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.serve import RecommenderService

service: RecommenderService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # FastAPI's lifespan hook: code before yield runs at startup, code after yield
    # runs at shutdown. Loading the model here means it happens once per process,
    # not once per request.
    # global is needed because `service` is declared at module scope — without it,
    # Python would treat this as a local variable and the route handlers couldn't see it.
    global service
    service = RecommenderService()
    yield


app = FastAPI(title="H&M Two-Tower Recommender", lifespan=lifespan)


class RecommendRequest(BaseModel):
    customer_id: str
    k: int = 12


class BatchRecommendRequest(BaseModel):
    customer_ids: list[str]
    k: int = 12


class RecommendResponse(BaseModel):
    customer_id: str
    recommendations: list[str]
    image_urls: list[str] = []
    source: str
    reason: str | None = None


@app.get("/health")
def health():
    """Kubernetes liveness/readiness probes hit this. Returns 200 only
    once the model + catalog are actually loaded, not just because the
    process is running."""
    if service is None:
        raise HTTPException(status_code=503, detail="Service is still starting up")
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    return service.recommend(req.customer_id, k=req.k)


@app.post("/recommend/batch", response_model=list[RecommendResponse])
def recommend_batch(req: BatchRecommendRequest):
    return service.recommend_batch(req.customer_ids, k=req.k)
