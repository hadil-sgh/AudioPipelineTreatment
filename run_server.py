import uvicorn
from fastapi import FastAPI
from api.routes import router

app = FastAPI(title="Real-Time Audio Processing API")

# Include the router
app.include_router(router, prefix="/api")

if __name__ == "__main__":
    uvicorn.run(
        "run_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 