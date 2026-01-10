from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from schemas import HeatmapResponse, ForecastResponse, RentalType
from inference_engine import inference_engine

app = FastAPI(title="Real Estate AI API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "Online", "message": "Use /api/v1/market/..."}

@app.get("/api/v1/market/heatmap", response_model=HeatmapResponse)
def get_heatmap(type: RentalType = Query(...)):
    if not inference_engine.stats_store:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train_model.py first.")
    
    return {
        "rental_type": type,
        "data": inference_engine.get_heatmap(type.value)
    }

@app.get("/api/v1/market/neighborhood/{neighborhood_id}/forecast", response_model=ForecastResponse)
def get_forecast(neighborhood_id: str, type: RentalType = Query(...)):
    result = inference_engine.get_forecast(neighborhood_id, type.value)
    if not result:
        raise HTTPException(status_code=404, detail="Neighborhood data not found")
    return result

# Endpoint to reload the model file without restarting the server
@app.post("/api/v1/system/reload")
def reload_model():
    inference_engine.load_model()
    return {"message": "Model reloaded from disk."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)