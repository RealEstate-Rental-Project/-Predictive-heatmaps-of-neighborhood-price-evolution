# HeatMap Prediction Model

A machine learning application for real estate market analysis and prediction.

## Features

- Generate synthetic property data
- Train K-means clustering model for market trends
- API for heatmap and forecast data

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`

## Usage

1. Generate data: `python generate_data.py`
2. Train the model: `python train_model.py`
3. Start the API server: `python -m model.app.main`

## API Endpoints

- GET /api/v1/market/heatmap
- GET /api/v1/market/neighborhood/{id}/forecast