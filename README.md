# Iris Species Predictor API

This is a simple FastAPI application that serves a machine learning model to predict the species of an Iris flower based on its measurements.

## What it predicts
The model takes four physical measurements of an Iris flower (sepal and petal dimensions) and classifies it into one of three species: `setosa`, `versicolor`, or `virginica`.

## Example Request Body for `/predict`
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}