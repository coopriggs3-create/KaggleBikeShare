# Bike Sharing Demand — Kaggle Project

## Overview
This project predicts hourly bike rental demand for the Kaggle *Bike Sharing* competition using weather, seasonality, and time-based features. The goal was to build and compare multiple machine-learning models to generate accurate Kaggle submissions.

## What the Code Does
- Cleans and preprocesses the data (log-transform target, fix weather category, remove unused columns)  
- Creates engineered features (hour, month, weekday, rush hour, weekend, interactions, polynomial terms)  
- Builds a full **tidymodels** workflow using recipes, workflows, tuning grids, and resampling  
- Trains a variety of models: linear regression, penalized regression, decision trees, random forests, boosted trees (LightGBM), BART, and H2O AutoML stacking  
- Generates predictions for the Kaggle test set, converts them back from log scale, clips negatives, and writes final submission files  
