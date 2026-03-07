# Customer Churn Prediction

A machine learning web application that predicts whether a telecom customer is likely to churn, built with scikit-learn, FastAPI, and Docker.

## Overview

Given a customer's account details — such as tenure, contract type, monthly charges, and subscribed services — the app predicts the probability that the customer will cancel their subscription. The model was trained on a real-world telecom dataset of ~7,000 customers and achieves a ROC-AUC of 0.84.

## How It Works

1. A user enters customer details into a web form
2. The frontend sends the data to a FastAPI backend
3. A trained Logistic Regression pipeline preprocesses the input and returns a churn prediction with a probability score
4. The result is displayed as **Will Churn** or **Will Not Churn** alongside the confidence percentage

## Features

- Trained and compared 5 models (Logistic Regression, Random Forest, AdaBoost, Decision Tree, MLP) using 5-fold cross-validation
- Final model uses a calibrated Logistic Regression pipeline with StandardScaler and OneHotEncoder
- Engineered features including `tenure_group`, `num_services`, and `tenure_contract_ratio`
- REST API with `/api/predict`, `/api/health`, and `/api/metrics` endpoints
- Fully containerised with Docker for easy deployment

## Tech Stack

- **ML:** scikit-learn, pandas, numpy
- **API:** FastAPI, Uvicorn
- **Frontend:** HTML/CSS/JavaScript
- **Deployment:** Docker, Docker Compose

## Running the App

Docker needs to be installed. Run the command "docker compose up" in the project root. Then navigate to `http://localhost:8000` in your browser.
