# 🚗 Car Price Prediction using Machine Learning

A Machine Learning regression project that predicts the **selling price of a used car** based on factors such as manufacturing year, present price, kilometers driven, fuel type, selling type, transmission, and previous owners.

The project implements data preprocessing, feature engineering, multiple regression algorithms, hyperparameter tuning, model evaluation, cross-validation, visualization, and an interactive car price prediction system.

---

## 📌 Project Overview

Used-car prices depend on several factors including the vehicle's age, current market value, kilometers driven, fuel type, transmission, and ownership history.

The objective of this project is to build a Machine Learning model that learns patterns from historical car data and predicts the expected selling price of a car.

### Problem Type

**Supervised Machine Learning → Regression**

### Target Variable

`Selling_Price`

---

## 🎯 Objectives

- Analyze and understand the car dataset
- Perform data cleaning and preprocessing
- Handle numerical and categorical features
- Perform feature engineering
- Train multiple regression models
- Compare model performance
- Perform hyperparameter tuning
- Apply 5-Fold Cross-Validation
- Select the best-performing model
- Save the trained model
- Predict the selling price for new car details

---

## 📊 Dataset

The project uses a used-car dataset containing information about cars and their selling prices.

### Main Features

| Feature | Description |
|---|---|
| `Car_Name` | Name of the car |
| `Year` | Manufacturing year |
| `Present_Price` | Current/present price of the car |
| `Driven_kms` | Kilometers driven |
| `Fuel_Type` | Fuel type such as Petrol, Diesel or CNG |
| `Selling_type` | Dealer or Individual |
| `Transmission` | Manual or Automatic |
| `Owner` | Number of previous owners |
| `Selling_Price` | Target variable |

---

## ⚙️ Feature Engineering

Additional features are created to improve the model's ability to learn useful patterns.

### 1. Car Age

```python
Car_Age = 2026 - Year