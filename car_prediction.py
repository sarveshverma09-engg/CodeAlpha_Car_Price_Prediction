# ============================================================
# CAR PRICE PREDICTION USING MACHINE LEARNING
# ============================================================

import os
import re
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    cross_val_score
)

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor


warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

FILE_PATH = "car_data.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20

CURRENT_YEAR = 2026

OUTPUT_FOLDER = "car_prediction_outputs"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("=" * 70)
print("CAR PRICE PREDICTION USING MACHINE LEARNING")
print("=" * 70)

print("\nLoading dataset...")

if not os.path.exists(FILE_PATH):

    print("\nERROR: Dataset not found!")
    print(f"Expected file: {FILE_PATH}")
    print("\nMake sure your folder looks like:")
    print("""
car_price_prediction_project/
│
├── car_prediction.py
└── car_data.csv
""")

    exit()


df = pd.read_csv(FILE_PATH)

print("\nDataset loaded successfully!")

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())


# ============================================================
# 2. DATA EXPLORATION
# ============================================================

print("\n" + "=" * 70)
print("DATA EXPLORATION")
print("=" * 70)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nDataset Information:")
df.info()

print("\nStatistical Summary:")
print(df.describe())


# ============================================================
# 3. REMOVE DUPLICATES
# ============================================================

before = len(df)

df = df.drop_duplicates()

after = len(df)

print("\nDuplicate rows removed:", before - after)

print("New dataset shape:", df.shape)


# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

print("\n" + "=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)


# ------------------------------------------------------------
# Car Age
# ------------------------------------------------------------

df["Car_Age"] = CURRENT_YEAR - df["Year"]


# ------------------------------------------------------------
# Kilometers Driven Per Year
# ------------------------------------------------------------

df["Km_Per_Year"] = (
    df["Driven_kms"] /
    (df["Car_Age"] + 1)
)


# ------------------------------------------------------------
# Brand Extraction
# ------------------------------------------------------------

def extract_brand(car_name):

    if pd.isna(car_name):
        return "Unknown"

    name = str(car_name).strip().lower()

    brand = re.split(
        r"[\s-]+",
        name
    )[0]

    return brand


df["Brand"] = df["Car_Name"].apply(
    extract_brand
)


# ------------------------------------------------------------
# Log Transformation of Driven KMs
# ------------------------------------------------------------

df["Log_Driven_kms"] = np.log1p(
    df["Driven_kms"].clip(lower=0)
)


print("\nNew features created:")

print("1. Car_Age")
print("2. Km_Per_Year")
print("3. Brand")
print("4. Log_Driven_kms")


# ============================================================
# 5. REMOVE UNNECESSARY COLUMN
# ============================================================

# Car_Name has many unique values.
# Brand is used instead.

df = df.drop(
    "Car_Name",
    axis=1
)


# ============================================================
# 6. DEFINE X AND Y
# ============================================================

X = df.drop(
    "Selling_Price",
    axis=1
)

y = df["Selling_Price"]


print("\nFeatures used for prediction:")

for column in X.columns:
    print("-", column)


print("\nTarget variable:")
print("Selling_Price")


# ============================================================
# 7. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=TEST_SIZE,

    random_state=RANDOM_STATE
)


print("\n" + "=" * 70)
print("TRAIN TEST SPLIT")
print("=" * 70)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 8. IDENTIFY FEATURES
# ============================================================

numeric_features = X.select_dtypes(
    include=[
        "int64",
        "float64",
        "int32",
        "float32"
    ]
).columns.tolist()


categorical_features = X.select_dtypes(
    include=[
        "object",
        "category"
    ]
).columns.tolist()


print("\nNumeric Features:")
print(numeric_features)

print("\nCategorical Features:")
print(categorical_features)


# ============================================================
# 9. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[

        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),

        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


preprocessor = ColumnTransformer(

    transformers=[

        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),

        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# 10. CREATE MODELS
# ============================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=500,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=3,
            random_state=RANDOM_STATE
        ),

    "XGBoost":
        XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
}


# ============================================================
# 11. TRAIN BASE MODELS
# ============================================================

print("\n" + "=" * 70)
print("TRAINING MODELS")
print("=" * 70)


results = []

trained_models = {}

predictions = {}


for model_name, model in models.items():

    print(
        f"\nTraining {model_name}..."
    )


    pipeline = Pipeline(
        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                model
            )
        ]
    )


    pipeline.fit(
        X_train,
        y_train
    )


    y_pred = pipeline.predict(
        X_test
    )


    mae = mean_absolute_error(
        y_test,
        y_pred
    )


    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )


    r2 = r2_score(
        y_test,
        y_pred
    )


    results.append({

        "Model": model_name,

        "MAE": mae,

        "RMSE": rmse,

        "R2 Score": r2
    })


    trained_models[
        model_name
    ] = pipeline


    predictions[
        model_name
    ] = y_pred


# ============================================================
# 12. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    by="R2 Score",
    ascending=False
)


print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.round(4).to_string(
        index=False
    )
)


# ============================================================
# 13. HYPERPARAMETER TUNING - RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("TUNING RANDOM FOREST")
print("=" * 70)


rf_pipeline = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            RandomForestRegressor(
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]
)


rf_parameters = {

    "model__n_estimators": [
        300,
        500,
        800
    ],

    "model__max_depth": [
        None,
        5,
        8,
        12,
        15
    ],

    "model__min_samples_split": [
        2,
        3,
        5
    ],

    "model__min_samples_leaf": [
        1,
        2
    ],

    "model__max_features": [
        0.7,
        1.0,
        "sqrt"
    ]
}


rf_search = RandomizedSearchCV(

    estimator=rf_pipeline,

    param_distributions=rf_parameters,

    n_iter=25,

    scoring="r2",

    cv=5,

    random_state=RANDOM_STATE,

    n_jobs=-1
)


rf_search.fit(
    X_train,
    y_train
)


best_rf = rf_search.best_estimator_


print("\nBest Random Forest Parameters:")

print(
    rf_search.best_params_
)


rf_pred = best_rf.predict(
    X_test
)


rf_mae = mean_absolute_error(
    y_test,
    rf_pred
)


rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_pred
    )
)


rf_r2 = r2_score(
    y_test,
    rf_pred
)


print("\nTuned Random Forest Results:")

print(
    "MAE:",
    round(rf_mae, 4)
)

print(
    "RMSE:",
    round(rf_rmse, 4)
)

print(
    "R2 Score:",
    round(rf_r2, 4)
)


# ============================================================
# 14. HYPERPARAMETER TUNING - XGBOOST
# ============================================================

print("\n" + "=" * 70)
print("TUNING XGBOOST")
print("=" * 70)


xgb_pipeline = Pipeline(

    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "model",
            XGBRegressor(
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]
)


xgb_parameters = {

    "model__n_estimators": [
        200,
        300,
        500,
        800
    ],

    "model__learning_rate": [
        0.01,
        0.03,
        0.05,
        0.1
    ],

    "model__max_depth": [
        2,
        3,
        4,
        5,
        6
    ],

    "model__subsample": [
        0.7,
        0.8,
        0.9,
        1.0
    ],

    "model__colsample_bytree": [
        0.7,
        0.8,
        0.9,
        1.0
    ]
}


xgb_search = RandomizedSearchCV(

    estimator=xgb_pipeline,

    param_distributions=xgb_parameters,

    n_iter=30,

    scoring="r2",

    cv=5,

    random_state=RANDOM_STATE,

    n_jobs=-1
)


xgb_search.fit(
    X_train,
    y_train
)


best_xgb = xgb_search.best_estimator_


print("\nBest XGBoost Parameters:")

print(
    xgb_search.best_params_
)


xgb_pred = best_xgb.predict(
    X_test
)


xgb_mae = mean_absolute_error(
    y_test,
    xgb_pred
)


xgb_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        xgb_pred
    )
)


xgb_r2 = r2_score(
    y_test,
    xgb_pred
)


print("\nTuned XGBoost Results:")

print(
    "MAE:",
    round(xgb_mae, 4)
)

print(
    "RMSE:",
    round(xgb_rmse, 4)
)

print(
    "R2 Score:",
    round(xgb_r2, 4)
)


# ============================================================
# 15. ADD TUNED MODELS TO RESULTS
# ============================================================

results_df = pd.concat(

    [
        results_df,

        pd.DataFrame([{

            "Model":
                "Tuned Random Forest",

            "MAE":
                rf_mae,

            "RMSE":
                rf_rmse,

            "R2 Score":
                rf_r2

        }]),

        pd.DataFrame([{

            "Model":
                "Tuned XGBoost",

            "MAE":
                xgb_mae,

            "RMSE":
                xgb_rmse,

            "R2 Score":
                xgb_r2

        }])
    ],

    ignore_index=True
)


results_df = results_df.sort_values(
    by="R2 Score",
    ascending=False
)


# ============================================================
# 16. FINAL MODEL SELECTION
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL SELECTION")
print("=" * 70)


best_model_name = (
    results_df.iloc[0]["Model"]
)


if best_model_name == "Tuned Random Forest":

    final_model = best_rf

    final_prediction = rf_pred


elif best_model_name == "Tuned XGBoost":

    final_model = best_xgb

    final_prediction = xgb_pred


else:

    final_model = trained_models[
        best_model_name
    ]

    final_prediction = predictions[
        best_model_name
    ]


final_r2 = r2_score(
    y_test,
    final_prediction
)


final_mae = mean_absolute_error(
    y_test,
    final_prediction
)


final_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        final_prediction
    )
)


print(
    "\nFINAL MODEL:",
    best_model_name
)


print(
    "R2 Score:",
    round(final_r2, 4)
)


print(
    "R2 Percentage:",
    round(final_r2 * 100, 2),
    "%"
)


print(
    "MAE:",
    round(final_mae, 4)
)


print(
    "RMSE:",
    round(final_rmse, 4)
)


# ============================================================
# 17. 5-FOLD CROSS VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("5-FOLD CROSS VALIDATION")
print("=" * 70)


cv_scores = cross_val_score(

    final_model,

    X,

    y,

    cv=5,

    scoring="r2"

)


print("\nCV Scores:")

print(
    np.round(
        cv_scores,
        4
    )
)


print(
    "\nAverage CV R2:",
    round(
        cv_scores.mean(),
        4
    )
)


print(
    "Average CV R2 Percentage:",
    round(
        cv_scores.mean() * 100,
        2
    ),
    "%"
)


# ============================================================
# 18. SAVE FINAL MODEL
# ============================================================

model_path = os.path.join(

    OUTPUT_FOLDER,

    "best_car_price_model.pkl"
)


joblib.dump(
    final_model,
    model_path
)


print("\nFinal model saved:")
print(model_path)


# ============================================================
# 19. SAVE MODEL COMPARISON
# ============================================================

results_path = os.path.join(

    OUTPUT_FOLDER,

    "model_comparison.csv"
)


results_df.to_csv(
    results_path,
    index=False
)


print(
    "\nModel comparison saved:"
)

print(
    results_path
)


# ============================================================
# 20. ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(
    figsize=(8, 6)
)


plt.scatter(

    y_test,

    final_prediction,

    alpha=0.75
)


minimum = min(

    y_test.min(),

    final_prediction.min()
)


maximum = max(

    y_test.max(),

    final_prediction.max()
)


plt.plot(

    [minimum, maximum],

    [minimum, maximum],

    linestyle="--"
)


plt.xlabel(
    "Actual Selling Price"
)


plt.ylabel(
    "Predicted Selling Price"
)


plt.title(
    f"Actual vs Predicted - {best_model_name}"
)


plt.tight_layout()


prediction_plot = os.path.join(

    OUTPUT_FOLDER,

    "actual_vs_predicted.png"
)


plt.savefig(

    prediction_plot,

    dpi=300
)


plt.show()


# ============================================================
# 21. FEATURE IMPORTANCE
# ============================================================

try:

    model_inside_pipeline = (
        final_model.named_steps["model"]
    )


    if hasattr(
        model_inside_pipeline,
        "feature_importances_"
    ):

        processor = (
            final_model
            .named_steps["preprocessor"]
        )


        feature_names = (
            processor
            .get_feature_names_out()
        )


        importances = (
            model_inside_pipeline
            .feature_importances_
        )


        importance_df = pd.DataFrame({

            "Feature":
                feature_names,

            "Importance":
                importances

        })


        importance_df = (
            importance_df
            .sort_values(
                by="Importance",
                ascending=False
            )
            .head(15)
        )


        print(
            "\nTop Important Features:"
        )

        print(
            importance_df
        )


        plt.figure(
            figsize=(10, 7)
        )


        plt.barh(

            importance_df[
                "Feature"
            ][::-1],

            importance_df[
                "Importance"
            ][::-1]
        )


        plt.xlabel(
            "Importance"
        )


        plt.ylabel(
            "Feature"
        )


        plt.title(
            "Top 15 Feature Importances"
        )


        plt.tight_layout()


        importance_path = os.path.join(

            OUTPUT_FOLDER,

            "feature_importance.png"
        )


        plt.savefig(

            importance_path,

            dpi=300
        )


        plt.show()


except Exception as error:

    print(
        "\nFeature importance could not be generated."
    )

    print(error)


# ============================================================
# 22. USER INPUT PREDICTION
# ============================================================

print("\n" + "=" * 70)
print("CAR PRICE PREDICTION")
print("=" * 70)


print(
    "\nEnter details of a car "
    "to predict its selling price."
)


try:

    car_name = input(
        "\nCar Name: "
    ).strip()


    year = int(
        input(
            "Manufacturing Year: "
        )
    )


    present_price = float(
        input(
            "Present Price (in lakhs): "
        )
    )


    driven_kms = int(
        input(
            "Driven Kilometres: "
        )
    )


    fuel_type = input(
        "Fuel Type (Petrol/Diesel/CNG): "
    ).strip()


    selling_type = input(
        "Selling Type (Dealer/Individual): "
    ).strip()


    transmission = input(
        "Transmission (Manual/Automatic): "
    ).strip()


    owner = int(
        input(
            "Previous Owners (0/1/2/3): "
        )
    )


    # Feature engineering
    car_age = max(
        0,
        CURRENT_YEAR - year
    )


    km_per_year = (
        driven_kms /
        (car_age + 1)
    )


    brand = extract_brand(
        car_name
    )


    log_driven_kms = np.log1p(
        driven_kms
    )


    # Create input dataframe
    new_car = pd.DataFrame({

        "Year": [year],

        "Present_Price": [
            present_price
        ],

        "Driven_kms": [
            driven_kms
        ],

        "Fuel_Type": [
            fuel_type
        ],

        "Selling_type": [
            selling_type
        ],

        "Transmission": [
            transmission
        ],

        "Owner": [
            owner
        ],

        "Car_Age": [
            car_age
        ],

        "Km_Per_Year": [
            km_per_year
        ],

        "Brand": [
            brand
        ],

        "Log_Driven_kms": [
            log_driven_kms
        ]
    })


    # Final prediction
    predicted_price = (
        final_model.predict(
            new_car
        )[0]
    )


    print("\n" + "=" * 70)

    print(
        "FINAL PREDICTION"
    )

    print("=" * 70)


    print(
        "\nPredicted Selling Price:"
    )


    print(
        f"₹ {predicted_price:.2f} Lakh"
    )


    print(
        "\nModel Used:"
    )


    print(
        best_model_name
    )


    print(
        "\nTest R2 Score:"
    )


    print(
        f"{final_r2 * 100:.2f}%"
    )


    print("=" * 70)


except Exception as error:

    print(
        "\nPrediction error:"
    )

    print(error)


# ============================================================
# PROJECT COMPLETED
# ============================================================

print("\n" + "=" * 70)

print(
    "CAR PRICE PREDICTION PROJECT COMPLETED!"
)

print("=" * 70)