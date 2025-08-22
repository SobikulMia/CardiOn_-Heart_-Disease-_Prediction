# model_training.py
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import joblib
from sklearn.model_selection import learning_curve, train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def load_and_preprocess_data(filepath="cardio_train.csv"):
    load_data = pd.read_csv(filepath, sep=";").sample(70000)
    
        # Valid BP Filter
    def valid_bp(data):
        return 50 <= data["ap_hi"] <= 250 and 30 <= data["ap_lo"] <= 200
    load_data = load_data[load_data.apply(valid_bp, axis=1)].reset_index(drop=True)

    # Age in Years
    load_data["age_year"] = (load_data["age"] // 365).astype(int)
    load_data.drop("age", axis=1, inplace=True)

    # BMI Calculation
    load_data["hight_m"] = load_data["height"] / 100
    load_data["BMI"] = load_data["weight"] / (load_data["hight_m"] ** 2)

    # Basic Features
    load_data['is_hypertensive'] = ((load_data['ap_hi'] >= 140) | (load_data['ap_lo'] >= 90)).astype(int)
    load_data["cholesterol_gluc_sum"] = load_data["cholesterol"] + load_data["gluc"]
    load_data["age_bmi_ratio"] = load_data["age_year"] / load_data["BMI"]
    load_data["is_obese"] = (load_data["BMI"] > 30).astype(int)
    load_data["pulse_pressure"] = load_data["ap_hi"] - load_data["ap_lo"]
    load_data["age_bmi_product"] = load_data["age_year"] * load_data["BMI"]
    load_data['lifestyle_score'] = load_data['smoke'] + load_data['alco'] + (1 - load_data['active'])

    # Additional Features
    load_data['weight_height_ratio'] = load_data['weight'] / load_data['height']
    load_data['map'] = (2 * load_data['ap_lo'] + load_data['ap_hi']) / 3
    load_data['smoke_alco_combo'] = (load_data['smoke'] & load_data['alco']).astype(int)
    load_data['pulse_pressure_ratio'] = load_data['pulse_pressure'] / load_data['ap_hi']
    load_data['BMI_squared'] = load_data['BMI'] ** 2
    load_data['is_senior'] = (load_data['age_year'] >= 60).astype(int)
    load_data['chol_gluc_diff'] = abs(load_data['cholesterol'] - load_data['gluc'])
    load_data['activity_score'] = (load_data['active'] * 2) - (load_data['smoke'] + load_data['alco'])

    # New Features added as requested:

    # BP Difference Ratio
    load_data['bp_diff_ratio'] = load_data['pulse_pressure'] / load_data['ap_hi']

    # Health Risk Score (Composite)
    load_data['health_risk_score'] = load_data['is_hypertensive'] + load_data['is_obese'] + load_data['lifestyle_score']

    # Systolic to Diastolic Ratio (BP Ratio)
    load_data['bp_ratio'] = load_data['ap_hi'] / load_data['ap_lo']

    # Cholesterol & Glucose Risk Category (Combined Category)
    def chol_gluc_risk(chol, gluc):
        if chol > 1 and gluc > 1:
            return 'HighRisk'
        elif chol > 1 or gluc > 1:
            return 'MediumRisk'
        else:
            return 'LowRisk'
    load_data['chol_gluc_risk'] = load_data.apply(lambda row: chol_gluc_risk(row['cholesterol'], row['gluc']), axis=1)

    # Age & Activity Interaction
    load_data['senior_inactive'] = ((load_data['age_year'] >= 60) & (load_data['active'] == 0)).astype(int)

    # Adjusted BMI (Gender-based)
    def adjusted_bmi(row):
        if row['gender'] == 1:
            return row['BMI'] * 1.1
        else:
            return row['BMI']
    load_data['adjusted_bmi'] = load_data.apply(adjusted_bmi, axis=1)

    #MAP Deviation from Normal
    NORMAL_MAP = 93
    load_data['map_deviation'] = abs(load_data['map'] - NORMAL_MAP)

    # Age Group
    def age_group(age):
        if age < 40:
            return 'young'
        elif age < 60:
            return 'middle-aged'
        else:
            return 'senior'
    load_data['age_group'] = load_data['age_year'].apply(age_group)

    # BMI Category
    def bmi_category(bmi):
        if bmi < 18.5:
            return 'Underweight'
        elif 18.5 <= bmi < 25:
            return 'Normal'
        elif 25 <= bmi < 30:
            return 'Overweight'
        else:
            return 'Obese'
    load_data['bmi_category'] = load_data['BMI'].apply(bmi_category)

    # Gender Map
    load_data["gender"] = load_data["gender"].map({1: 0, 2: 1})

    # BP Category
    def bp_category(systolic, diastolic):
        if systolic >= 140 or diastolic >= 90:
            return "High"
        elif systolic < 90 or diastolic > 60:
            return "low"
        else:
            return "Normal"
    load_data["bp_category"] = load_data.apply(lambda row: bp_category(row["ap_hi"], row["ap_lo"]), axis=1)

    # One-Hot Encoding
    categorical_cols = ['bp_category', 'age_group', 'bmi_category', 'chol_gluc_risk']
    load_data = pd.get_dummies(load_data, columns=categorical_cols, drop_first=True)

    return load_data  

def generate_polynomial_features(load_data):
    selected_features = [
        'age_year', 'BMI', 'ap_hi', 'ap_lo', 'pulse_pressure',
        'cholesterol', 'gluc', 'activity_score', 'map', 'bp_diff_ratio',
        'health_risk_score', 'bp_ratio', 'map_deviation', 'adjusted_bmi',
        'senior_inactive'
    ]
    
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
    poly_features = poly.fit_transform(load_data[selected_features])
    poly_feature_names = poly.get_feature_names_out(selected_features)
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=load_data.index)
    
    return pd.concat([load_data, poly_df], axis=1)

def split_and_scale(data):
    X = data.drop(["cardio","id"], axis=1)
    y = data["cardio"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_res)
    X_test_sc = scaler.transform(X_test)
    
    return X_train_sc, X_test_sc, y_train_res, y_test, scaler

def train_best_model(X_train, y_train, X_test, y_test):
    # Simplified model training for demonstration
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model

def save_model_artifacts(model, scaler, feature_names,X_test,y_test):
    joblib.dump(X_test, "X_test.pkl")
    joblib.dump(y_test, "y_test.pkl")
    joblib.dump(model, 'heart_disease_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(feature_names, 'feature_names.pkl')
    print("Model artifacts saved successfully")

def main():
    print("Loading and preprocessing data...")
    data = load_and_preprocess_data()
    data = generate_polynomial_features(data)
    
    print("Splitting and scaling data...")
    X_train_sc, X_test_sc, y_train_res, y_test, scaler = split_and_scale(data)
    
    print("Training model...")
    model = train_best_model(X_train_sc, y_train_res, X_test_sc, y_test)
    
    print("Saving model artifacts...")
    feature_names = list(data.drop(["cardio","id"], axis=1).columns)
    save_model_artifacts(model, scaler, feature_names,X_test_sc,y_test)

if __name__ == "__main__":
    main()