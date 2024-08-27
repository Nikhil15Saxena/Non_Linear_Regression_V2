#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from factor_analyzer import FactorAnalyzer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from sklearn.tree import export_graphviz
import pydotplus
from io import StringIO
import xgboost as xgb

# CSS to inject contained in a string
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stButton>button {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 24px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 16px;
            }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def main():
    st.title("Non-Linear Classification Analysis")

    # Enhanced About section
    st.sidebar.title("About")
    st.sidebar.markdown("""
            ### About this App
            This app was created by Nikhil Saxena for LMRI team use. It allows for comprehensive data analysis, including filtering, factor analysis, and random forest classification. 
                
            **Contact:** 
            - Email: [Nikhil.Saxena@lilly.com](mailto:Nikhil.Saxena@lilly.com)
                
            **Features:**
            - Upload and filter datasets
            - Perform factor analysis with customizable settings
            - Train and evaluate a Random Forest classifier with optional hyperparameter tuning
            - Visualize results with ROC curves and feature importance
                
            ---
            """, unsafe_allow_html=True)
    
    st.header("Upload your dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data preview:")
        st.write(df.head())

        # Show original data shape
        st.write("Original Data Shape:")
        st.write(df.shape)

        # Multiple filtering options
        st.header("Filter Data")
        filter_columns = st.multiselect("Select columns to filter:", df.columns)
        filters = {}
        for col in filter_columns:
            unique_values = df[col].unique()
            if pd.api.types.is_numeric_dtype(df[col]):
                selected_values = st.multiselect(f"Select values for '{col}':", unique_values)
                filters[col] = selected_values
            else:
                selected_values = st.multiselect(f"Select values for '{col}':", unique_values)
                filters[col] = selected_values

        filtered_df = df.copy()
        for col, selected_values in filters.items():
            if selected_values:
                filtered_df = filtered_df[filtered_df[col].isin(selected_values)]

        st.write("Filtered Data:")
        st.write(filtered_df)

        # Show filtered data shape
        st.write("Filtered Data Shape:")
        st.write(filtered_df.shape)

        # Allow user to select the outcome and independent variables
        outcome_var = st.selectbox("Select the outcome variable:", filtered_df.columns)
        independent_vars = st.multiselect("Select independent variables:", filtered_df.columns)
        
        if outcome_var and independent_vars:
            df2 = filtered_df[independent_vars]
            y = filtered_df[outcome_var]

            # Perform statistical tests and plots
            st.header("Statistical Tests and Plots")

            # Bartlett’s Test of Sphericity
            chi2, p = calculate_bartlett_sphericity(df2)
            st.write("Bartlett’s Test of Sphericity:")
            st.write(f"Chi-squared value: {chi2}, p-value: {p}")

            # Kaiser-Meyer-Olkin (KMO) Test
            kmo_values, kmo_model = calculate_kmo(df2)
            st.write("Kaiser-Meyer-Olkin (KMO) Test:")
            st.write(f"KMO Test Statistic: {kmo_model}")

            # Scree Plot
            fa = FactorAnalyzer(rotation=None, impute="drop", n_factors=df2.shape[1])
            fa.fit(df2)
            ev, _ = fa.get_eigenvalues()
            plt.figure(figsize=(10, 6))
            plt.scatter(range(1, df2.shape[1] + 1), ev)
            plt.plot(range(1, df2.shape[1] + 1), ev, "o-", color="b")
            plt.xlabel('Number of Factors')
            plt.ylabel('Eigenvalue')
            plt.title('Scree Plot')
            st.pyplot(plt)

            # Factor Analysis
            num_factors = st.slider("Select number of factors for factor analysis:", min_value=1, max_value=10, value=2)
            fa = FactorAnalyzer(rotation='varimax', n_factors=num_factors)
            fa.fit(df2)
            st.write("Factor Loadings:")
            st.write(pd.DataFrame(fa.loadings_, index=df2.columns))

            # Random Forest Classifier
            st.header("Train Random Forest Classifier")
            X_train, X_test, y_train, y_test = train_test_split(df2, y, test_size=0.3, random_state=42)
            rf = RandomForestClassifier()
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            st.write("Random Forest Classification Report:")
            st.write(classification_report(y_test, y_pred))
            st.write("Confusion Matrix:")
            st.write(confusion_matrix(y_test, y_pred))

            # ROC Curve
            y_proba = rf.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(10, 6))
            plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC)')
            plt.legend(loc='lower right')
            st.pyplot(plt)

            # GridSearchCV for hyperparameter tuning
            st.header("Hyperparameter Tuning with GridSearchCV")
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
            grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            st.write("Best Parameters from Grid Search:")
            st.write(grid_search.best_params_)
            st.write("Best Score from Grid Search:")
            st.write(grid_search.best_score_)

if __name__ == "__main__":
    main()
