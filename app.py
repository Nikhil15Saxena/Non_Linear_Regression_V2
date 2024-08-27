#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from scipy import stats

def main():
    st.title("Linear Regression Assumptions Analysis")

    # Upload and preview data
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data preview:")
        st.write(df.head())

        # Allow user to select the outcome and independent variables
        outcome_var = st.selectbox("Select the outcome variable:", df.columns)
        independent_vars = st.multiselect("Select independent variables:", df.columns)

        if outcome_var and independent_vars:
            X = df[independent_vars]
            y = df[outcome_var]

            # Add constant term for intercept
            X_with_const = add_constant(X)
            
            # Linearity: Residuals vs Fitted Values
            st.header("Linearity Test")
            st.subheader("Residuals vs Fitted Values")
            model = sm.OLS(y, X_with_const).fit()
            predictions = model.predict(X_with_const)
            residuals = y - predictions
            plt.figure(figsize=(10, 6))
            plt.scatter(predictions, residuals)
            plt.axhline(0, color='red', linestyle='--')
            plt.xlabel('Fitted Values')
            plt.ylabel('Residuals')
            plt.title('Residuals vs Fitted Values')
            st.pyplot(plt)

            # Homoscedasticity: Breusch-Pagan Test
            st.header("Homoscedasticity Test")
            _, p_value, _ = sm.stats.diagnostic.het_breuschpagan(residuals, X_with_const)
            st.write(f"Breusch-Pagan p-value: {p_value}")

            # Normality: Jarque-Bera Test
            st.header("Normality Test")
            jb_test_stat, jb_p_value = stats.jarque_bera(residuals)
            st.write(f"Jarque-Bera Test Statistic: {jb_test_stat}")
            st.write(f"Jarque-Bera p-value: {jb_p_value}")

            # Autocorrelation: Durbin-Watson Test
            st.header("Autocorrelation Test")
            dw_stat = sm.stats.durbin_watson(residuals)
            st.write(f"Durbin-Watson Statistic: {dw_stat}")

            # Influence: Cook's Distance
            st.header("Influence Test")
            influence = model.get_influence()
            cooks_d = influence.cooks_distance[0]
            plt.figure(figsize=(10, 6))
            plt.stem(np.arange(len(cooks_d)), cooks_d, markerfmt=",")
            plt.xlabel('Observation Index')
            plt.ylabel("Cook's Distance")
            plt.title("Cook's Distance")
            st.pyplot(plt)

            # Display all results
            st.subheader("Test Results Summary")
            st.write(f"Linearity (Residuals vs Fitted): Check the plot for patterns.")
            st.write(f"Homoscedasticity (Breusch-Pagan Test): p-value = {p_value}")
            st.write(f"Normality (Jarque-Bera Test): p-value = {jb_p_value}")
            st.write(f"Autocorrelation (Durbin-Watson Test): Statistic = {dw_stat}")

if __name__ == "__main__":
    main()
