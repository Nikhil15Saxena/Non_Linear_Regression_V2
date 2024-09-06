import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, GridSearchCV
from factor_analyzer import FactorAnalyzer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import graphviz
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
    
# Streamlit app
def main():
    st.title("Non-Linear Driver Analysis XGBoost App")

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
            X = filtered_df[independent_vars]
            y = filtered_df[outcome_var]

            # Heatmap of correlation matrix
            st.write("Correlation Matrix:")
            plt.figure(figsize=(20, 10))
            sns.heatmap(X.corr(), cmap="Reds", annot=True)
            st.pyplot(plt)
            with st.expander("Description"):
                st.markdown("""
                **What it is**: A visual representation of the correlation matrix where the strength of correlation is represented by color intensity.
                
                **What it tells us**: Helps to identify the strength and direction of relationships between variables. High correlation values indicate multicollinearity.
                """)

            # Variance Inflation Factor (VIF)
            df2_with_const = add_constant(X)
            vif_data = pd.DataFrame()
            vif_data["Variable"] = df2_with_const.columns
            vif_data["VIF"] = [variance_inflation_factor(df2_with_const.values, i) for i in range(df2_with_const.shape[1])]
            vif_data = vif_data[vif_data["Variable"] != "const"]
            st.write("Variance Inflation Factor (VIF):")
            st.write(vif_data)
            with st.expander("Description"):
                st.markdown("""
                **What it is**: Measures the increase in variance of the estimated regression coefficients due to collinearity.
                
                **What it tells us**: VIF values above 10 indicate high multicollinearity, suggesting that the predictor variables are highly correlated and may not be suitable for regression analysis.
                """)

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

            # XGBoost with GridSearchCV
            param_grid = {
                'n_estimators': [100, 300, 500, 1000],
                'learning_rate': [0.01, 0.1, 0.3],
                'max_depth': [3, 4, 5],
                'alpha': [0, 0.1, 1],  # L1 regularization
                'lambda': [1, 2, 5]    # L2 regularization
            }
            xgb_model = xgb.XGBClassifier(random_state=42)
            grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='accuracy', cv=5, verbose=1, n_jobs=-1)
            grid_search.fit(X_train, y_train)

            # Best parameters and score
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            st.write(f"Best Parameters: {best_params}")
            st.write(f"Best Cross-Validation Score: {best_score}")

            # Best model
            best_model = grid_search.best_estimator_

            # Classification Report
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            st.write("Classification Report - Train Data:")
            st.text(classification_report(y_train, y_train_pred))

            st.write("Classification Report - Test Data:")
            st.text(classification_report(y_test, y_test_pred))

            # Feature importance
            importances = best_model.feature_importances_
            feature_importance_df = pd.DataFrame({'Feature': independent_vars, 'Importance': importances})
            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
            st.write("Feature Importance:")
            st.write(feature_importance_df)

            # Option to display tree
            if st.button("Display XGBoost Tree"):
                booster = best_model.get_booster()

                # Display the first tree (or any specific tree by changing num_trees parameter)
                st.write("Displaying the first tree from the XGBoost ensemble:")
                dot_data = xgb.to_graphviz(booster, num_trees=0)
                
                # Show the tree using graphviz in Streamlit
                st.graphviz_chart(dot_data.source)

if __name__ == "__main__":
    main()
