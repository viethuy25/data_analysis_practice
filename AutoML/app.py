import streamlit as st
import plotly.express as px
import pandas as pd
import ydata_profiling
# from autoviz.AutoViz_Class import AutoViz_Class
from pmdarima import auto_arima


from pycaret.regression import setup, compare_models, pull, save_model, load_model
from streamlit_pandas_profiling import st_profile_report
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from transformers import BertTokenizer, BertModel
import torch

def load_data(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path, index_col=None)
    return None

def save_data(df, file_path='dataset.csv'):
    df.to_csv(file_path, index=None)

def upload_dataset():
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset")
    if file:
        df = pd.read_csv(file, index_col=None)
        save_data(df)
        st.dataframe(df)
        return df
    return None

def exploratory_data_analysis(df):
    st.title("Exploratory Data Analysis")
    # profile_df = autoviz.auto_explore(df)
    profile_df = df.profile_report()
    st_profile_report(profile_df)

def run_modelling(df):
    chosen_target = st.selectbox('Choose the Target Column', df.columns)

    # Convert datetime columns to timestamp if selected as the target
    if pd.api.types.is_datetime64_any_dtype(df[chosen_target]):
        df[chosen_target] = pd.to_datetime(df[chosen_target]).astype(int) // 10**9

    if st.button('Run Modelling'):
        # setup(df, target=chosen_target)
        # setup_df = pull()
        # st.dataframe(setup_df)


        # best_model = compare_models()
        # compare_df = pull()
        # st.dataframe(compare_df)
        # save_model(best_model, 'best_model')

        st.write(df.dtypes)

        # Make sure the DataFrame is properly indexed
        df.reset_index(drop=True, inplace=True)

         # Exclude "ride_id" and other non-numeric columns from features
        non_numeric_columns = ["ride_id", "rideable_type", "start_station_name", "end_station_name", "member_casual"]
        numeric_columns = df.select_dtypes(include='float64').columns
        features_to_exclude = [chosen_target] + non_numeric_columns
        X = df.drop(columns=features_to_exclude)
        y = df[chosen_target]

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Linear Regression
        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)
        linear_pred = linear_model.predict(X_test)
        st.write("Linear Regression MSE:", mean_squared_error(y_test, linear_pred))

        # LightGBM
        lgbm_model = LGBMRegressor()
        lgbm_model.fit(X_train, y_train)
        lgbm_pred = lgbm_model.predict(X_test)
        st.write("LightGBM MSE:", mean_squared_error(y_test, lgbm_pred))

        # MLP (Neural Network)
        mlp_model = MLPRegressor()
        mlp_model.fit(X_train, y_train)
        mlp_pred = mlp_model.predict(X_test)
        st.write("MLP (Neural Network) MSE:", mean_squared_error(y_test, mlp_pred))

        # Random Forest (Additional Model)
        rf_model = RandomForestRegressor()
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        st.write("Random Forest MSE:", mean_squared_error(y_test, rf_pred))

        # # BERT (Transformer) - This is a placeholder; BERT is typically used for NLP tasks
        # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # model = BertModel.from_pretrained('bert-base-uncased')
        # inputs = tokenizer("Hello, this is a sample input.", return_tensors="pt")
        # outputs = model(**inputs)
        # st.write("BERT is not directly applicable for tabular data; this is just a placeholder.")

def download_model():
    with open('best_model.pkl', 'rb') as f:
        st.download_button('Download Model', f, file_name="best_model.pkl")


def main():
    if os.path.exists('dataset.csv'):
        df = load_data('dataset.csv')
    else:
        df = None

    with st.sidebar:
        st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
        st.title("AutoBikeML")
        choice = st.radio("Navigation", ["Upload", "Profiling", "Modelling", "Download"])
        st.info("This project application helps you build and explore your data.")

    if choice == "Upload":
        df = upload_dataset()

    if df is not None:
        if choice == "Profiling":
            exploratory_data_analysis(df)
        elif choice == "Modelling":
            run_modelling(df)
        elif choice == "Download":
            st.title ("TBD") #download_model()

if __name__ == "__main__":
    main()