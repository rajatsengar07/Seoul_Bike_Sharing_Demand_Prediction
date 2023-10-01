import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle
from datetime import datetime
import os

app = Flask(__name__)

model = pickle.load(open('Model/xgboost_regressor_r2_0_point_95_v1.pkl' ,'rb'))

df = pd.DataFrame()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    global df
     
    ml_module_path = r"Model/xgboost_regressor_r2_0_point_95_v1.pkl"
    standard_scaler_path = r"Model/mmc.pkl"

    
    model = pickle.load(open(ml_module_path , 'rb'))
    mmc = pickle.load(open(standard_scaler_path , 'rb'))
    
         
    
    
    def date_str_to_datetime(date):
        dt = datetime.strptime( date , '%Y-%m-%d')
        return {'Day': dt.day , 'Month': dt.month ,'Year':dt.year , 'Day_Name': dt.strftime('%A')}
    
    
    def seasons_to_df( Seasons):
        seasons_col = ['Spring', 'Summer', 'Winter']
        seasons_data = np.zeros((1 , len(seasons_col)))
    
        df_seasons = pd.DataFrame(seasons_data , columns =seasons_col ,dtype = int)
        if Seasons in seasons_col:
            df_seasons[Seasons] = 1
        return df_seasons
    
    
    def days_to_df(Day_Name):
        days_col = ['Monday', 'Saturday', 'Sunday','Thursday', 'Tuesday', 'Wednesday']
        days_data = np.zeros((1 , len(days_col)))
        
        df_days = pd.DataFrame(days_data , columns =days_col ,dtype = int)
        if Day_Name in days_col:
            df_days[Day_Name] = 1
        return df_days
    
     
    
    def user_input():
        input_features = request.form.to_dict() 
        Date = input_features['Date']
        Hour = input_features['Hour']
        Temperature = float(input_features['Temperature'])  # Assuming Temperature is a numeric value
        Humidity = float(input_features['Humidity'])
        Wind_speed = float(input_features['Wind_speed'])
        Visibility = float(input_features['Visibility'])
        Solar_Radiation = float(input_features['Solar_Radiation'])
        Rainfall = float(input_features['Rainfall'])
        Snowfall = float(input_features['Snowfall'])
        Seasons = input_features['Seasons']
        Holiday = input_features['Holiday']
        Functioning_Day = input_features['Functioning_Day']


    
        holiday_dic = {'No Holiday': 0 ,'Holiday': 1}
        functioning_Day_dic = {'Yes': 0 ,'No': 1}
        
        str_to_date = date_str_to_datetime(Date)
        
        user_input_list = [Hour ,Temperature,Humidity ,Wind_speed ,Visibility ,Solar_Radiation , Rainfall , Snowfall ,
                       holiday_dic[Holiday], functioning_Day_dic[Functioning_Day],str_to_date['Day'],str_to_date['Month'],
                       str_to_date['Year']]
        
        feature_names = ['Hour', 'Temperature(°C)', 'Humidity(%)',
                'Wind speed (m/s)', 'Visibility (10m)', 'Solar Radiation (MJ/m2)',
                'Rainfall(mm)', 'Snowfall (cm)', 'Holiday', 'Functioning Day', 'Day',
                'Month', 'Year']
        
        df_u_input = pd.DataFrame([user_input_list] , columns = feature_names )
        
        seasons_df = seasons_to_df(Seasons)
        
        days_name_df = days_to_df(str_to_date['Day_Name'])
        
        df_for_pred = pd.concat([df_u_input ,seasons_df ,days_name_df],axis =1)
        
        return df_for_pred
    
    
    def scaled_data(df):
        df = user_input()
        scaled_df = mmc.transform(df)
        return scaled_df
    
        
    features_value = scaled_data(user_input)
    feature_names = ['Date','Hour', 'Temperature(°C)', 'Humidity(%)',
                'Wind speed (m/s)', 'Visibility (10m)', 'Solar Radiation (MJ/m2)',
                'Rainfall(mm)', 'Snowfall (cm)', 'Seasons','Holiday', 'Functioning Day']
        
    output = int(model.predict(features_value)[0].round())


    input_feat = request.form.to_dict() 
    key_names = list(input_feat.keys())
    key_values = list(input_feat.values())

    # Create a DataFrame from the dictionary
    new_data = {**dict(zip(key_names, key_values)), **{'Predicted Output': output}}
    new_df = pd.DataFrame([new_data])

    # Concatenate the DataFrames
    df = pd.concat([df, new_df])
    print(df)   
    # input and predicted value store in df then save in csv file
    df.to_csv('smp_data_from_app.csv', index = False)
        
        
    return render_template('index.html', prediction_text="Rented Bike Count Prediction with respect to time and hour is {} ".format(output ))



    


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
    