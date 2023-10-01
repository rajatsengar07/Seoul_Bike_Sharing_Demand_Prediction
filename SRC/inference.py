import pickle
import os
import pandas as pd
import numpy as np
from datetime import datetime

class Inference:
    
    def __init__(self, model_path, sc_path):
        self.model_path = model_path
        self.sc_path = sc_path

        if os.path.exists(self.model_path) and os.path.exists(self.sc_path):
            self.model = pickle.load(open(self.model_path , 'rb'))
            self.sc = pickle.load(open(self.sc_path , 'rb'))

        else:
            print('Model and Standard Scaler Path is not correct')


    def date_str_to_datetime(self, date):
        dt = datetime.strptime( date , '%d/%m/%Y')
        return {'Day': dt.day , 'Month': dt.month ,'Year':dt.year , 'Day_Name': dt.strftime('%A')}


    def seasons_to_df(self, Seasons):
        seasons_col = ['Spring', 'Summer', 'Winter']
        seasons_data = np.zeros((1 , len(seasons_col)))
    
        df_seasons = pd.DataFrame(seasons_data , columns =seasons_col ,dtype = int)
        if Seasons in seasons_col:
            df_seasons[Seasons] = 1
        return df_seasons

    
    def days_to_df(self, Day_Name):
        days_col = ['Monday', 'Saturday', 'Sunday','Thursday', 'Tuesday', 'Wednesday']
        days_data = np.zeros((1 , len(days_col)))
        
        df_days = pd.DataFrame(days_data , columns =days_col ,dtype = int)
        if Day_Name in days_col:
            df_days[Day_Name] = 1
        return df_days
    
    def user_input(self , ):
        print('Enter correct credentials for predicting rental bike count respected to time and hour')
        Date = input("Enter a date (format - dd/mm/yy): ")
        Hour = int(input("Enter an hour (range- 1 to 24): "))
        Temperature = float(input("Enter a temperature in (°C) : "))
        Humidity = float(input("Enter Humidity(%) (range - 1 to 100) : "))
        Wind_speed = float(input("Enter Wind speed in (m/s) : "))
        Visibility = float(input("Enter Visibility (10m) : "))
        Solar_Radiation = float(input("Enter Solar Radiation (MJ/m2) : "))
        Rainfall = float(input("Enter Rainfall in (mm) : "))
        Snowfall = float(input("Enter Snowfall in (cm) : "))
        Seasons = input("Enter a Season (choose from the given names - Spring, Summer, Winter & Autumn): ")
        Holiday = input("Enter Holiday or 'No Holiday' : ")
        Functioning_Day = input("Enter Yes or No : ")


        holiday_dic = {'No Holiday': 0 ,'Holiday': 1}
        functioning_Day_dic = {'Yes': 0 ,'No': 1}

        str_to_date = self.date_str_to_datetime(Date)

        user_input_list = [Hour ,Temperature,Humidity ,Wind_speed ,Visibility ,Solar_Radiation , Rainfall , Snowfall ,
                   holiday_dic[Holiday], functioning_Day_dic[Functioning_Day],str_to_date['Day'],str_to_date['Month'],
                   str_to_date['Year']]

        feature_names = ['Hour', 'Temperature(°C)', 'Humidity(%)',
            'Wind speed (m/s)', 'Visibility (10m)', 'Solar Radiation (MJ/m2)',
            'Rainfall(mm)', 'Snowfall (cm)', 'Holiday', 'Functioning Day', 'Day',
            'Month', 'Year']

        df_u_input = pd.DataFrame([user_input_list] , columns = feature_names )

        seasons_df = self.seasons_to_df(Seasons)

        days_name_df = self.days_to_df(str_to_date['Day_Name'])

        df_for_pred = pd.concat([df_u_input ,seasons_df ,days_name_df],axis =1)

        return df_for_pred

    
    def Prediction(self):
        df = self.user_input()
        scaled_data = self.sc.transform(df)
        pred = self.model.predict(scaled_data)
        return pred


if __name__ == '__main__':
    ml_module_path = r'C:\Users\rajat\OneDrive\Desktop\My projects-Data Analysis\Seoul Bike Sharing Demand Prediction Project\Model\xgboost_regressor_r2_0_point_95_v1.pkl'
    standard_scaler_path = r'C:\Users\rajat\OneDrive\Desktop\My projects-Data Analysis\Seoul Bike Sharing Demand Prediction Project\Model\sc.pkl'
    inference = Inference(ml_module_path ,standard_scaler_path)

    prediction  = inference.Prediction()
    print(" Rented Bike Count Prediction respected to time and hours :" ,round(prediction[0]))
