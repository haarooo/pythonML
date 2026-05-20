import httpx
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

class CarService:


    def __init__(self):
        self.model = None
        self.poly = None
        self.ss = None
        self.r2 = None


    def car_data(self, car_list):

        df = pd.DataFrame(car_list)
        x = df[['efficiency' , 'mileage' , 'month' , 'accident' , 'changeUser']].values
        y = df['price'].values

        train_input , test_input , train_target , test_target = train_test_split(x , y , test_size=0.2 , random_state=42)

        poly = PolynomialFeatures(degree= 3 ,include_bias=False)
        poly.fit(train_input)
        train_poly = poly.transform(train_input)
        test_poly = poly.transform(test_input)
       
        ss = StandardScaler()
        ss.fit(train_poly)
        train_scale = ss.transform(train_poly)
        test_scale = ss.transform(test_poly)
        
        ridge = Ridge(alpha=1)
        ridge.fit(train_scale , train_target)
        r2 = ridge.score(test_scale , test_target)
        print(r2)

        
        self.model = ridge
        self.poly = poly
        self.ss = ss
        self.r2 = r2

        return r2



    def predict(self , car):
        
        car_df = pd.DataFrame([{
        'efficiency': car['efficiency'],
        'mileage': car['mileage'],
        'month': car['month'],
        'accident': car['accident'],
        'changeUser': car['changeUser']
        }])
        
        input_poly = self.poly.transform(car_df)
        input_scale = self.ss.transform(input_poly)

        predicted_price = self.model.predict(input_scale)[0]
        return predicted_price

    

    



car_service = CarService()