import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
from geopy.distance import geodesic
from sklearn.model_selection import train_test_split
import re
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

df = pd.read_csv('./deliverytime.csv')

df.info()

# Checking data / Gathering information


68.063824 -> lewo

# processing

target = "Time_taken(min)"

# Excluding non important predicators
df.drop(columns=["ID"])

X = df.drop(columns=[target])
y = df[target]

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.columns.difference(num_cols).tolist()

print(num_cols)
print(cat_cols)

numeric_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ]
)

baseline_model = Pipeline(steps=[
    ("prep", preprocess),
    ("reg", LinearRegression()),
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
baseline_model.fit(X_train, y_train)

# feature engineering
# first let's create feature to have distance between restaurant and target
df['distance_km'] = [
    geodesic((lat1, lon1), (lat2, lon2)).km 
    for lat1, lon1, lat2, lon2 
    in zip(df['Restaurant_latitude'], df['Restaurant_longitude'], 
           df['Delivery_location_latitude'], df['Delivery_location_longitude'])
]

# extract city and restaurant id 
df['city_code'] = df['Delivery_person_ID'].str.split('RES').str[0].str.strip()
df['restaurant_id'] = df['Delivery_person_ID'].str.extract(r'(RES\d+)')
df['city_res_code'] = df['city_code'] + df['restaurant_id']

# world direction
def calculate_angle(lat_r, lon_r, lat_t, lon_t):
    lat_r, lon_r, lat_t, lon_t = map(np.radians, [lat_r, lon_r, lat_t, lon_t])
    dlon = lon_t - lon_r
    y = np.sin(dlon) * np.cos(lat_t)
    x = np.cos(lat_r) * np.sin(lat_t) - np.sin(lat_r) * np.cos(lat_t) * np.cos(dlon)
    angle = np.degrees(np.arctan2(y, x))

    return (angle + 360) % 360


def get_direction(ang):
    if (ang >= 337.5) or (ang < 22.5): return 'N'
    if 22.5 <= ang < 67.5: return 'NE'
    if 67.5 <= ang < 112.5: return 'E'
    if 112.5 <= ang < 157.5: return 'SE'
    if 157.5 <= ang < 202.5: return 'S'
    if 202.5 <= ang < 247.5: return 'SW'
    if 247.5 <= ang < 292.5: return 'W'
    if 292.5 <= ang < 337.5: return 'NW'


df['angle'] = [
    calculate_angle(lat_r, lon_r, lat_t, lon_t)
    for lat_r, lon_r, lat_t, lon_t in 
    zip(df['Restaurant_latitude'], df['Restaurant_longitude'], 
        df['Delivery_location_latitude'], df['Delivery_location_longitude'])
]

df['world_direction'] = df['angle'].apply(get_direction)
