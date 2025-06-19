from rest_framework import viewsets, status
from rest_framework.response import Response
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import os
from .serializer import KeplerMissionSerializer
import numpy as np
# Define a global variable to store the trained model
model = None
scaler = None

# Function to train the SVR model
def train_model():
    global model, scaler
    
    # Load the dataset
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'keplermission.csv')
    star_data = pd.read_csv(dataset_path)
    sa=SimpleImputer(missing_values=np.nan,strategy='mean')
    star_data['mass']=sa.fit_transform(star_data[['mass']])
    star_data['temp_calculated']=sa.fit_transform(star_data[['temp_calculated']])
    star_data['star_mass']=sa.fit_transform(star_data[['star_mass']])
    star_data['star_age']=sa.fit_transform(star_data[['star_age']]) 
    star_data['star_teff']=sa.fit_transform(star_data[['star_teff']])

    # Preprocess the data
    scaler = MinMaxScaler()
    star_data[['mass', 'star_metallicity', 'star_radius', 'star_teff']] = scaler.fit_transform(
        star_data[['mass', 'star_metallicity', 'star_radius', 'star_teff']])

    # Split data into features and target
    X = star_data[['mass', 'star_metallicity', 'star_radius', 'star_teff']]
    y = star_data['star_age']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the SVR model
    model = SVR(kernel='rbf', gamma='scale', C=1000)
    model.fit(X_train, y_train)

# Train the model when the module is loaded
train_model()

# Define a viewset to handle user input and make predictions
class StarAgePredictionViewSet(viewsets.ViewSet):
    def create(self, request):
        global model, scaler

        serializer = KeplerMissionSerializer(data=request.data)
        if serializer.is_valid():
            # Extract data from the serializer
            mass = serializer.validated_data.get('mass')
            metallicity = serializer.validated_data.get('metallicity')
            radius = serializer.validated_data.get('radius')
            temperature = serializer.validated_data.get('temperature')

            # Preprocess the input data
            input_data = [[mass, metallicity, radius, temperature]]
            input_data_scaled = scaler.transform(input_data)

            # Predict the output based on user input
            predicted_age = model.predict(input_data_scaled)[0]

            # Return the prediction to the frontend
            return Response({'predicted_age': predicted_age}, status=status.HTTP_200_OK)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
