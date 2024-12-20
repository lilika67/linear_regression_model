# 🌾 Linear Regression Model

## 🎯 Mission
I created this linear regression model with a mission to help people in the field of agriculture get accurate information about history of crop yield for making decisions related to agricultural risk management and future predictions. Crops yield production value in hectogram per hectare (Hg/Ha) is got in a certain year according to the crop, weather conditions(Average rain fall per year,temperature) and Pesticides used in tonnes.

## 🏗️ Components of this Repository

### 📊 Model
To train the model I first choosed a dataset containing historical agricultural data and processed it to get the best model with low loss metric which I saved to it for prediction

###Description of dataset
The  dataset used contains agricultural yield data across various regions, crops, and years. It includes key factors such as rainfall, temperature, and pesticide usage, aiming to analyze their impact on crop yields.
[Source of dataset: Kaggle](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset)

### 🚀 Crop Yield Prediction API
This repository also contains a FastAPI-based RESTful API for predicting crop yields based on various inputs like crop type, year, rainfall, pesticide usage, and average temperature. This API is powered by a machine learning model (best_model_cropyield.joblib) trained before to predict crop yield (in hectograms per hectare) with high accuracy.

### 📱 FlutterApp
To help users predict the crop yield I created mobile app using flutter to help them get predictions in a better and a user friendly way and also to help them access the cropyield information through their mobile phone in an easy way.

## ✨ Features
- Predict crop yields for multiple crop types.
- Input validation using Pydantic.
- Asynchronous prediction using FastAPI and a thread pool executor.
- CORS support for cross-origin requests.

## 🛠️ How to test the FASTapi Locally
1. clone the repository by using the command 
```bash
git clone https://github.com/lilika67/linear_regression_model.git
```

2. navigate to project directory by 
```bash
cd linear_regression_model
```

3. Install all required packages by 
```bash
pip install -r requirements.txt
```

4. run the api by 
```bash
uvicorn summative.cropyieldApi.prediction:app
```

## 🌐 How to run the FASTapi on production
To run this fastApi you can use the swagger docs through the link https://linear-regression-model-66rf.onrender.com/docs

## 📱 How to run the mobile app.
1. clone the repo using the command 
```bash
git clone https://github.com/lilika67/linear_regression_model.git
```

2. navigate to project directory by 
```bash
cd flutterApp
```

3. install all required dependencies by 
```bash
flutter pub get
```

4. launch emulator by 
```bash
flutter run
```
**LINK TO YOUTUBE VIDEO**

[Video link](https://youtu.be/_XCb6AH0xVU)

**FlutterApp UI**

**Screen before making prediction**

![image](https://github.com/user-attachments/assets/a54e3c78-4d5b-48af-bbf9-9c4de664210a)

**Screen After making prediction**

![image](https://github.com/user-attachments/assets/7acde096-fcea-481b-ae51-044770aa2fe1)



