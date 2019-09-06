# Disaster Response Pipeline Project

## Table of Content
1. Project Motivation
2. Toolkit Used
3. Files Used
4. Output
5. Instruction

### Project Motivation

  This project intents to classify messages sent at the time of disaster with the application of data enginnering on natural language and building a pipeline of machine learning for text data classification
  
  App: 
![alt text](https://github.com/Akhil-Anurag/Disaster-Response-/blob/master/Capture.PNG "Flask App")

### Toolkit Used

  This project was build on Python version 3.5 with NLTK library for text data engineering. Used re, pickle, sqlalchemy, numpy, pandas, sklearn from python.
  
### Files Used

- app
  - template
     - master.html  # main page of web app
     - go.html  # classification result page of web app
  - run.py  # Flask file that runs app

- data
  - disaster_categories.csv  # data to process 
  - disaster_messages.csv  # data to process
  - process_data.py # code for text data enginenring
  - InsertDatabaseName.db   # database to save clean data to

- models
  - train_classifier.py # code for model creation
  - classifier.pkl  # saved model 

### Output

  - ELT Pipeline: Pipeline is to Extract, Transform, and Load process. Here, we read the dataset, clean the data, and then store it in a SQLite database after cleaning with pandas and processing using NLTK
  
  - Machine Learning Pipeline : After spliting data into a training set and a test set create a machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model that uses the message column to predict classifications for 36 categories (multi-output classification) and finally export your model to a pickle file.
  
  - Flask App : Dispaly results in a Flask web app with database file and saved model as input

### Instruction
  - Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

  - Run the following command in the app's directory to run your web app.
    `python run.py`

  - Go to http://0.0.0.0:3001/
  
