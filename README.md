# Uda-disasterProject

# Udacity Data Scientist Nanodegree - Disaster Response Project

This repository contains the code for the disaster response project. The goal of the project is to analyze disaster data provided by Figure Eight to build a model for an API that classifies disaster messages. 

To achieve this goal, an ETL pipeline and a machine learning pipeline has been build, which are integrated into a Flask web app, where the user can enter a new message and it will
be categorized into the 36 different category.
## ETL Pipeline
An ETL Pipline is used to Extract, Transform, and Load data. Here the raw messages from Figure Eight are read, cleaned, and then store into a SQLite database. 


## Machine Learning Pipeline
In this part, the data is split into a training set and a test set. Then a MultioutputClassifier is trained with the data coming out of the ETL Pipeline. To find the optimal parameters for the classifier, GridSearch Method is used. In the end, the final model is saved into a pickle file, which uses the messages to predict classifications.


## How to train the model and run the web app
1. Place the disaster messages csv file and the disaster categories csv file from Figure Eight under the `data` folder.

2. Run the following commands in the project's root directory to set up your database and model.

	- To run ETL pipeline that cleans data and stores in database
`python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
	- To run ML pipeline that trains classifier and saves
`python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Adapt the database name and classifier name if needed in `app/run.py` file if needed. Then run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001 you should see the web app running. Put in a new message and see how it gets classified!

