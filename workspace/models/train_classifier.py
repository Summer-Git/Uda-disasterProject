import sys
import re
from sqlalchemy import create_engine
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import  MultiOutputClassifier
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    engine = create_engine('sqlite:///InsertDatabaseName.db')
    df = pd.read_sql_table("df_uda",engine)
    X = df["message"]
    Y = df.iloc[:,4:] 
    category_names = Y.columns.values.tolist()
    return X,Y,category_names


def tokenize(text):
    """进行文本处理"""
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    
    stemmed = [PorterStemmer().stem(w) for w in words]
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in stemmed]
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]
    
    return lemmed
    
    #sentences = sent_tokenize(lemmed)
    #print(sentences)


def build_model():
    """进使用pipeline搭建机器学习模型"""
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=20,class_weight=None)))
    ])
    # 调参
    # parameters = {
    # 'clf__estimator__n_estimators': (10,20),
    # 'clf__estimator__class_weight':["balanced",None]}
    # cv = GridSearchCV(pipeline, param_grid=parameters)

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """对模型进行评估"""
    y_pred = model.predict(X_test)
    y_pred_d = pd.DataFrame(y_pred).iloc    
    
    i = 0 
    for c_name in category_names:
        print("_______________________")
        print(c_name)
        print("precision_score",precision_score(Y_test[c_name], y_pred_d[:,i], average="macro"))
        print("recall_score",recall_score(Y_test[c_name], y_pred_d[:,i], average="macro"))
        print("f1_score",f1_score(Y_test[c_name], y_pred_d[:,i], average="macro"))
        i = i + 1


def save_model(model, model_filepath):
    import pickle
    s = pickle.dumps(model)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        # model = model.best_estimator_ 
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
