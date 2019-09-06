import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sqlalchemy import create_engine
import pickle 

def load_data(database_filepath):
    '''
    This function take the location of DataBase
    and creates features and target column with category names
    '''
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterMessages', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names=df.columns[4:].tolist()
   
    return X, Y, category_names

def tokenize(text):    
    
    ''' Take the text and tokenize it 
        preprocessing : removing special symbols, stopwords, stemming &  returning lemmatized text
    '''
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)    
    tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    
    clean_tokens = []
    for tok in tokens:
        stemmed = stemmer.stem(tok)
        clean_tok = lemmatizer.lemmatize(stemmed).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Build a ML pipeline using tfidf, random forest, and gridsearch 
    and return the resut of grid search
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
#     grid Search taking too long for all parameters , so commenting for now - uncomment if required
    parameters = {
                  'clf__estimator__n_estimators': [50, 100]
#                   ,'clf__estimator__min_samples_split': [2, 3, 4],
#                   'clf__estimator__max_depth': [10, 20, 50 ],
#                   'clf__estimator__criterion': ['entropy', 'gini']
                  }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Will use the model fit to evaluate model performance on  test data
    Arguments:
    Model : trained model
    X_test, Y_test : Test feature and response data
    catgeory_name : Category Names
    
    Output: Model accuarcy and Classification Report 
    ''' 
    Y_pred = model.predict(X_test)
    
    for i in range(Y_test.shape[1]):    
        print("Category:", category_names[i],"\n", classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))
        print("Accuracy",(Y_test.iloc[:, i] == Y_pred[:, i]).mean())


def save_model(model, model_filepath):
    ''' 
    save the model as a pickle file
    Arguments:
    model: Model to be saved
    model_filepath : path of pickle file
    
    Output:
    A pickle file of saved model
    '''
    
    pickle.dump(model, open(model_filepath, 'wb'))


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