#import libraries
import pandas as pd
import joblib
from sklearn.svm import SVC

#Create dataframe. Download the dataset from https://www.kaggle.com/uciml/pima-indians-diabetes-database
data = pd.read_csv('diabetes.csv')  
#Drop the null values
data.dropna(inplace=True)
X = data.drop('Outcome',axis=1)
y = data['Outcome']

# Initiate the model and train the data
model = SVC(probability=True)
model.fit(X,y)

#Dump the model to disk
joblib.dump(model,open('model_pickle','wb'))
#To load our model back into memory
model = joblib.load(open('model_pickle','rb'))