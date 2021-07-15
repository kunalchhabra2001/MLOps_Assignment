import dvc.api
import pandas
import joblib
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
# with dvc.api.open(repo="https://github.com/kunalchhabra2001/MLOps_Assignment", path="data/creditcard.csv", mode="r") as fd:
# 		df = pandas.read_csv(fd)
# df_permutated = df.sample(frac=1)

# train_size = 0.8
# train_end = int(len(df_permutated)*train_size)

# df_train = df_permutated[:train_end]
# df_test = df_permutated[train_end:]
# df_train.to_csv('C:/Users/Kunal Chhabra/Documents/GitHub/MLOps_Assignment/data/processed/train.csv')
# df_test.to_csv('C:/Users/Kunal Chhabra/Documents/GitHub/MLOps_Assignment/data/processed/test.csv')
train = pandas.read_csv('C:/Users/Kunal Chhabra/Documents/GitHub/MLOps_Assignment/data/processed/train.csv')
test = pandas.read_csv('C:/Users/Kunal Chhabra/Documents/GitHub/MLOps_Assignment/data/processed/test.csv')

Y_train = train.pop('Class')
X_train = train

Y_test = test.pop('Class')
X_test = test

clf = DecisionTreeClassifier()

# # Train Decision Tree Classifer
clf = clf.fit(X_train,Y_train)

# #Predict the response for test dataset
y_pred = clf.predict(X_test)
filename = 'C:/Users/Kunal Chhabra/Documents/GitHub/MLOps_Assignment/models/model.pkl'
joblib.dump(clf, filename)
print("Accuracy:",metrics.accuracy_score(Y_test, y_pred))
print("F1 Score:",metrics.f1_score(Y_test, y_pred))
