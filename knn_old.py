import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import sys


#f_training = pd.read_csv("wine-training", delimiter=' ')
#f_test = open("wine-test", "r")

#df = pd.DataFrame(f_training)
#print(df)#df.to_excel('wine-training.xlsx', index = False, header= True)


#wine_training = f_training.read()
#wine_test = f_test.read()

filename1 = sys.argv[1]
filename2 = sys.argv[2]
k = int(sys.argv[3])

wine_training = pd.read_csv(filename1, delimiter=" ")
wine_test = pd.read_csv(filename2, delimiter=" ")

x_train = wine_training.drop(columns=['Class'])
x_test = wine_test.drop(columns=['Class'])

y_train = wine_training['Class'].values
y_test = wine_test['Class'].values


#print(y_test)
#creating KNN Classifier
knn = KNeighborsClassifier(n_neighbors = k)

#fit the classifier to the data
knn.fit(x_train,y_train)

#predict the class for test dataset
y_pred = knn.predict(x_test)

#print(y_pred)

#calculate model accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))