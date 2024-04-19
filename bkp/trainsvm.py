import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

cell_df = pd.read_csv('Research-Nit/output/output.csv')[:10]
malignant_df = cell_df[cell_df['pii_exist'] == 1]
benign_df = cell_df[cell_df['pii_exist'] == 0]

feature_df= cell_df.drop(cell_df.columns[:2], axis=1)
print(cell_df)

# #let say it have 100 rows, i am just, picked 9 colums out of it  
# #thus 9 colums our of 11

# # independent variable
# X = np.asarray(feature_df)
# #dependent variable
# y=np.asarray(cell_df['pii_exist'])


# from sklearn.model_selection import train_test_split
# X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size = 0.1,random_state=4)

# from sklearn import svm

# classifier = svm.SVC(kernel='linear',gamma='auto',C=1)
# classifier.fit(X_train,Y_train)
# y_predict = classifier.predict(X_test)

# from sklearn.metrics import classification_report,confusion_matrix
# print(classification_report(Y_test,y_predict))
# print(confusion_matrix(Y_test,y_predict))