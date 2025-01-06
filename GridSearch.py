from sklearn.datasets import  load_breast_cancer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

ds = load_breast_cancer()

x = ds.data
y = ds.target

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25)

model = SVC()

model.fit(x_train, y_train)

prediction = model.predict(x_test)

print(classification_report(y_test, prediction))

param_grid = {
    'C': [0, 1, 10, 100],
    'gamma': ['scale', 'auto', 1, 0.1, 0.01, 0.001, 0.00001],
    'kernel': ['linear']
}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3, n_jobs=-11)

grid.fit(x_train, y_train)

grid.best_params_

grid_prediction = grid.predict(x_test)

print(classification_report(y_test,grid_prediction))
