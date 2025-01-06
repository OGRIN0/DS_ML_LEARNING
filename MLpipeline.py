from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from GridSearch import x_test, y_train

iris = datasets.load_iris()

x = iris.data
y = iris.target



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25)
pipe = Pipeline([('pca', PCA(n_components=2)), ('std', StandardScaler()),('dt', DecisionTreeClassifier())], verbose=True)
pipe.fit(x_train,y_train)

accuracy_score(y_test,pipe.predict(x_test))




