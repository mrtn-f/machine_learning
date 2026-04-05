from ucimlrepo import fetch_ucirepo 
from datasets.base import Dataset
  
def load_iris():
    # fetch dataset 
    iris = fetch_ucirepo(id=53) 
  
    # data (as pandas dataframes) 
    X = iris.data.features 
    y = iris.data.targets 

    d = Dataset(
        X=X, 
        y=y, 
        task="classification", 
        name="iris", 
        feature_names=list(iris.data.features.columns.values), 
        target_name=list(iris.data.targets.columns.values)
    )

    d.summary()

    return d

