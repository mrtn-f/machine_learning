from ucimlrepo import fetch_ucirepo 
from datasets.base import Dataset
  
def load_breast_cancer():
    breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
  
    # data (as pandas dataframes) 
    X = breast_cancer_wisconsin_diagnostic.data.features 
    y = breast_cancer_wisconsin_diagnostic.data.targets 

    d = Dataset(
        X=X, 
        y=y, 
        task="classification", 
        name="iris", 
        feature_names=list(breast_cancer_wisconsin_diagnostic.data.features.columns.values), 
        target_name=list(breast_cancer_wisconsin_diagnostic.data.targets.columns.values)
    )

    d.summary()

    return d