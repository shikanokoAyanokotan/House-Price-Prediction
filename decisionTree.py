from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from linearRegression import R_squared
import pickle


def train_decision_tree(X_train, X_test, y_train, y_test):
    # Define the parameter grid
    param_grid = {
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 8],
        'min_samples_leaf': [2, 5, 8],
        'max_features': ['log2', 'sqrt', 0.8],
        'max_leaf_nodes': [20, 30, 50], 
    }

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(estimator=DecisionTreeRegressor(), param_grid=param_grid, cv=5)
    grid_search.fit(X_train, y_train)

    # Use the model with best parameter
    model = grid_search.best_estimator_
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    print(f"Train R-squared: {R_squared(y_train_pred, y_train):.4f}")
    print(f"Test R-squared: {R_squared(y_test_pred, y_test):.4f}")


    # Save the model
    model_pkl_file = "DecisionTreeModel.pkl"
    with open(model_pkl_file, 'wb') as file:
        pickle.dump(model, file)
        print("Save model successfully.")

    return model
