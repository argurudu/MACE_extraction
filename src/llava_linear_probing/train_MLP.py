import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib
import pandas as pd
import ast

#Load data
train_data = pd.read_pickle("train_features.pkl")
val_data = pd.read_pickle("val_features.pkl")
test_data = pd.read_pickle("test_features.pkl")
X_train = np.stack(train_data["extracted_features"].values)
y_train = train_data["mistral_binary"].values
X_val = np.stack(val_data["extracted_features"].values)
y_val = val_data["mistral_binary"].values
X_all = np.concatenate([X_train, X_val], axis=0)
y_all = np.concatenate([y_train, y_val], axis=0)
X_test = np.stack(test_data["extracted_features"].values)
y_test = test_data["mistral_binary"].values

#Train sklearn MLP
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    max_iter=300,
    batch_size=32,
    early_stopping=True,
    n_iter_no_change=10,
    validation_fraction=10/90,
    random_state=42,
    verbose=True)
mlp.fit(X_all, y_all)

#Test
y_pred = mlp.predict(X_test)
y_prob = mlp.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))
auc = roc_auc_score(y_test, y_prob)
print(f"Test AUC: {auc:.4f}")

joblib.dump(mlp, "MACE_mlp_mistral.pkl")
print("MLP saved")
