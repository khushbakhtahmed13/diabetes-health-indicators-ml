import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve, f1_score, confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import DecisionBoundaryDisplay
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches


def plot_performance(model, x_train, y_train, x_test, y_test):

    #Training model on training data
    model.fit(x_train, y_train)

    #Prediction
    y_pred = model.predict(x_test)
    y_probs = model.predict_proba(x_test)[:,1]

    # Printing Accuracy & Classification Report
    print(f"Training Accuracy = {model.score(x_train, y_train):.4f}")
    print(f"Testing Accuracy = {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report\n", classification_report(y_test, y_pred, target_names=["Non-diabetic", "Diabetic"]))

    # Plot ROC
    auc = roc_auc_score(y_test, y_probs)
    fpr, tpr, _ = roc_curve(y_test, y_probs)

    plt.plot(fpr, tpr, c = 'b', linewidth = 2.5, label=f"Model (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], 'r--', label = f"Random Guess (AUC Score = 0.5)")
    plt.fill_between(fpr, tpr, color = 'blue', alpha=0.1)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve", fontweight = "bold")

    plt.legend(loc = "lower right")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_boundary(model, x_train, y_train, x_test, y_test):
    #Training on only 2 features
    model.fit(x_train.iloc[:,:2], y_train)
    x_subset = x_test.iloc[:,:2]

    #custom colors
    colors = ['#2196F3', '#FF9800']
    custom_cmap = ListedColormap(colors)

    #legend patch
    legend_elements = [
        mpatches.Patch(color = colors[0], label = 'Non-diabetic'),
        mpatches.Patch(color = colors[1], label = 'Diabetic')
    ]

    DecisionBoundaryDisplay.from_estimator(model, x_subset, response_method = "predict", cmap = custom_cmap, alpha = 0.5)
    plt.scatter(x_subset.iloc[:, 0], x_subset.iloc[:, 1], c = y_test, cmap = custom_cmap, edgecolors = "k", alpha = 0.8)

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision Boundary", fontweight = "bold")

    plt.legend(handles = legend_elements, loc = 'upper left', title = "Status")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model, x_train, y_train, x_test, y_test, labels):

    #Training Model
    model.fit(x_train, y_train)

    #Prediciton
    y_pred = model.predict(x_test)


    #Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels = [0,1,2,3,4])

    sns.heatmap(cm,
                annot = True,
                fmt = 'd',
                cmap = "coolwarm",
                xticklabels = labels,
                yticklabels = labels)

    plt.xlabel("\nPredicted Label", fontweight = "bold")
    plt.ylabel("Actual Label", fontweight = "bold")
    plt.title("Confusion Matrix\n", fontweight="bold")

    plt.tight_layout()
    plt.show()


def get_accuracy(model, x_train, y_train, x_test, y_test):

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print(f"Training Accuracy = {model.score(x_train, y_train):.4f}")
    print(f"Testing Accuracy = {accuracy_score(y_test, y_pred):.4f}")

def get_macro_f1_score(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    macro = f1_score(y_test, y_pred, average = "macro")
    print(f"Macro-F1 Score = {macro:.4f}")

def get_regression_metrics(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error = {mae:.2f}")
    print(f"Mean Squared Error = {mse:.2f}")
    print(f"Root Mean Squared Error = {rmse:.2f}")
    print(f"Training R² = {model.score(x_train, y_train):.2f}")
    print(f"Testing R² = {r2:.2f}")
