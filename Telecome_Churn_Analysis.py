import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# ==========================================
# LOAD DATASET
# ==========================================

dataset = pd.read_csv("Telco_Customer_churn.csv")

print("Dataset Shape:", dataset.shape)

# ==========================================
# DATA CLEANING
# ==========================================

dataset['TotalCharges'] = pd.to_numeric(
    dataset['TotalCharges'],
    errors='coerce'
)

dataset['TotalCharges'] = dataset['TotalCharges'].fillna(
    dataset['TotalCharges'].median()
)

print("\nMissing Values:")
print(dataset.isnull().sum())

# ==========================================
# KPI 1 - OVERALL CHURN RATE
# ==========================================

total_customers = len(dataset)

churn_customers = dataset['Churn'].value_counts()['Yes']

churn_rate = (churn_customers / total_customers) * 100

print("\n==============================")
print("EXECUTIVE SUMMARY")
print("==============================")
print("Total Customers:", total_customers)
print("Churn Customers:", churn_customers)
print(f"Churn Rate: {churn_rate:.2f}%")

# ==========================================
# CHURN DISTRIBUTION
# ==========================================

plt.figure(figsize=(6,4))

ax = sns.countplot(
    x='Churn',
    data=dataset,
    hue='Churn',
    palette='coolwarm',
    legend=False
)

for container in ax.containers:
    ax.bar_label(container)

plt.tight_layout()
plt.title("Customer Churn Distribution")
plt.show()

# ==========================================
# CONTRACT TYPE VS CHURN
# ==========================================

contract_churn = pd.crosstab(
    dataset['Contract'],
    dataset['Churn'],
    normalize='index'
) * 100

print("\nContract Type vs Churn (%)")
print(contract_churn)

ax = contract_churn.plot(
    kind='bar',
    stacked=True,
    figsize=(10,6)
)

for container in ax.containers:
    ax.bar_label(
        container,
        fmt='%.1f%%'
    )

plt.title("Contract Type vs Churn")
plt.ylabel("Percentage")
plt.xlabel("Contract Type")

plt.xticks(
    rotation=30,
    ha='right'
)

plt.tight_layout()
plt.show()

# ==========================================
# PAYMENT METHOD VS CHURN
# ==========================================

payment_churn = pd.crosstab(
    dataset['PaymentMethod'],
    dataset['Churn'],
    normalize='index'
) * 100

print("\nPayment Method vs Churn (%)")
print(payment_churn)

ax = payment_churn.plot(
    kind='bar',
    figsize=(12,6)
)

plt.title("Payment Method vs Churn")
plt.ylabel("Percentage")

plt.xticks(
    rotation=45,
    ha='right'
)

for container in ax.containers:
    ax.bar_label(
        container,
        fmt='%.1f%%'
    )

plt.tight_layout()
plt.show()

# ==========================================
# TENURE GROUP ANALYSIS
# ==========================================

dataset['TenureGroup'] = pd.cut(
    dataset['tenure'],
    bins=[0,12,24,48,72],
    labels=['0-12','12-24','24-48','48-72']
)

tenure_churn = pd.crosstab(
    dataset['TenureGroup'],
    dataset['Churn'],
    normalize='index'
) * 100

print("\nTenure Group vs Churn (%)")
print(tenure_churn)

ax = tenure_churn.plot(
    kind='bar',
    figsize=(10,6)
)

for container in ax.containers:
    ax.bar_label(
        container,
        fmt='%.1f%%'
    )

plt.title("Tenure Group vs Churn")

plt.tight_layout()

plt.show()

# ==========================================
# MONTHLY CHARGES VS CHURN
# ==========================================

plt.figure(figsize=(8,5))

sns.boxplot(
    x='Churn',
    y='MonthlyCharges',
    data=dataset
)

plt.title("Monthly Charges vs Churn")
plt.show()

# ==========================================
# REVENUE AT RISK
# ==========================================

revenue_risk = dataset.loc[
    dataset['Churn'] == 'Yes',
    'MonthlyCharges'
].sum()

print(f"\nRevenue At Risk Per Month: ${revenue_risk:.2f}")

# ==========================================
# CORRELATION HEATMAP
# ==========================================

# ==========================================
# CORRELATION HEATMAP
# ==========================================

temp_data = dataset.copy()

for col in temp_data.select_dtypes(
    include=['object', 'category']
).columns:
    temp_data[col] = LabelEncoder().fit_transform(
        temp_data[col].astype(str)
    )

plt.figure(figsize=(14,10))

sns.heatmap(
    temp_data.corr(numeric_only=True),
    cmap='coolwarm',
    center=0,
    annot=False
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.show()

# ==========================================
# ONE HOT ENCODING
# ==========================================

dataset = pd.get_dummies(
    dataset,
    columns=[
        'gender',
        'Partner',
        'Dependents',
        'PhoneService',
        'MultipleLines',
        'InternetService',
        'OnlineSecurity',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport',
        'StreamingTV',
        'StreamingMovies',
        'Contract',
        'PaperlessBilling',
        'PaymentMethod'
    ],
    drop_first=True
)

dataset['Churn'] = LabelEncoder().fit_transform(
    dataset['Churn']
)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

from sklearn.model_selection import train_test_split

X = dataset.drop(
    ['customerID', 'Churn', 'TenureGroup'],
    axis=1
)

y = dataset['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# FEATURE SCALING
# ==========================================

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# ==========================================
# LOGISTIC REGRESSION MODEL
# ==========================================

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

# Accuracy

lr_accuracy = accuracy_score(
    y_test,
    lr_pred
)

print("\n==============================")
print("LOGISTIC REGRESSION RESULTS")
print("==============================")
print(f"Accuracy: {lr_accuracy:.2%}")

# Classification Report

print("\nClassification Report")
print(classification_report(
    y_test,
    lr_pred
))

# Confusion Matrix

lr_cm = confusion_matrix(
    y_test,
    lr_pred
)

print("\nConfusion Matrix")
print(lr_cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=lr_cm,
    display_labels=["No Churn", "Churn"]
)

disp.plot(cmap="Greens")

plt.title("Logistic Regression Confusion Matrix")

plt.show() 

# ==========================================
# RANDOM FOREST MODEL
# ==========================================

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ==========================================
# MODEL EVALUATION
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")
print(f"Accuracy: {accuracy:.2%}")

print("\nClassification Report")
print(classification_report(
    y_test,
    y_pred
))

# ==========================================
# ACCURACY COMPARISON
# ==========================================

comparison = pd.DataFrame({
    'Model': [
        'Logistic Regression',
        'Random Forest'
    ],
    'Accuracy': [
        lr_accuracy,
        accuracy
    ]
})

print("\n==============================")
print("ACCURACY COMPARISON")
print("==============================")
print(comparison)

# Plot Comparison

comparison['Accuracy'] = comparison['Accuracy'] * 100

plt.figure(figsize=(8,5))

ax = sns.barplot(
    data=comparison,
    x='Model',
    y='Accuracy'
)

for container in ax.containers:
    ax.bar_label(
    container,
    fmt='%.2f%%'
)

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.ylim(0,100)

plt.tight_layout()
plt.show()

# ==========================================
# CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "No Churn",
        "Churn"
    ]
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")
plt.show()

# ==========================================
# RANDOM FOREST ROC CURVE
# ==========================================

from sklearn.metrics import roc_curve, roc_auc_score

y_prob = model.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

auc_score = roc_auc_score(
    y_test,
    y_prob
)

plt.figure(figsize=(8,6))

plt.plot(
    fpr,
    tpr,
    label=f'AUC = {auc_score:.3f}'
)

plt.plot(
    [0,1],
    [0,1],
    '--'
)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.title('Random Forest ROC Curve')

plt.legend()

plt.show()

print(f"\nROC AUC Score: {auc_score:.3f}")

# ==========================================
# LOGISTIC REGRESSION ROC CURVE
# ==========================================
lr_prob = lr_model.predict_proba(X_test)[:,1]

fpr_lr, tpr_lr, _ = roc_curve(
    y_test,
    lr_prob
)

auc_lr = roc_auc_score(
    y_test,
    lr_prob
)

plt.figure(figsize=(8,6))

plt.plot(
    fpr_lr,
    tpr_lr,
    label=f'Logistic Regression AUC = {auc_lr:.3f}'
)

plt.plot([0,1],[0,1],'--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.title('Logistic Regression ROC Curve')

plt.legend()

plt.show()

print(f"\nLogistic Regression ROC AUC Score: {auc_lr:.3f}")

# ==========================================
# FEATURE IMPORTANCE
# ==========================================

feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nTop 10 Features Affecting Churn")
print(feature_importance.head(10))

plt.figure(figsize=(12,7))

ax = sns.barplot(
    data=feature_importance.head(10),
    x='Importance',
    y='Feature'
)

for container in ax.containers:
    ax.bar_label(
        container,
        fmt='%.3f'
    )

plt.title("Top 10 Features Affecting Churn")

plt.tight_layout()

plt.show()

# ==========================================
# KEY BUSINESS INSIGHTS
# ==========================================

print("\n==============================")
print("KEY BUSINESS INSIGHTS")
print("==============================")

print(f"""
1. Total Customers Analyzed : {total_customers}

2. Churn Customers : {churn_customers}

3. Overall Churn Rate : {churn_rate:.2f}%

4. Revenue At Risk Per Month : ${revenue_risk:.2f}

5. Month-to-Month contract customers show the highest churn risk.

6. Customers using Electronic Check payment method have the highest churn rate.

7. Customers with tenure less than 12 months are significantly more likely to churn.

8. Customers paying higher Monthly Charges tend to churn more frequently.

9. Long-term contract customers (1 Year and 2 Year) have substantially lower churn rates.

10. Top Features Influencing Churn:
""")

print("\nTop 5 Churn Drivers:")

for i, row in enumerate(
    feature_importance.head(5).itertuples(),
    start=1
):
    print(
        f"{i}. {row.Feature} ({row.Importance:.3f})"
    )


# ==========================================
# BUSINESS RECOMMENDATIONS
# ==========================================

print("\n==============================")
print("BUSINESS RECOMMENDATIONS")
print("==============================")

print("""
1. Target Month-to-Month customers with retention offers.

2. Encourage customers to switch to long-term contracts.

3. Improve customer support and technical support services.

4. Investigate high churn payment methods.

5. Focus on retaining customers during their first year.

6. Monitor high monthly charge customers closely.

7. Launch loyalty programs for high-risk customers.
""")