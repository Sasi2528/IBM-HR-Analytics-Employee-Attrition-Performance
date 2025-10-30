# 👩‍💼 IBM HR Analytics – Employee Attrition & Performance

**Author:** Sasi Manivannan 

This project analyzes employee attrition patterns using the **IBM HR Analytics Employee Attrition & Performance** dataset.  
It applies multiple machine learning classification models to predict whether an employee is likely to leave the company.

---

## Project Overview
The goal of this project is to predict employee attrition (Yes/No) based on HR-related factors such as age, job satisfaction, salary, and work-life balance.  
The notebook includes data preprocessing, model training, evaluation, and visualization.

---

## Dataset
**Source:** IBM HR Analytics Employee Attrition & Performance Dataset  
**Target Variable:** `Attrition` (Binary: Yes / No)  
**Key Features:** `Age`, `BusinessTravel`, `Department`, `DistanceFromHome`, `Education`, `EnvironmentSatisfaction`, `JobRole`, `MonthlyIncome`, `OverTime`, etc.

---

## Machine Learning Models Used
- Logistic Regression  
- K-Nearest Neighbours (KNN)  
- Decision Tree Classifier  
- Random Forest Classifier  
- Support Vector Classifier (SVC)  
- Naïve Bayes  
- XGBoost Classifier  

---

## Evaluation Metrics
The models were evaluated using the following metrics:
- Accuracy  
- Precision  
- Recall  
- F1-Score  

---

## 📈 Model Performance Summary (After Tuning)

| Model | Accuracy | Attrition Recall | Attrition F1-score |
|--------|-----------:|----------------:|------------------:|
| **SVC** | **0.87** | 0.13 | 0.20 |
| Random Forest | 0.85 | 0.15 | 0.21 |
| Logistic Regression | 0.87 | 0.10 | 0.17 |

> **Best Model:** The **Support Vector Classifier (SVC)** achieved the best overall performance after hyperparameter tuning, offering the most balanced accuracy and generalization across both classes.

---

## Key Insights
- All models show challenges in predicting the minority “Attrition” class due to class imbalance.  
- Random Forest slightly improves recall but tends to overfit compared to SVC.  
- **SVC provides the most stable and generalizable performance after tuning.**  
- Logistic Regression remains a strong, interpretable baseline model.  

---

## Tools & Libraries
- Python 3.9+  
- Pandas | NumPy  
- Matplotlib | Seaborn  
- Scikit-learn  
- XGBoost  
- Google Colab

---

## Visualizations
- Correlation matrix of numerical features  
- Class imbalance analysis  
- Confusion matrix visualization for tuned models  

---

## ▶️ How to Run
1. Open `IBM_HR_Analytics_Employee_Attrition_&_Performance.ipynb` in **Google Colab**.  
2. Upload the dataset (CSV file).  
3. Run all cells sequentially.  
4. Review printed metrics and charts at the end for model comparison.

---

## Conclusion
- **Decision Tree** → Overfits (poor generalization).  
- **Random Forest** → Strong recall but slightly biased.  
- **SVC (Support Vector Classifier)** → **Best overall generalized model after tuning.**  
- **Logistic Regression** → Reliable and interpretable baseline.  

✅ **Final Conclusion:** The **SVC model** achieved the best performance for predicting employee attrition in this dataset.

---

## Author
**Sasi Manivannan**
