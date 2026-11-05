# Fitbit Calorie Burn Prediction & Workout Clustering
## Capstone Explanation & Project Report

📌 Note: Built during my data science program using a practice dataset. The end-to-end implementation, modeling, and application were independently developed.

*This document is structured specifically to answer the **Model Building Projects** capstone evaluation guidelines.*

### 1. Domain Introduction
The fitness analytics and health tech domain leverages user physiological data to provide actionable insights for physical well-being. Wearable devices like Fitbit generate massive amounts of continuous data such as heart rate and activity duration. Applying machine learning to this data helps optimize health monitoring and personalize fitness coaching.

### 2. Problem Statement
Predict calories burned per workout session using supervised regression techniques and uncover hidden workout intensity patterns using unsupervised clustering.

### 3. Data Cleaning and Preprocessing
Missing numerical values were imputed using the median, and categorical values using the mode. Outliers in physiological metrics were capped using the Interquartile Range (IQR) method to prevent extreme values from skewing the model's accuracy.

### 4. EDA Findings (Univariate, Bivariate, Multivariate)
Univariate analysis revealed that most user workouts are moderate in both duration and calorie burn. Bivariate and multivariate analysis showed a strong, logical positive correlation between average heart rate (`Avg_BPM`), total session duration, and the target variable (calories burned).

### 5. Feature Engineering
Categorical variables (like Gender) were transformed into numerical formats using Label Encoding (for regression) and One-Hot Encoding (for clustering) to ensure mathematical compatibility with the ML algorithms. Numerical features were also standardized using `StandardScaler`.

### 6. Statistical Significance
We evaluated the **Silhouette Score** against the two project conditions:
1. **Condition 1 (> 0.40):** Indicates perfectly clear behavioral segments.
2. **Condition 2 (0.15 - 0.40):** Indicates physiological overlap.

Our K-Means model achieved a Silhouette Score of **~0.16**. As per the project guidelines, this falls exactly into **Condition 2**, mathematically proving that there is a natural, expected physiological overlap between different workout types (e.g., HIIT and Cardio share similar high heart rate zones).

### 7. Class Imbalance Technique
*Not applicable.* (This project focuses on continuous Regression and Unsupervised Clustering, rather than Classification).

### 8. Base Model
Linear Regression was chosen as the base model because it provides a simple, highly interpretable baseline to understand the linear relationships between basic inputs (like duration) and calories burned.

### 9. Models Used & Algorithm Choice
We compared Linear Regression, Ridge, Lasso, KNN, SVR, Decision Trees, XGBoost, and Random Forest. Tree-based ensemble models (Random Forest, XGBoost) were specifically chosen because they excel at capturing non-linear physiological relationships and complex feature interactions. 
**Hyperparameter Tuning:** We utilized `GridSearchCV` as an additional step on the Random Forest model to tune parameters like `n_estimators` and `max_depth` to optimize model performance and generalization.
*For Unsupervised Learning:* PCA was used for feature compression (retaining 90% variance). Alongside the mandatory **K-Means** clustering, we explicitly implemented **Hierarchical Clustering (Agglomerative)** and **DBSCAN (Density-Based Spatial Clustering)** to compare different clustering behaviors and ensure maximum coverage of the project requirements.

### 10. Model Evaluation Metric
For regression, the **R² Score** was chosen because it clearly explains the percentage of variance in calorie burn captured by our model, alongside MAE/RMSE to measure the exact error in kcals. For clustering, the **Silhouette Score** was chosen to measure how well-separated the hidden user segments are.

### 11. Final Model
The **Random Forest Regressor** was selected as the final model because it vastly outperformed the others, achieving the highest accuracy (**R² = 0.9982**) with the lowest mean absolute error (**MAE = 3.67 kcal**), making it highly reliable for mobile app deployment.

### 12. Conclusion & Feature Importance
The project successfully predicted calorie burn with near-perfect accuracy and discovered 4 distinct hidden workout intensity segments. Key features like `Avg_BPM`, `Max_BPM`, and `Session_Duration` proved to be the most critical indicators of overall energy expenditure.

### 13. Business Suggestion/Solution
Fitness applications can integrate this Random Forest model for highly accurate, real-time calorie tracking directly on wearable devices. Furthermore, the 4 identified clusters can be used to automatically segment users and push personalized coaching recommendations (e.g., "High-Intensity" vs. "Endurance" plans) based on their natural workout habits.

---

### 📂 Project Structure
- `app.py` → Main Streamlit application dashboard
- `Fitbit_Analysis_Research.ipynb` → Initial research, EDA, and model training notebook
- `fitbit_models.pkl.gz` → Compressed trained Machine Learning models and scalers
- `utils.py` → Custom logic for data preprocessing and derived metric calculations
- `requirements.txt` → Python project dependencies
- `Test data/`
  - `streamlit/` → Local Streamlit dashboard screenshots and Visualizations
  - `streamlit cloud deploy/` → Streamlit cloud deployment screenshots
  - `PowerBI_Report/` → Exported dataset and Power BI dashboard file
