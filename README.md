# **Wine Quality Prediction Analysis**

## **🌟 Overview**
This project focuses on predicting wine quality based on its chemical properties using Machine Learning techniques. The goal is to classify wines as either Good or Bad, leveraging various feature selection methods and supervised learning algorithms. By analyzing and visualizing key characteristics, this project aims to extract valuable insights that can help winemakers improve their products.

## **❓ Why This Project?**
Wine quality assessment traditionally relies on expert sensory evaluation, which is time-consuming and subjective. This project aims to develop an automated and data-driven approach to predict wine quality efficiently.<br>
Key motivations:<br>
✅ Improve wine classification accuracy using ML models.<br>
✅ Identify the most important chemical components influencing wine quality.<br>
✅ Compare different classification models to find the best-performing one.<br>

## **🔑 Key Features & Learnings**
🔹 Exploratory Data Analysis (EDA) to uncover trends in wine quality.<br>
🔹 Feature Engineering to enhance prediction accuracy.<br>
🔹 Comparison of multiple ML models (Logistic Regression, Decision Tree, Random Forest).<br>
🔹 Hyperparameter tuning to optimize model performance.<br>
🔹 Model evaluation using metrics like Accuracy, Precision, Recall, F1-Score.<br>
🔹 Data Visualization for deeper insights into chemical influences.<br>

## **🛠 Technologies, Tools, and Frameworks**
🔹 Programming Language:	Python<br>
🔹 Data Processing:	Pandas, NumPy<br>
🔹 Data Visualization:	Matplotlib, Seaborn<br>
🔹 Machine Learning:	Scikit-learn<br>
🔹 Feature Engineering:	PCA, StandardScaler<br>
🔹 Model Tuning:	GridSearchCV<br>
🔹 Notebook: Jupyter Notebook<br>

## **🚀 Data Source**
The dataset used in this project is the Wine Quality Dataset from Kaggle.<br>
🔹 Features: 11 physicochemical attributes (e.g., acidity, alcohol content).<br>
🔹 Target: Wine quality score (0-10), converted to binary classification (Good or Bad).<br>

## **👉 Installation & Usage**
📌 Prerequisites<br>
Ensure you have the following installed:<br>
✅ Python 3.8+<br>
✅ Jupyter Notebook<br>
✅ Required Python libraries<br>

**Quick Start**


## **📊 Exploratory Data Analysis (EDA)**
EDA helps uncover patterns and relationships between wine attributes.<br>
Key observations:<br>
📌 Alcohol content has a strong positive correlation with wine quality.<br>
📌 Volatile acidity negatively impacts quality—higher acidity = lower ratings.<br>
📌 Outliers in pH and sulfur dioxide levels were detected and handled.<br>

## **🔍 Key Visualizations**
🔹 Correlation Heatmap: To identify relationships between variables.
🔹 Boxplots: To detect outliers.
🔹 Histogram Distribution: To understand feature distributions.

## **🔬 Model Selection & Evaluation**

📌 Random Forest performed the best, achieving 85% accuracy due to its robustness against overfitting.

## **🛠 Feature Engineering & Selection**


## **⚡ Hyperparameter Tuning**
Using GridSearchCV, we optimized Random Forest hyperparameters:<b>
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)<b>
✅ Best parameters found: {'n_estimators': 200, 'max_depth': 15, 'min_samples_split': 5}


## **🎯 Key Insights & Conclusions**
📌 Alcohol content is the most important feature influencing wine quality.<b>
📌 Random Forest outperforms Logistic Regression & Decision Tree.<b>
📌 Feature engineering improved accuracy by 10%.<b>
📌 The model can help winemakers adjust chemical compositions for better quality.<b>
