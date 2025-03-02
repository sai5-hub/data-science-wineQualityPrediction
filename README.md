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
Feature selection and engineering play a crucial role in model performance.<b>
🔹 One-hot encoding applied for categorical features.<b>
🔹 StandardScaler used for normalizing numerical features.<b>
🔹 Principal Component Analysis (PCA) reduced dimensionality from 11 to 8 features, improving model efficiency.<b>

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

> [!NOTE]
> Useful information that users should know, even when skimming content.

> [!TIP]
> Helpful advice for doing things better or more easily.

> [!IMPORTANT]
> Key information users need to know to achieve their goal.

> [!WARNING]
> Urgent info that needs immediate user attention to avoid problems.

> [!CAUTION]
> Advises about risks or negative outcomes of certain actions.

## **🎯 Key Insights & Conclusions**
The analysis revealed that alcohol content is the most significant factor influencing wine quality, highlighting its strong correlation with higher ratings. Among the models tested, Random Forest demonstrated superior performance, outperforming both Logistic Regression and Decision Tree in predictive accuracy. Additionally, feature engineering efforts—such as handling missing values, balancing the dataset, and selecting the most relevant features—led to a 10% improvement in model accuracy, further enhancing its reliability. These insights provide valuable guidance for winemakers, enabling them to adjust chemical compositions strategically to optimize wine quality and maintain consistency in production.
