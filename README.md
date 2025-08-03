# Health Insurance Cost Predictor End-to-End ML Project

This project aims to build and deploy a machine learning model to accurately predict individual health insurance costs using demographic and health-related features. It follows a complete ML lifecycle: from problem definition to deployment, with advanced error handling and model optimization strategies.

![HomepageUI](./screenshorts/student_prediction_ui.png)

![HomepageUI](./screenshorts/predicted_value.png)

## Objective:
- Develop a high-accuracy model (target: >97%).
- Ensure that 95% of the predictions have less than 10% error from actual values.
- Deploy the model in the cloud for global access by insurance underwriters.
- Create an interactive and user-friendly Streamlit app for real-time predictions.

## Why this Model Require?
Health insurance pricing is a complex process influenced by multiple variables like age, BMI, number of children, smoking status, and region. Insurers often lack a fast, accurate, and automated way to estimate premiums, especially in real-time with minimal manual input. An ML model could streamline decision-making, minimize pricing errors, and enhance user experience.

## Task
- Develop an end-to-end ML pipeline to predict insurance charges with high accuracy.
- Analyze and handle outliers, data segmentation challenges, and improve model generalizability.
- Deploy the best model and build a Streamlit-based front-end for accessibility and ease of use.

## Solution Approch:

###  Data Collection, Analysis & Preprocessing
- Performed EDA to identify feature correlations.
- Applied encoding for categorical variables (LabelEncoder, OneHotEncoder).
- Used feature scaling where necessary (e.g., Min-Max Scaling for numeric features).

### Feature Selection
- Used variance_inflation_factor (VIF) to  measures how much the variance of a regression coefficient is inflated due to multicollinearity with other features.
- It helps detect redundant (highly correlated) features in a regression model and found income_level has highest VIF i.e 12.44 and income_level has 12.44.
- Once income_level is removed from feature rest of the feature VIF value comes  < 5. which shows less No multicollinearity.

### Model Development
- Trained and compared Linear Regression and XGBoost Regressor.
- Performed hyperparameter tuning using GridSearchCV and RandomizedSearchCV but both has same result.
- Chose XGBoost as the best-performing model with improved accuracy and error handling.

### Error Handling & Model Segmentation
- Identified that age feature was causing high error (>30%) in some cases.
- Applied segmentation-based modeling:
    - Trained two additional models:
        - One for age < 25.
        - Another for age ≥ 25
    - This reduced high-error predictions from 30% of total error to less than 2%.

### Evaluation Metrics
Model achieved:
  - 99.26% accuracy on training data
  - 98.75% accuracy on test data
  - Average prediction error margin: 1.63%
  - XGBoost slightly outperformed Linear Regression (by ~1%) on both datasets.

### Deployment
 - Built an interactive Streamlit app for prediction input and results display.
 - Packaged the best-performing model with joblib.
 - Designed a clean and intuitive UI.  - Application is cloud-deployable for underwriters to use remotely.

### Results
  - Built a robust and accurate ML model (XGBoost) meeting the project’s target metrics.
  - Reduced prediction errors significantly through data segmentation and analysis.
  - Delivered a production-ready solution with:
    - 98.7% accuracy on test data
    - <2% high error margin cases
  - Developed a Streamlit interface with dynamic input forms for real-time use by non-technical users.
  - Enabled cloud-based access, ensuring usability from any location.

### Key Achivements
Metric & Value
  - Train Accuracy	99.26%
  - Test Accuracy	98.75%
  - Average Error Margin	1.63%
  - Error > 30% (Initial)	30% of total predictions
  - Error > 30% (After Fixes)	<2% (after age-based segmentation)
  - Model Used	XGBoost (Best performer)
  - Deployment	Streamlit app, cloud-ready

## Tools and Technologies
- Languages: Python
- ML Libraries: scikit-learn, XGBoost, pandas, NumPy
- Visualization: Seaborn, Matplotlib
- Model Persistence: joblib
- Frontend: Streamlit
- Version Control: Git, GitHub
- Cloud Deployment: Streamlit community server
