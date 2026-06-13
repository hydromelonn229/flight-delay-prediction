# Flight Delay Predictor and Analytics

This project analyzes flight delays using exploratory data analysis and trains machine learning models to predict operational delay risks for carrier-airport pairs on a monthly basis. It features an interactive dashboard to explore insights and perform real-time delay risk predictions.

---

## Project Steps

### Step 1: Exploratory Data Analysis (EDA)
Individual flight-level passenger records were analyzed to evaluate baseline delay statistics and test if features like pilot name, passenger age, origin continent, or operating month had a statistically significant relationship with flight delays.

### Step 2: Target Engineering
To create a more practical and reliable forecasting target, the objective was shifted from predicting individual chaotic flight delays to predicting whether a carrier-airport combination would experience a high delay rate (defined as 20% or more of arriving flights delayed) in a given month.

### Step 3: Data Cleaning and Leakage Prevention
Features that occur downstream of a delay (such as cancellation counts, diversion counts, and specific delay category breakdowns) were dropped to prevent data leakage. Identification fields like carrier name and airport name were also dropped, leaving categorical codes for processing.

### Step 4: Preprocessing and Pipeline Construction
Categorical variables (carrier and airport) were one-hot encoded using scikit-learn's OneHotEncoder. Numerical variables (monthly flight volumes, cancellation rates, diversion rates, and cyclically encoded month values using sine and cosine functions) were passed through a ColumnTransformer pipeline.

### Step 5: Model Training and Serialization
Two pipelines were trained on historical records using an 80/20 train-test split:
1. Logistic Regression (configured with balanced class weights).
2. Random Forest Classifier (configured with 200 estimators).
The trained pipelines were serialized and saved into the models folder.

### Step 6: Interactive Dashboard Implementation
A Python Streamlit application was constructed to serve as the project interface, providing visual charts of the EDA findings and a prediction query form to calculate live route risk indexes.

---

## Key Findings

1. Demographic and Pilot Factors: Chi-Square and Pearson correlation tests proved that passenger age, passenger nationality, pilot identity, and origin continent do not have a statistically significant correlation with flight delays.
2. Seasonality: Operating months have a statistical impact on delays, with peaks clustering in summer vacation periods and winter weather months.
3. Key Drivers of Delay: The most critical predictors of a high-delay month are the cancellation rate (which indicates operational distress) and the airport traffic volume. Carrier identity has a minor impact compared to general traffic congestion.
4. Model Performance: 
   - Logistic Regression achieved a higher Recall (74%) but lower Precision (55%).
   - Random Forest achieved a higher Precision (70%) but lower Recall (42%).
   - The models present a direct operational trade-off depending on how false warnings are penalized.

---

## Stakeholder Support and Applications

### 1. Airline Operations Managers
- Standby Crew Scheduling: Use the high-recall Logistic Regression model to identify high-risk routes and pre-schedule backup flight crews, minimizing downstream scheduling disruptions.
- Buffer Margins: Adjust route time buffers during peak risk months identified by the seasonality analysis.

### 2. Airport Terminal Authorities
- Traffic Control and Gate Allocation: Anticipate gate congestion and taxi-way bottlenecks by monitoring the Delay Risk Index of arriving carriers during high-volume months.
- Ground Support Operations: Stage additional baggage handling and refueling crews during peak seasonal congestion periods.

### 3. Corporate Travel Managers
- Route Selection: Review the Delay Risk Index of major carrier-airport pairs to select more reliable flight routes for business travel, reducing missed connections and lost travel hours.

### 4. Passengers and Travel Booking Platforms
- Booking Decisions: Integrate the Delay Risk Index into search results to warn passengers about routes that are historically prone to severe delays, helping them make more informed connection choices.
