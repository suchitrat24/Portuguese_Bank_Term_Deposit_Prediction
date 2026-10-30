# Portuguese Bank Term Deposit Prediction

## Project Overview

This project analyzes a Portuguese bank's telemarketing campaigns and predicts which customers are likely to subscribe to a term deposit, using data science and machine learning techniques.

The project was completed as part of the **DataMites AI Engineer Program (Internship Project)**.

The primary objective was to perform exploratory data analysis, identify the factors that drive subscriptions, build a classification model to rank customers by their likelihood of subscribing, and give the bank's marketing team practical suggestions on which customers to focus on.

## Project Objectives

- Perform data cleaning and exploratory data analysis (EDA).
- Handle hidden missing values, special codes, and duplicate records.
- Identify and prevent data leakage.
- Analyze how customer profile, contact history, timing, and economic conditions affect subscriptions.
- Build and compare multiple classification models on an imbalanced target.
- Tune hyperparameters and evaluate advanced ensembles.
- Provide data-driven recommendations to the marketing team.

## Dataset

The dataset contains **41,188 records and 21 columns**. Each row is one phone call made to a customer during the bank's direct marketing campaigns between **May 2008 and November 2010**.

The dataset includes information such as:
- Customer details: age, job, marital status, education
- Financial status: credit default, housing loan, personal loan
- Contact details: contact type, month, day of week, call duration
- Campaign history: number of calls, days since last contact, previous outcome
- Economic indicators: employment variation rate, consumer price index, consumer confidence index, Euribor 3-month rate, number of employees
- Target: whether the customer subscribed to a term deposit (`y`)

*Note: The target is imbalanced. Only 11.3% of customers subscribed.*

## Technologies & Libraries

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy

## Project Workflow

### 1. Data Cleaning and Preprocessing
- Read the file with the correct `;` separator.
- Examined dataset structure, data types, and duplicates (12 duplicate rows removed).
- Found hidden missing values recorded as `'unknown'` (e.g. 8,597 rows in `default`) and kept them as their own category.
- Replaced the special code `pdays = 999` ("never contacted", 96.3% of rows) with a `PREV CONTACTED` flag.
- Created age groups for analysis.
- Converted categorical features using Label Encoding and scaled features with `StandardScaler`.

### 2. Exploratory Data Analysis (EDA)
- Analyzed age, call duration, number of calls, and economic indicator distributions.
- Examined subscription rates by job, age group, previous outcome, contact type, and month.
- Studied the relationship between interest rates and subscriptions over time.
- Used histograms, boxplots, relational plots, pair plots, line plots, bar charts, and correlation heatmaps.

### 3. Outlier Analysis
Used the Interquartile Range (IQR) method and boxplots to identify extreme values.

Key findings:
- 468 customers were older than 69.5 years. They were kept because they are real customers and subscribe the most.
- 2,406 customers were called more than 6 times (maximum 56). These were kept and analyzed as a campaign issue.
- No outliers were data errors, so no rows were removed.

### 4. Feature Selection
- **Removed `duration`** to prevent data leakage, because call length is only known after the call ends.
- Removed `emp.var.rate` and `euribor3m` because they are highly correlated (0.91–0.97) with `nr.employed`.
- Final model used 17 features covering customer profile, campaign history, timing, and economic conditions.

### 5. Machine Learning Model Development

This project uses supervised binary classification to predict whether a customer will subscribe to a term deposit.

The following algorithms were implemented and evaluated:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN) Classifier
4. Random Forest Classifier
5. Gradient Boosting Classifier

Class imbalance was handled with a stratified train/test split (80/20) and balanced class weights.

### 6. Model Evaluation and Comparison

Because only 11.3% of customers subscribed, a model that always predicts "no" would be 88.7% accurate and useless. Models were therefore evaluated using:
- ROC-AUC
- Precision, Recall, and F1 Score
- PR-AUC
- 5-fold stratified cross-validation

| Model | Test ROC-AUC | Precision | Recall | F1 | CV ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Random Forest | **0.811** | 0.424 | 0.633 | 0.507 | **0.799** |
| Gradient Boosting | 0.807 | 0.375 | 0.644 | 0.474 | 0.797 |
| Decision Tree | 0.795 | 0.436 | 0.619 | **0.512** | 0.784 |
| Logistic Regression | 0.787 | 0.298 | **0.692** | 0.416 | 0.778 |
| KNN | 0.760 | **0.636** | 0.247 | 0.356 | 0.755 |

### 7. Hyperparameter Tuning and Advanced Ensembles
- **GridSearchCV** on Random Forest (`n_estimators` × `max_depth`) confirmed `max_depth=10`. Deeper trees overfit, and more than 100 trees added nothing.
- **RandomizedSearchCV** on Random Forest and HistGradientBoosting.
- **Soft Voting** and **Stacking** ensembles.

All tuned models and ensembles scored between **0.796 and 0.799** cross-validated ROC-AUC. These differences are well within normal fold-to-fold variation, which shows that the limit is the information in the data rather than the algorithm.

**Best Performing Model: Random Forest Classifier**

Random Forest achieved the highest ROC-AUC on both the test set and cross-validation. It was kept over the stacking ensemble because it performs equally well and is simpler to explain, maintain, and retrain.

### 8. Business Impact

If the bank calls only the customers ranked highest by the model:

| Customers called | Subscribers reached | Success rate |
|---|---:|---:|
| Top 10% | 46.7% | 52.6% |
| Top 20% | 65.7% | 37.0% |
| Top 30% | 74.6% | 28.0% |
| Everyone | 100% | 11.3% |

Calling the top 20% reaches about two-thirds of all subscribers with roughly 80% fewer calls.

## Key Findings

- Economic conditions had the largest effect: customers subscribed **24.5%** of the time when the Euribor rate was below 2%, compared with **4.8%** when it was above 2%.
- Customers who said yes in a previous campaign subscribed **65%** of the time.
- Customers over 60, retired customers, and students subscribed well above average, even when compared within the same economic period.
- Success rates dropped with each additional call: **13%** on the first call and **3.1%** after more than 10 calls.
- Mobile phone calls performed about **1.4 times** better than landline calls within the same economic period.
- May had the most calls (13,767) but the lowest success rate (6.4%).

## Marketing Recommendations

- Call past subscribers, older/retired customers, and students first.
- Use the model's ranking to decide who to call.
- Limit each customer to about 3–5 calls per campaign.
- Prefer mobile numbers over landlines.
- Spread calls more evenly through the year instead of concentrating them in May.
- Run larger campaigns when interest rates are low.

## Challenges and Solutions

Some of the main challenges encountered included:

- **Hidden missing values:** Kept `'unknown'` as its own category instead of dropping over 20% of the data.
- **Special codes:** Replaced `pdays = 999` with a yes/no "previously contacted" flag.
- **Data leakage:** Removed call duration from the model because it is only known after the call.
- **Imbalanced target:** Used stratified splitting, balanced class weights, and ROC-AUC/recall/F1 instead of accuracy.
- **Feature redundancy:** Removed highly correlated economic indicators.
- **Misleading group comparisons:** Compared customer groups within the same economic period before making recommendations.

## Disclaimer

This project was completed for educational and portfolio purposes as part of the DataMites AI Engineer Program. The model results are based on historical campaign data from 2008–2010 and should not be interpreted as a validated production marketing system.

The dataset is not included in this repository.

## Author

**Suchitra Thanumalayan**

Data Science | Machine Learning | Artificial Intelligence

