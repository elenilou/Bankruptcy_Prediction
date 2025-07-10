# Bankruptcy Prediction using Machine Learning

## Project Overview

This project focuses on developing and evaluating various machine learning models to predict company bankruptcy based on financial and operational indicators. The analysis includes data loading, exploration, pre-processing, handling class imbalance, and evaluating model performance using cross-validation.

This was developed as the first assignment for the 

*Machine Learning* course at the *Department of Applied Informatics* of the *University of Macedonia*.

## Dataset

The dataset used in this project was provided as part of an academic assignment for the course "[ Machine Learning]" at "[University of Macedonia] ". Due to distribution restrictions, the dataset itself cannot be shared publicly in this repository.

The dataset contains financial and operational data for companies over several years (2006-2009). Each entry represents a company-year observation with features such as:
- Financial ratios (e.g., Liquidity, Profitability, Debt Ratios)
- Operational indicators (e.g., Inventory Turnover, Personnel Size)
- Binary indicators (e.g., Exports, Imports, Agencies)
- A class label indicating whether the company was Healthy (1) or Bankrupt (2) in the following year.

The dataset exhibits a significant class imbalance, with a much larger number of healthy companies compared to bankrupt ones.

## Project Steps

The following steps were performed in this project:

1.  **Data Loading and Initial Exploration:** Loaded the data from an Excel file into a pandas DataFrame and performed basic checks.
2.  **Data Pre-processing:**
    *   Converted the class labels to a binary format (0 for Healthy, 1 for Bankrupt).
    *   Checked for missing values (no missing values were found).
    *   Applied Min-Max Scaling to normalize the input features to a range between 0 and 1.
3.  **Handling Class Imbalance:** Used Stratified K-Fold cross-validation to ensure representative splits of the data. Applied undersampling to the training set within each fold to achieve a balanced ratio of 3 Healthy companies for every 1 Bankrupt company, addressing the class imbalance issue during model training.
4.  **Model Training and Evaluation:** Trained and evaluated several machine learning classifiers using the balanced training data and the original test data for each fold of the cross-validation. The classifiers included:
    *   Linear Discriminant Analysis (LDA)
    *   Logistic Regression (LR)
    *   Decision Tree Classifier (Ctree)
    *   Random Forest Classifier (RF)
    *   K-Nearest Neighbors (kNN)
    *   Gaussian Naive Bayes (NB)
    *   Support Vector Machine (SVM)
    *   Gradient Boosting Classifier (GB)
5.  **Performance Metrics:** Evaluated the models using various metrics, including Accuracy, Precision, Recall, F1-Score, and AUC-ROC. Confusion matrices were generated and saved for each model and fold.
6.  **Results:** The performance metrics and confusion matrices were saved to CSV and image files, respectively, for further analysis and comparison.

## Code Structure

The project code is organized in a Jupyter Notebook (`.ipynb` file). Key sections include:

*   Data loading and initial setup.
*   Data exploration and visualization (bar plots, box plots).
*   Data pre-processing (normalization, handling class labels).
*   Implementation of Stratified K-Fold cross-validation and undersampling.
*   Definition of a function for training and evaluating individual models.
*   Looping through different classifiers and folds to perform the simulation.
*   Saving results and confusion matrices.

## Results Summary

The performance of the models was evaluated based on their average metrics across the 4 folds.

### Key Findings:

* [cite_start]**Best Overall Models:** Based on the **F1-Score**, which provides a balance between Precision and Recall, four models stood out: **Random Forest (RF)**, **k-Nearest Neighbors (kNN)**, **Decision Tree (Ctree)**, and **Gradient Boosting (GB)**[cite: 189].
* [cite_start]**Top Performer:** **Random Forest (RF)** emerged as the most balanced and effective model with the highest F1-Score of **58.24%**[cite: 149, 167, 196]. [cite_start]It also achieved the highest TNR (96.59%), making it excellent at correctly identifying healthy companies[cite: 153].
* [cite_start]**Best at Identifying Bankruptcies (Recall):** The **Decision Tree (Ctree)** model achieved the highest Recall (**75.81%**), making it the most effective model at correctly identifying bankrupt companies[cite: 147, 181].
* [cite_start]**Performance Discrepancy:** A significant drop in performance was observed between the balanced training set and the imbalanced test set for all models, particularly in the Precision metric[cite: 162, 163]. This highlights the challenge posed by the dataset's class imbalance.
* [cite_start]**Underperforming Models:** Models like Logistic Regression, Naive Bayes, Linear Discriminant Analysis, and SVM showed significantly lower F1-Scores and were less suitable for this specific classification task[cite: 191, 198].

| Model                | F1-Score (Avg) | Recall (Avg) | Precision (Avg) | Accuracy (Avg) | TNR (Avg)  |
| :------------------- | :------------: | :----------: | :-------------: | :------------: | :--------: |
| **Random Forest (RF)** | [cite_start]**58.24%** [cite: 139]   | [cite_start]69.96% [cite: 139]   | [cite_start]55.96% [cite: 139]      | [cite_start]94.97% [cite: 139]     | [cite_start]**96.59%** [cite: 139] |
| **k-Nearest Neighbors (kNN)**| [cite_start]57.88% [cite: 139]   | [cite_start]73.19% [cite: 139]   | [cite_start]54.75% [cite: 139]      | [cite_start]94.27% [cite: 139]     | [cite_start]95.26% [cite: 139] |
| **Decision Tree (Ctree)** | [cite_start]55.66% [cite: 139]   | [cite_start]**75.81%** [cite: 139]   | [cite_start]53.18% [cite: 139]      | [cite_start]90.63% [cite: 139]     | [cite_start]90.74% [cite: 139] |
| **Gradient Boosting (GB)** | [cite_start]55.03% [cite: 139]   | [cite_start]68.62% [cite: 139]   | [cite_start]54.12% [cite: 139]      | [cite_start]92.92% [cite: 139]     | [cite_start]95.50% [cite: 139] |

[cite_start]*(Data sourced from the results table in the project report [cite: 139])*

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Contact

[cite_start]Ελένη Λούλα - Eleni Loula [cite: 6, 12]  
[cite_start][Ελένη Λούλα - Eleni Loula](https://github.com/elenilou) [cite: 6, 12]  
