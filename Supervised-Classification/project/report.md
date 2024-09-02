# Report: Crop prediction

## Overview Data Set
In agriculture, the precise recommendation of crops is pivotal in ensuring optimal yield and sustainability. As farmers and agricultural experts delve deeper into data-driven approaches, the significance of leveraging comprehensive datasets, particularly those about soil composition, becomes increasingly evident. [The dataset under consideration embodies a wealth of information encompassing key factors such as Nitrogen, Phosphorus, and Potassium levels, alongside environmental variables like Temperature, Humidity, pH_Value, and Rainfall](https://www.kaggle.com/datasets/varshitanalluri/crop-recommendation-dataset). Understanding and analyzing this dataset is fundamental to making informed decisions that may enhance agricultural productivity, resource management, and overall crop health.

**RangeIndex:** 2200 entries, 0 to 2199  
**Data Columns (total 8 columns):**
| #  | Column       | Non-Null Count | Dtype   |
|----|--------------|----------------|---------|
| 0  | Nitrogen     | 2200 non-null  | int64   |
| 1  | Phosphorus   | 2200 non-null  | int64   |
| 2  | Potassium    | 2200 non-null  | int64   |
| 3  | Temperature  | 2200 non-null  | float64 |
| 4  | Humidity     | 2200 non-null  | float64 |
| 5  | pH_Value     | 2200 non-null  | float64 |
| 6  | Rainfall     | 2200 non-null  | float64 |
| 7  | Crop         | 2200 non-null  | object  |

**Dtypes:** float64(4), int64(3), object(1)  
**Memory usage:** 137.6+ KB

## Objective
In this project, the Gradio library was used to train and evaluate multiple classification models aimed at predicting crop types based on specific environmental conditions. The target variable was the crop type, and the models were assessed using feature importance and confusion matrix visualizations, presented in interactive tabs.

### Methodology
In this analysis, we focused on a classification task using a dataset with well-distributed classes, eliminating the need for techniques to address class imbalance. The steps in the methodology are as follows:

1. **Data Preprocessing**:
   - **Scaling**: The data was scaled using the `MinMaxScaler` function to normalize feature values and enhance model performance.
   - **Null Values**: We verified that there were no null values in the dataset, so no imputation was necessary.

2. **Encoding**:
   - **Label Encoding**: Given the presence of multiple classes to predict, we employed `LabelEncoder` to convert categorical labels into numerical format.

3. **Data Splitting**:
   - The dataset was split into training and testing subsets using an 80-20 split ratio.

4. **Model Selection and Hyperparameter Tuning**:
   - We utilized `GridSearchCV` to find the optimal hyperparameters for our models, ensuring that we select the best configuration based on cross-validated performance.

5. **Model Saving**:
   - The final trained model was saved using the `pickle` library for future use.

6. **Evaluation**:
   - **Feature Importances**: We plotted feature importances to understand the impact of each feature on the model's predictions.
   - **Confusion Matrix**: A confusion matrix was generated to visualize the performance of the classification model.

These steps were implemented primarily using `pandas` for data manipulation and `sklearn` for scaling, encoding, and modeling, you can see the notebook by clicking on this [link](https://github.com/galois1915/IBM-Machine-Learning/blob/main/Supervised-Classification/project/train-models.ipynb).

## Results and Discussion:
The metrics used are the F1 score (with beta=1) and accuracy. Since the target is a multiclass problem, these metrics were calculated globally by counting the total true positives, false negatives, and false positives (<code>'micro'</code> option in the function <code>fbeta_score</code>).
| Metrics|   Logistic Regression |   KNN |   Support Vector |   Desicion Tree |   Random Forest |   AdaBoost |   Bagging |
|--------|----------------------:|------:|-----------------:|----------------:|----------------:|-----------:|----------:|
|f score |                 0.982 | 0.986 |            0.986 |           0.986 |           0.993 |      0.227 |     0.993 |
|accuracy|                 0.982 | 0.986 |            0.986 |           0.986 |           0.993 |      0.227 |     0.993 |

The results show that **Random Forest** and **Bagging** performed best, both achieving an accuracy and F-score of **0.993**. This indicates that these ensemble methods are highly effective in handling the classification task for crop prediction, likely due to their ability to reduce overfitting and capture complex patterns in the data. **Logistic Regression**, **KNN**, **Support Vector**, and **Decision Tree** also performed well, with accuracy and F-scores ranging from **0.982** to **0.986**, showing they are robust but might not capture all the intricacies that the ensemble methods do. 

**AdaBoost** performed poorly with both metrics at **0.227**, which suggests that the base learners used in AdaBoost were not strong enough for this particular problem, or that the model overemphasized difficult cases, leading to a decline in overall performance.

Across all models, **Humidity** and **Rainfall** consistently appear as crucial features, indicating their strong influence on the crop prediction task. However, the importance of other features like **Potassium** and **Nitrogen** varies depending on the model used, reflecting different model-specific behaviors and interactions within the data. Therefore, model choice can significantly impact the interpretation of feature importance in this context.

### Conclusion
The study aimed to accurately predict crop types based on environmental conditions using multiple classification models. **Random Forest** and **Bagging** emerged as the most effective solutions, achieving high accuracy and F-scores of **0.993**, demonstrating their robustness in capturing complex data patterns. While **Humidity** and **Rainfall** consistently influenced predictions, the importance of **Potassium** and **Nitrogen** varied, suggesting that model choice impacts the significance of these features. This highlights the value of ensemble methods in providing reliable crop recommendations, contributing to better agricultural decision-making.

