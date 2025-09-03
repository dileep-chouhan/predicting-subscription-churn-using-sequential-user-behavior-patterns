# Predicting Subscription Churn Using Sequential User Behavior Patterns

## Overview

This project analyzes user behavior data from a subscription service to predict churn.  The goal is to identify patterns in user activity that precede cancellation and build a predictive model to proactively target at-risk subscribers.  The analysis involves exploring various user engagement metrics, identifying significant behavioral sequences, and training a machine learning model to classify users as likely to churn or not.  The resulting model can be used to inform targeted retention strategies.

## Technologies Used

* Python 3.x
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn

## How to Run

1. **Install Dependencies:**  Ensure you have Python 3.x installed.  Then, install the required Python libraries using pip:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Analysis:** Execute the main script:

   ```bash
   python main.py
   ```

## Example Output

The script will print key findings of the analysis to the console, including model performance metrics (e.g., accuracy, precision, recall, F1-score).  Additionally, the analysis generates several visualization files (e.g., plots showing churn rates over time, feature importance, etc.) in the `output` directory.  These visualizations aid in understanding the factors contributing to churn.  The exact filenames and content of the output will depend on the analysis performed.  Expect to see files such as `churn_rate_over_time.png` and potentially others depending on the chosen visualization methods.