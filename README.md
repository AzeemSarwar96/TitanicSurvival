# TitanicSurvival

# Titanic Survival Prediction

A beginner-friendly machine learning project that predicts whether a passenger survived the Titanic disaster, based on features like age, sex, ticket class, and fare.

Built with **Python, pandas, scikit-learn, and matplotlib**.

---

## Project Overview

On April 15, 1912, the Titanic sank after hitting an iceberg. Out of 891 passengers in this dataset, only **342 survived (38.4%)**. This project explores *who* survived and *why*, then builds a machine learning model to predict survival.

**Type:** Classification (predict a category: survived / did not survive)
**Final accuracy:** ~83%

---



## Dataset

The dataset (`titanic.csv`) contains **891 passengers** with the following columns:


| Column     | Description                                     |
| ---------- | ----------------------------------------------- |
| `survived` | **Target** — 0 = died, 1 = survived             |
| `pclass`   | Ticket class (1 = first, 2 = second, 3 = third) |
| `sex`      | male / female                                   |
| `age`      | Age in years                                    |
| `sibsp`    | Number of siblings / spouses aboard             |
| `parch`    | Number of parents / children aboard             |
| `fare`     | Ticket price                                    |
| `embarked` | Port of embarkation (S / C / Q)                 |


*(The raw file has a few extra columns like* `deck`*,* `alive`*,* `class` *— these are dropped or unused because they are mostly empty or duplicate other columns.)*

---



## Project Structure

```
titanic-project/
├── data/
│   └── titanic.csv
├── titanic.py          # main project script
└── README.md
```

---



## Setup & How to Run



### 1. Install the required libraries

```bash
pip install pandas scikit-learn matplotlib seaborn
```



### 2. Make sure the data is in place

Put `titanic.csv` inside a folder called `data/` (so the path is `data/titanic.csv`).

### 3. Run the project

```bash
python titanic.py
```

---



## The Phases

The project is split into four clear phases.

### PHASE 1 — Exploration (understand the data)

Before touching a model, we look at the data to find patterns.

```python
import pandas as pd

df = pd.read_csv('data/titanic.csv')

# How many survived vs died?
print(df['survived'].value_counts())

# Overall survival rate
print('Survival rate:', round(df['survived'].mean() * 100, 1), '%')

# Did gender matter?
print(df.groupby('sex')['survived'].mean())

# Did ticket class matter?
print(df.groupby('pclass')['survived'].mean())
```

**Key findings:**

- **549 died, 342 survived** (38.4% survival rate)
- **Women:** 74% survived — **Men:** only 19% survived → "women and children first" was real
- **1st class:** 63% survived — **3rd class:** only 24% survived → wealth mattered

---



### PHASE 2 — Visualization (see the patterns)

Charts make the patterns obvious.

```python
import matplotlib.pyplot as plt

# Survival by gender
gender_survival = df.groupby('sex')['survived'].mean()
plt.bar(gender_survival.index, gender_survival.values)
plt.title('Survival Rate by Gender')
plt.ylabel('Survival Rate')
plt.show()

# Survival by class
class_survival = df.groupby('pclass')['survived'].mean()
plt.bar(class_survival.index.astype(str), class_survival.values)
plt.title('Survival Rate by Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.show()

# Age distribution
plt.hist(df['age'].dropna(), bins=20)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Number of Passengers')
plt.show()
```

---



### PHASE 3 — Cleaning & Preparation

Real data is messy. We fix missing values and convert text to numbers (models only understand numbers).

```python
# Keep only the useful columns
df = df[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]

# Fill missing values
df['age'] = df['age'].fillna(df['age'].median())      # median age for missing ages
df['embarked'] = df['embarked'].fillna('S')            # most common port

# Convert text to numbers (models need numbers)
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})
```

**What we did and why:**

- Missing `age` → filled with the **median** age (safe middle value)
- Missing `embarked` → filled with `'S'` (the most common port)
- `sex` and `embarked` are text → converted to numbers with `.map()`, because ML models only work with numbers

---



### PHASE 4 — Model Training & Prediction

Now we train a model and measure how well it predicts survival.

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Features (X) and target (y)
X = df[['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]
y = df['survived']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy on unseen test data
print('Accuracy:', round(model.score(X_test, y_test), 3))

# Which features mattered most?
for feature, importance in sorted(
        zip(X.columns, model.feature_importances_),
        key=lambda item: -item[1]):
    print(feature, '->', round(importance * 100, 1), '%')
```

**Results:**

- **Accuracy: ~83%** on unseen test data
- **Most important features:** `fare` (27%), `sex` (27%), `age` (26%) — money, gender, and age decided survival the most

---



### BONUS — Predict for a new passenger

```python
import pandas as pd

# "Would I have survived?"
# Example: 3rd class, female, age 25, no family, fare 10, embarked S
new_passenger = pd.DataFrame({
    'pclass':   [3],
    'sex':      [1],    # 1 = female
    'age':      [25],
    'sibsp':    [0],
    'parch':    [0],
    'fare':     [10],
    'embarked': [0]     # 0 = Southampton
})

prediction = model.predict(new_passenger)
result = 'SURVIVED' if prediction[0] == 1 else 'DID NOT SURVIVE'
print('Prediction:', result)

# Confidence
proba = model.predict_proba(new_passenger)[0]
print('Chance of survival:', round(proba[1] * 100, 1), '%')
```

---



## Key Insights

- **Gender was the biggest social factor** — women survived at nearly 4x the rate of men.
- **Class mattered a lot** — 1st class passengers survived at 2.6x the rate of 3rd class.
- **Fare and age** were the strongest numeric predictors — a proxy for wealth and vulnerability.
- The "women and children first" protocol clearly shows up in the data.

---



## Tech Stack

- **Python 3**
- **pandas** — data loading, cleaning, analysis
- **matplotlib / seaborn** — visualization
- **scikit-learn** — machine learning (Random Forest)

---



## What I Learned

- Loading and cleaning a real, messy dataset (missing values, text-to-number conversion)
- Exploratory data analysis with `groupby` and charts
- Training a classification model and measuring accuracy on unseen data
- Interpreting feature importance to understand *why* the model makes its predictions

---



## Possible Improvements

- Try other models (Logistic Regression, Decision Tree) and compare accuracy
- Create new features (e.g. family size = sibsp + parch, or extract titles from names)
- Tune the model's settings for better accuracy
- Add cross-validation for a more reliable score

---

*This project is part of my machine learning learning journey — from NumPy fundamentals to real-world ML projects.*