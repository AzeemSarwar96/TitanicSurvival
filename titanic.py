import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('data/titanic.csv')
# print(df.head())
# print(df.info())

# How many survived vs died?
print("Survival counts:", df['survived'].value_counts())

# Overall survival rate
print('Survival rate:', round(df['survived'].mean() * 100, 1), "%")

# Did gender matter?
print("Gender survival rate:", round(df.groupby('sex')['survived'].mean() * 100, 1))

# Did ticket class matter?
print("Ticket class survival rate:", round(df.groupby('pclass')['survived'].mean() * 100, 1))


# Survival by gender
gender_survival = df.groupby('sex')['survived'].mean()
print(gender_survival.index)
print(gender_survival.values)
plt.bar(gender_survival.index, gender_survival.values * 100)
plt.xlabel('Sex')
plt.ylabel('Survival Rate (%)')
plt.title('Survival by Gender')
plt.show()

# Survival by class
class_survival = df.groupby('pclass')['survived'].mean()
print(class_survival.index)
print(class_survival.values)
plt.bar(class_survival.index, class_survival.values * 100)
plt.xlabel('Class')
plt.ylabel('Survival Rate (%)')
plt.title('Survival by Class')
plt.show()

# Age distribution
plt.hist(df['age'], bins=20)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')
plt.show()

# Keep only the useful columns
df = df[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]
# Fill missing values
df['age'] = df['age'].fillna(df['age'].median())      # median age for missing ages
df['embarked'] = df['embarked'].fillna('S') 

# Convert text to numbers (models need numbers)
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})

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