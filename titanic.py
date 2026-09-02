import pandas as pd
import matplotlib.pyplot as plt

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