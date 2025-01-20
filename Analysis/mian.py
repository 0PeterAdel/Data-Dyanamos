import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl


df = pd.read_excel('Data.xlsx', sheet_name='Employee')
print(df.head())
print(df.isnull().sum())
print(df.info())

# Is there a correlation between overtime work and performance or attrition?
sns.countplot(x='OverTime', hue='Attrition', data=df)
plt.title('Overtime vs Attrition')
plt.show()

sns.boxplot(x='OverTime', y='Salary', data=df)
plt.title('Overtime vs Salary')
plt.show()


#How does business travel affect employee satisfaction and retention?
sns.countplot(x='BusinessTravel', hue='Attrition', data=df)
plt.title('Business Travel vs Attrition')
plt.show()

