
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics, tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount','Loan_Amount_Term']
def remove_outliers(df, cols):

    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

df=pd.read_csv('loan_eligibility.csv')


df.drop(columns=['Loan_ID','Gender','Married'], inplace=True)


df['Self_Employed'].fillna('Yes', inplace=True)
df['Dependents'].fillna('3+', inplace=True)


df['CoapplicantIncome'].interpolate(inplace=True)
df['LoanAmount'].interpolate(inplace=True)
df['Loan_Amount_Term'].interpolate(inplace=True)
df['Credit_History'].interpolate(inplace=True)


df.dropna(axis=0,how='any', inplace=True)



encoder = LabelEncoder()
for col in [ 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']:
    df[col] = encoder.fit_transform(df[col])

X = df.drop(columns=['Loan_Status'])
y = df['Loan_Status']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)

model = DecisionTreeClassifier(criterion='entropy')
model=model.fit(x_train, y_train)


print('Accuracy: {:0.4f}'. format(model.score(x_test, y_test)*100))
y_pred = model.predict(x_test)
print('Accuracy: {:0.4f}'. format(metrics.accuracy_score(y_test, y_pred)*100))

report = classification_report(y_test, y_pred)
print(report)

text_representation = tree.export_text(model)
print(text_representation)

fig = plt.figure(figsize=(15,10))
class_name = [str(cls) for cls in model.classes_]
tree.plot_tree(model,feature_names = df.columns[:-1],class_names = class_name,filled = True,rounded = True)

plt.show()