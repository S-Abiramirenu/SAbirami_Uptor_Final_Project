# importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

# reading data into the dataframe
df = pd.read_csv('SAbirami_uptor_final_project.csv')

# displaying first five rows
print(df.head())

# shape of the dataframe
df.shape
print(df.shape)

# concise summary of dataframe
df.info()
print(df.info())

# checking for null values
df.isnull().sum()

# dropping 'Unnamed: 32' column.
df.drop("Unnamed: 32", axis=1, inplace=True)

# dropping id column
df.drop('id',axis=1, inplace=True)
print(df.info())

# descriptive statistics of data
df.describe()
print(df.describe())

label_encoding_object = LabelEncoder()
df['diagnosis'] = label_encoding_object.fit_transform(df['diagnosis'])
print(df)

# Counts of unique values
finding_unique = df['diagnosis'].value_counts()
print(finding_unique)

# One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['diagnosis'])
print(df_encoded)

from sklearn.model_selection import train_test_split

X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# splitting data
X_train, X_test, y_train, y_test = train_test_split(df.drop('diagnosis', axis=1), df['diagnosis'], test_size=0.2,random_state=42)

print("Shape of training set:", X_train.shape)
print("Shape of test set:", X_test.shape)

from sklearn.preprocessing import StandardScaler

Standard_scaler_object= StandardScaler()
X_train = Standard_scaler_object.fit_transform(X_train)
print(X_train)
X_test = Standard_scaler_object.fit_transform(X_test)
print(X_test)

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
predictions1 = logreg.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report

print("Confusion Matrix: \n", confusion_matrix(y_test, predictions1))
print('\n')
print(classification_report(y_test, predictions1))

from sklearn.metrics import accuracy_score

logreg_acc = accuracy_score(y_test, predictions1)
print("Accuracy of the Logistic Regression Model is: ", logreg_acc)

scores = cross_val_score(logreg, X_train, y_train, cv=5)
print("Cross-Validation Accuracy Scores:", scores)
print("Average Accuracy:", scores.mean())

# Applying KMeans
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)
cluster_labels = kmeans.labels_

# Compare cluster labels with actual diagnosis (just for reference)
comparison_df = pd.DataFrame({'Actual Diagnosis': y, 'Cluster': cluster_labels})
print("\nCluster vs Diagnosis Crosstab:")
print(pd.crosstab(comparison_df['Actual Diagnosis'], comparison_df['Cluster']))

# Optional: Evaluate clustering quality using silhouette score
sil_score = silhouette_score(X, cluster_labels)
print("Silhouette Score for KMeans Clustering:", sil_score)

# Reducing dimensions for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plotting Clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=cluster_labels, palette='Set2')
plt.title("KMeans Clustering Visualization (2D PCA Projection)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster")
plt.show()

df = df.dropna()  # Remove missing values
df = df.select_dtypes(include=['number'])  # Keep only numeric columns
print(df)

# countplot
plt.figure(figsize = (8,7))
sns.countplot(x="diagnosis", hue="diagnosis", data=df, palette='magma', legend=False)
plt.title("Diagnosis Distribution")
plt.xlabel("Diagnosis (0 = Benign, 1 = Malignant)")
plt.ylabel("Count")
plt.show()

# heatmap
plt.figure(figsize=(20,18))
sns.heatmap(df.corr(), annot=True,linewidths=.5, cmap="Purples")
plt.title("Feature Correlation Heatmap")
plt.show()





