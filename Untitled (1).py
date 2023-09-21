#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Define the dataset as a list of dictionaries
data = [
    {"age": "<=30", "income": "high", "student": "no", "credit_rating": "fair", "buys_computer": "no"},
    {"age": "<=30", "income": "high", "student": "no", "credit_rating": "excellent", "buys_computer": "no"},
    {"age": "31...40", "income": "high", "student": "no", "credit_rating": "fair", "buys_computer": "yes"},
    {"age": ">40", "income": "medium", "student": "no", "credit_rating": "fair", "buys_computer": "yes"},
    {"age": ">40", "income": "low", "student": "yes", "credit_rating": "fair", "buys_computer": "yes"},
    {"age": ">40", "income": "low", "student": "yes", "credit_rating": "excellent", "buys_computer": "no"},
    {"age": "31...40", "income": "low", "student": "yes", "credit_rating": "excellent", "buys_computer": "yes"},
    {"age": "<=30", "income": "medium", "student": "no", "credit_rating": "fair", "buys_computer": "no"},
    {"age": "<=30", "income": "low", "student": "yes", "credit_rating": "fair", "buys_computer": "yes"},
    {"age": ">40", "income": "medium", "student": "yes", "credit_rating": "fair", "buys_computer": "yes"},
    {"age": "<=30", "income": "medium", "student": "yes", "credit_rating": "excellent", "buys_computer": "yes"},
    {"age": "31...40", "income": "medium", "student": "no", "credit_rating": "excellent", "buys_computer": "yes"},
    {"age": "31...40", "income": "high", "student": "yes", "credit_rating": "fair", "buys_computer": "yes"},
    {"age": ">40", "income": "medium", "student": "no", "credit_rating": "excellent", "buys_computer": "yes"},
    {"age": "31...40", "income": "high", "student": "no", "credit_rating": "excellent", "buys_computer": "no"},
]

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Print the DataFrame
print(df)


# In[3]:


import pandas as pd
from math import log2
# Define the dataset
data = pd.DataFrame({
    'age': ['<=30', '<=30', '31-40', '>40', '>40', '>40', '31-40', '<=30', '<=30', '>40', '<=30', '31-40', '31-40', '>40'],
    'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'],
    'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no'],
    'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair'],
    'buys_computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
})

# Calculate the entropy for the 'buys_computer' feature

def calculate_entropy(data):
    class_counts = data.value_counts(normalize=True)
    entropy = -sum( f* log2(f) for f in class_counts)
    return entropy

entropy_buys_computer = calculate_entropy(data['buys_computer'])
print(f'Entropy of "buys_computer": {entropy_buys_computer:.4f}')


# In[5]:


# Calculate the entropy for each feature
feature_entropies = {}
for feature in data.columns[:-1]: 
    unique_values = data[feature].unique()
    weighted_entropy = 0
    for value in unique_values:
        subset = data[data[feature] == value]
        entropy = calculate_entropy(subset['buys_computer'])
        weight = len(subset) / len(data)
        weighted_entropy += weight * entropy
    feature_entropies[feature] = weighted_entropy
    
    
    
    # Print the entropy for each feature
print('\nEntropy for each feature at the root node:')
for feature, entropy in feature_entropies.items():
    print(f'{feature}: {entropy:.4f}')


# In[6]:


import pandas as pd
import math

# Load the dataset
data = {
    'age': ['<=30', '<=30', '31…40', '>40', '>40', '>40', '31…40', '<=30', '<=30', '>40', '<=30', '31…40', '31…40', '>40'],
    'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'],
    'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],
    'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'excellent'],
    'buys_computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
}

df = pd.DataFrame(data)

# Calculate the entropy of the target variable 'buys_computer' at the root node
total_samples = len(df)
entropy_root = 0

for value in df['buys_computer'].unique():
    value_count = len(df[df['buys_computer'] == value])
    if value_count > 0:
        p_value = value_count / total_samples
        entropy_root -= p_value * math.log2(p_value)

print(f'Entropy at the root node: {entropy_root:.4f}')


# In[7]:


import pandas as pd
import math

# Load the dataset
data = {
    'age': ['<=30', '<=30', '31-40', '>40', '>40', '>40', '31-40', '<=30', '<=30', '>40', '<=30', '31-40', '31-40', '>40'],
    'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'],
    'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],
    'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'excellent'],
    'buys_computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
}

df = pd.DataFrame(data)

# Calculate the entropy of the target variable 'buys_computer' at the root node
total_samples = len(df)
entropy_root = 0

for value in df['buys_computer'].unique():
    value_count = len(df[df['buys_computer'] == value])
    if value_count > 0:
        p_value = value_count / total_samples
        entropy_root -= p_value * math.log2(p_value)

information_gain = {}
for feature in df.columns[:-1]:  # Exclude the target variable
    entropy_feature = 0
    attribute_counts = df.groupby([feature, 'buys_computer']).size().unstack(fill_value=0)
    
    for value in attribute_counts.index:
        p_value = sum(attribute_counts.loc[value]) / total_samples
        for class_value in attribute_counts.columns:
            p_class = attribute_counts.loc[value, class_value] / (sum(attribute_counts.loc[value]) + 1e-10)  # Add a small epsilon to avoid division by zero
            if p_class > 0:
                entropy_feature -= p_value * p_class * math.log2(p_class)
    
    information_gain[feature] = entropy_root - entropy_feature

print("Information Gain for each feature:")
for feature, gain in information_gain.items():
    print(f'{feature}: {gain:.4f}')


# In[8]:


# Calculate information gain for each feature as previously shown

# Identify the feature with the highest Information Gain as the root node
root_node = max(information_gain, key=information_gain.get)

print("Entropy at the root node:", entropy_root)

print("\nInformation Gain for each feature:")
for feature, ig in information_gain.items():
    print(f"{feature}: {ig:.4f}")

print("The first feature to select for the decision tree:", root_node)


# In[9]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
data = {
    'age': ['<=30', '<=30', '31-40', '>40', '>40', '>40', '31-40', '<=30', '<=30', '>40', '<=30', '31-40', '31-40', '>40'],
    'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'],
    'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],
    'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'excellent'],
    'buys_computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
}

df = pd.DataFrame(data)

# Separate the features (X) and the target variable (y)
X = df.drop('buys_computer', axis=1)
y = df['buys_computer']

# Perform one-hot encoding on the categorical features
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X)

# Create and train the Decision Tree model
model = DecisionTreeClassifier()
model.fit(X_encoded, y)

# Get the training set accuracy
training_accuracy = model.score(X_encoded, y)
print("Training Set Accuracy:", training_accuracy)

# Get the depth of the constructed tree
tree_depth = model.get_depth()
print("Tree Depth:", tree_depth)


# In[10]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import tree

# Define the dataset
data = {
    'age': ['<=30', '<=30', '31-40', '>40', '>40', '>40', '31-40', '<=30', '<=30', '>40', '<=30', '31-40', '31-40', '>40'],
    'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'],
    'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],
    'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'excellent'],
    'buys_computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
}

df = pd.DataFrame(data)

# Encode categorical features
le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])

# Define features and target variable
X = df.drop('buys_computer', axis=1)
y = df['buys_computer']

# Create and fit the decision tree model
model = DecisionTreeClassifier()
model.fit(X, y)

# Visualize the decision tree
plt.figure(figsize=(20, 10))
tree.plot_tree(model, filled=True, feature_names=list(df.columns[:-1]), class_names=['No', 'Yes'], rounded=True, fontsize=12)
plt.show()


# In[ ]:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#loading the project data
df = pd.read_excel("embeddingsdatasheet-1.xlsx")
df
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


# Separate features (X) and target (y)
X = df[['embed_1', 'embed_2']]
y = df['Label']

# Split the data into training and test sets (70% training, 30% test)
Tr_X, Te_X, Tr_y, Te_y = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a Decision Tree classifier
model = DecisionTreeClassifier()

# Fit the model on the training data
model.fit(Tr_X, Tr_y)

# Training Set accuracy
train_accuracy = model.score(Tr_X, Tr_y)

# Test Set Accuracy
test_accuracy = model.score(Te_X, Te_y)

print(f"Training Set Accuracy: {train_accuracy}")
print(f"Test Set Accuracy: {test_accuracy}")
class_names = df['Label'].unique().astype(str).tolist()
plt.figure(figsize=(10, 6))
plot_tree(model, filled=True, feature_names=['embed_1', 'embed_2'], class_names=class_names)
plt.title("Decision Tree")
plt.show()
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Create a Decision Tree classifier with max_depth constraint
model = DecisionTreeClassifier(max_depth=5)

# Fit the model on the training data
binary_df = df[df['Label'].isin([0, 1])]
X = binary_df[['embed_1', 'embed_2']]
y = binary_df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Fit the model on the training data
model.fit(X_train, y_train)

# Calculate training set accuracy
train_accuracy = model.score(X_train, y_train)

# Calculate test set accuracy
test_accuracy = model.score(X_test, y_test)

# Print the accuracies
print(f"Training Set Accuracy (max_depth=5): {train_accuracy}")
print(f"Test Set Accuracy (max_depth=5): {test_accuracy}")

# Visualize the tree with max_depth constraint
plt.figure(figsize=(20, 10))
plot_tree(model, filled=True, feature_names=['embed_0', 'embed_1'], class_names=['no', 'yes'], rounded=True)
plt.show()




