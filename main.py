import streamlit as st
import numpy as np
import pandas as pd

#import plotly.express as px
#import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


def add_parameter_ui(class_name):
	params = dict()
	if class_name == "Logistic Regression":
		pass
	elif class_name == "SVM":
		C = st.sidebar.slider("C",0.01,10.0)
		params["C"] = C
	elif class_name == "Random Forest":
		max_depth = st.sidebar.slider("Max_Depth", 2, 100)
		n_estimators = st.sidebar.slider("N-Estimators", 1, 1000)
		params["max_depth"] = max_depth
		params["n_estimators"] = n_estimators
	elif class_name == "Decision Tree":
		max_depth = st.sidebar.slider("Max_Depth", 2, 100)
		params["max_depth"] = max_depth
	return params


def get_classifier(class_name, params):
	if class_name == "Logistic Regression":
		classifier = LogisticRegression()
	elif class_name == "SVM":
		classifier = SVC(C=params["C"])
	elif class_name=="Random Forest":
		classifier = RandomForestClassifier(n_estimators=params["n_estimators"],max_depth=params["max_depth"], random_state=0)
	elif class_name == "Decision Tree":
		classifier = DecisionTreeClassifier(max_depth=params["max_depth"])
	return classifier


# Title
st.title('Read Giant Branch (RGB) vs Helium Burning (HeB)')

df = pd.read_csv("classification_in_asteroseismology.csv")
x = df.iloc[:, 1:].values
y = df.iloc[:, 0].values


classifier_name = st.sidebar.selectbox("Select the classifier", ("SVM", "Random Forest", "Logistic Regression", "Decision Tree"))
params = add_parameter_ui(classifier_name)

# Get data discription
st.subheader("First 5 Entries of Data")
st.write(df.head())

st.subheader("Describtion of Data")
st.write(df.describe())

st.subheader("Information of Data")
import io
buffer = io.StringIO()
df.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

st.subheader('Explore the original data')
st.write('Shape of Dataset', x.shape)
st.write('Number of classes', len(np.unique(y)))

st.subheader("Correaltion Matrix of the data")
corr = df.corr()
fig_size_val = 3
fig = plt.figure(figsize=(fig_size_val, fig_size_val))
sns.heatmap(corr, xticklabels=corr.columns.values,yticklabels=corr.columns.values, annot=True)
st.pyplot(fig)

st.subheader("Pairplot")
st.pyplot(sns.pairplot(df, hue="POP", height=3))

st.title("Classification using {}".format(classifier_name))
classifier = get_classifier(classifier_name, params)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

st.write(f"Classifier = {classifier_name}")
conf_mat = confusion_matrix(y_test, y_pred)
st.write('Confusion matrix: ', conf_mat)
st.write('Classification Report: ', classification_report(y_test, y_pred))
