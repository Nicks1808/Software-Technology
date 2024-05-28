import streamlit as st
import pandas as pd

st.title("Web-based Data Analysis Application")

uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=['csv', 'xlsx'])
if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.write("Data Loaded Successfully")
    st.dataframe(df)
else:
    st.write("Please upload a file.")


if uploaded_file:
    st.write("Data Preview:")
    st.dataframe(df.head())
    st.write(f"Number of Samples: {df.shape[0]}")
    st.write(f"Number of Features: {df.shape[1] - 1}")
    st.write(f"Labels: {df.iloc[:, -1].unique()}")


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

tab1, tab2 = st.tabs(["2D Visualization", "EDA"])

with tab1:
    st.subheader("2D Visualizations")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df.iloc[:, :-1])
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_result = tsne.fit_transform(df.iloc[:, :-1])

    st.write("PCA")
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=df.iloc[:, -1])
    st.pyplot(plt)

    st.write("t-SNE")
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=df.iloc[:, -1])
    st.pyplot(plt)

with tab2:
    st.subheader("Exploratory Data Analysis")
    st.write("Pairplot")
    sns.pairplot(df, hue=df.columns[-1])
    st.pyplot(plt)

    st.write("Heatmap")
    plt.figure(figsize=(10, 5))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    st.pyplot(plt)


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

tab1, tab2 = st.tabs(["2D Visualization", "EDA"])

with tab1:
    st.subheader("2D Visualizations")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df.iloc[:, :-1])
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    tsne_result = tsne.fit_transform(df.iloc[:, :-1])

    st.write("PCA")
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=df.iloc[:, -1])
    st.pyplot(plt)

    st.write("t-SNE")
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=df.iloc[:, -1])
    st.pyplot(plt)

with tab2:
    st.subheader("Exploratory Data Analysis")
    st.write("Pairplot")
    sns.pairplot(df, hue=df.columns[-1])
    st.pyplot(plt)

    st.write("Heatmap")
    plt.figure(figsize=(10, 5))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    st.pyplot(plt)



if uploaded_file:
    st.write("Algorithm Performance Metrics")
    st.write(f"KNN Accuracy: {accuracy_score(y_test, y_pred_knn)}")
    st.write(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred_dt)}")



st.sidebar.header("About")
st.sidebar.write("This application was developed to analyze and visualize tabular data using machine learning algorithms.")
st.sidebar.write("Team Members: [Your Team Members]")




