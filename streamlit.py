import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


st.title("Web-based Εφαρμογή για Tabular Data με Οπτικοποιήσεις και Αλγορίθμους Μηχανικής Μάθησης")


uploaded_file = st.file_uploader("Φορτώστε το αρχείο σας", type=["csv", "xlsx"])

if uploaded_file is not None:
   
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

  
    st.write("Πρώτες γραμμές των δεδομένων:")
    st.write(df.head())

  
    st.write("Τελευταία στήλη:")
    st.write(df.iloc[:, -1])

    
    st.write("Στατιστικά περιγραφικά των δεδομένων:")
    st.write(df.describe())

   
    st.write("Ολόκληρος ο πίνακας δεδομένων:")
    st.dataframe(df)

   
    dim_reduction_algo = st.selectbox("Επιλέξτε αλγόριθμο μείωσης διαστάσεων", ("PCA", "t-SNE"))

   
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    
    if dim_reduction_algo == "PCA":
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)
    elif dim_reduction_algo == "t-SNE":
        tsne = TSNE(n_components=2)
        X_reduced = tsne.fit_transform(X)

   
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis')
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title(f"2D Visualization using {dim_reduction_algo}")
    st.pyplot(fig)

   
    st.write("Exploratory Data Analysis (EDA) Diagrams")

    # Histogram of a selected column
    selected_column = st.selectbox("Επιλέξτε στήλη για ιστόγραμμα", df.columns[:-1])
    fig, ax = plt.subplots()
    df[selected_column].hist(ax=ax, bins=30)
    plt.xlabel(selected_column)
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {selected_column}")
    st.pyplot(fig)

   
    selected_columns = st.multiselect("Επιλέξτε δύο στήλες για scatter plot", df.columns[:-1], default=df.columns[:2])
    if len(selected_columns) == 2:
        fig, ax = plt.subplots()
        ax.scatter(df[selected_columns[0]], df[selected_columns[1]], c=y, cmap='viridis')
        plt.xlabel(selected_columns[0])
        plt.ylabel(selected_columns[1])
        plt.title(f"Scatter Plot between {selected_columns[0]} and {selected_columns[1]}")
        st.pyplot(fig)

    
    selected_column_box = st.selectbox("Επιλέξτε στήλη για box plot", df.columns[:-1])
    fig, ax = plt.subplots()
    df.boxplot(column=selected_column_box, ax=ax)
    plt.title(f"Box Plot of {selected_column_box}")
    st.pyplot(fig)

    
    tab1, tab2 = st.tabs(["Κατηγοριοποίηση", "Ομαδοποίηση"])

    with tab1:
        st.header("Κατηγοριοποίηση")
        
      
        k_neighbors = st.slider("Αριθμός γειτόνων για K-NN", 1, 15, 5)
        max_depth = st.slider("Μέγιστο βάθος για Δέντρο Αποφάσεων", 1, 15, 5)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        
        knn = KNeighborsClassifier(n_neighbors=k_neighbors)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        accuracy_knn = accuracy_score(y_test, y_pred_knn)
        
        
        tree = DecisionTreeClassifier(max_depth=max_depth)
        tree.fit(X_train, y_train)
        y_pred_tree = tree.predict(X_test)
        accuracy_tree = accuracy_score(y_test, y_pred_tree)
        
        st.write(f"K-NN Ακρίβεια: {accuracy_knn:.2f}")
        st.write(f"Δέντρο Αποφάσεων Ακρίβεια: {accuracy_tree:.2f}")

    with tab2:
        st.header("Ομαδοποίηση")
        
       
        k_clusters = st.slider("Αριθμός συστάδων για K-Means", 2, 10, 3)
        n_components_gmm = st.slider("Αριθμός συστατικών για GMM", 2, 10, 3)
        
      
        kmeans = KMeans(n_clusters=k_clusters, random_state=42)
        y_pred_kmeans = kmeans.fit_predict(X)
        score_kmeans = adjusted_rand_score(y, y_pred_kmeans)
        
        
        gmm = GaussianMixture(n_components=n_components_gmm, random_state=42)
        y_pred_gmm = gmm.fit_predict(X)
        score_gmm = adjusted_rand_score(y, y_pred_gmm)
        
        st.write(f"K-Means Adjusted Rand Index: {score_kmeans:.2f}")
        st.write(f"GMM Adjusted Rand Index: {score_gmm:.2f}")

else:
    st.write("Φορτώστε ένα αρχείο CSV ή Excel για να δείτε τα δεδομένα σας.")
