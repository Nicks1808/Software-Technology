import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, adjusted_rand_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# Τίτλος της εφαρμογής
st.title("Web-based Εφαρμογή για Tabular Data με Οπτικοποιήσεις και Αλγορίθμους Μηχανικής Μάθησης")


info = """
    Η εφαρμογή αυτή είναι μια web-based εφαρμογή που επιτρέπει την ανάλυση και οπτικοποίηση δεδομένων πίνακα (tabular data) 
    με τη χρήση διαφόρων αλγορίθμων μηχανικής μάθησης.

    Τρόπος Λειτουργίας:
    1. Ο χρήστης φορτώνει ένα αρχείο CSV ή Excel με τα δεδομένα που θέλει να αναλύσει.
    2. Τα δεδομένα εμφανίζονται στην οθόνη, δίνοντας μια πρώτη ματιά στις πρώτες γραμμές και στα στατιστικά περιγραφικά τους.
    3. Ο χρήστης μπορεί να επιλέξει έναν αλγόριθμο μείωσης διαστάσεων (PCA ή t-SNE) για να οπτικοποιήσει τα δεδομένα σε 2 διαστάσεις.
    4. Παρέχονται επίσης δυνατότητες για εξερεύνηση των δεδομένων μέσω διαγραμμάτων (ιστόγραμμα, scatter plot, box plot).
    5. Η εφαρμογή υποστηρίζει κατηγοριοποίηση χρησιμοποιώντας K-Nearest Neighbors (K-NN) και Δέντρο Αποφάσεων.
    6. Επίσης υποστηρίζει ομαδοποίηση χρησιμοποιώντας K-Means και Gaussian Mixture Model (GMM).
    7. Τα αποτελέσματα των αλγορίθμων εμφανίζονται στην οθόνη, επιτρέποντας στον χρήστη να συγκρίνει την απόδοσή τους.

    
    Για την εγγραφή αυτού του κώδικα δεν υπήρξε ομάδα. Την έγραψα μόνος μου 
    Σορτικός Νίκος (inf2021207).
    """

st.sidebar.markdown(info)


# Επιλογή αρχείου
uploaded_file = st.file_uploader("Φορτώστε το αρχείο σας", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Ανάγνωση δεδομένων από το αρχείο
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Εμφάνιση των πρώτων γραμμών του πίνακα
    st.write("Πρώτες γραμμές των δεδομένων:")
    st.write(df.head())

    # Διαχείριση της τελευταίας στήλης ως ετικέτα εξόδου
    st.write("Ετικέτα εξόδου (τελευταία στήλη):")
    st.write(df.iloc[:, -1])

    # Στατιστικά περιγραφικά του πίνακα
    st.write("Στατιστικά περιγραφικά των δεδομένων:")
    st.write(df.describe())

    # Εμφάνιση ολόκληρου του πίνακα
    st.write("Ολόκληρος ο πίνακας δεδομένων:")
    st.dataframe(df)

    # Επιλογή αλγορίθμου μείωσης διαστάσεων
    dim_reduction_algo = st.selectbox("Επιλέξτε αλγόριθμο μείωσης διαστάσεων", ("PCA", "t-SNE"))

    # Εξαγωγή χαρακτηριστικών και ετικετών
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Εφαρμογή επιλεγμένου αλγορίθμου μείωσης διαστάσεων
    if dim_reduction_algo == "PCA":
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)
    elif dim_reduction_algo == "t-SNE":
        tsne = TSNE(n_components=2)
        X_reduced = tsne.fit_transform(X)

    # Οπτικοποίηση των αποτελεσμάτων
    fig, ax = plt.subplots()
    scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis')
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.title(f"2D Visualization using {dim_reduction_algo}")
    st.pyplot(fig)

    # Exploratory Data Analysis (EDA) Diagrams
    st.write("Exploratory Data Analysis (EDA) Diagrams")

    # Histogram of a selected column
    selected_column = st.selectbox("Επιλέξτε στήλη για ιστόγραμμα", df.columns[:-1].tolist())
    fig, ax = plt.subplots()
    df[selected_column].hist(ax=ax, bins=30)
    plt.xlabel(selected_column)
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {selected_column}")
    st.pyplot(fig)

    # Scatter plot between two selected columns
    selected_columns = st.multiselect(
        "Επιλέξτε δύο στήλες για scatter plot", 
        df.columns[:-1].tolist(), 
        default=df.columns[:2].tolist()
    )
    if len(selected_columns) == 2:
        fig, ax = plt.subplots()
        ax.scatter(df[selected_columns[0]], df[selected_columns[1]], c=y, cmap='viridis')
        plt.xlabel(selected_columns[0])
        plt.ylabel(selected_columns[1])
        plt.title(f"Scatter Plot between {selected_columns[0]} and {selected_columns[1]}")
        st.pyplot(fig)

    # Box plot of a selected column
    selected_column_box = st.selectbox("Επιλέξτε στήλη για box plot", df.columns[:-1].tolist())
    fig, ax = plt.subplots()
    df.boxplot(column=selected_column_box, ax=ax)
    plt.title(f"Box Plot of {selected_column_box}")
    st.pyplot(fig)

    # Διαχωρισμός των καρτελών για κατηγοριοποίηση και ομαδοποίηση
    tab1, tab2 = st.tabs(["Κατηγοριοποίηση", "Ομαδοποίηση"])

    with tab1:
        st.header("Κατηγοριοποίηση")
        
        # Επιλογή αλγορίθμων κατηγοριοποίησης
        k_neighbors = st.slider("Αριθμός γειτόνων για K-NN", 1, 15, 5)
        max_depth = st.slider("Μέγιστο βάθος για Δέντρο Αποφάσεων", 1, 15, 5)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Εκπαίδευση και αξιολόγηση K-NN
        knn = KNeighborsClassifier(n_neighbors=k_neighbors)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        accuracy_knn = accuracy_score(y_test, y_pred_knn)
        report_knn = classification_report(y_test, y_pred_knn)
        matrix_knn = confusion_matrix(y_test, y_pred_knn)
        
        # Εκπαίδευση και αξιολόγηση Δέντρου Αποφάσεων
        tree = DecisionTreeClassifier(max_depth=max_depth)
        tree.fit(X_train, y_train)
        y_pred_tree = tree.predict(X_test)
        accuracy_tree = accuracy_score(y_test, y_pred_tree)
        report_tree = classification_report(y_test, y_pred_tree)
        matrix_tree = confusion_matrix(y_test, y_pred_tree)
        
        st.write(f"K-NN Ακρίβεια: {accuracy_knn:.2f}")
        st.text("Classification Report για K-NN:")
        st.text(report_knn)
        st.text("Confusion Matrix για K-NN:")
        st.write(matrix_knn)
        
        st.write(f"Δέντρο Αποφάσεων Ακρίβεια: {accuracy_tree:.2f}")
        st.text("Classification Report για Δέντρο Αποφάσεων:")
        st.text(report_tree)
        st.text("Confusion Matrix για Δέντρο Αποφάσεων:")
        st.write(matrix_tree)
        
        if accuracy_knn > accuracy_tree:
            st.write("Ο K-NN έχει καλύτερη απόδοση.")
        else:
            st.write("Το Δέντρο Αποφάσεων έχει καλύτερη απόδοση.")

    with tab2:
        st.header("Ομαδοποίηση")
        
        # Επιλογή αλγορίθμων ομαδοποίησης
        k_clusters = st.slider("Αριθμός συστάδων για K-Means", 2, 10, 3)
        n_components_gmm = st.slider("Αριθμός συστατικών για GMM", 2, 10, 3)
        
        # Εκπαίδευση και αξιολόγηση K-Means
        kmeans = KMeans(n_clusters=k_clusters, random_state=42)
        y_pred_kmeans = kmeans.fit_predict(X)
        score_kmeans = adjusted_rand_score(y, y_pred_kmeans)
        
        # Εκπαίδευση και αξιολόγηση GMM
        gmm = GaussianMixture(n_components=n_components_gmm, random_state=42)
        y_pred_gmm = gmm.fit_predict(X)
        score_gmm = adjusted_rand_score(y, y_pred_gmm)
        
        st.write(f"K-Means Adjusted Rand Index: {score_kmeans:.2f}")
        st.write(f"GMM Adjusted Rand Index: {score_gmm:.2f}")

        if score_kmeans > score_gmm:
            st.write("Ο K-Means έχει καλύτερη απόδοση.")
        else:
            st.write("Ο GMM έχει καλύτερη απόδοση.")

else:
    st.write("Παρακαλώ φορτώστε ένα αρχείο CSV ή Excel για να δείτε τα δεδομένα σας.")



