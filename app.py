import streamlit as st
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanorama
import decoupler as dc
import os
import socket
import webbrowser
from io import StringIO
import anndata as ad
from datetime import datetime

st.set_page_config(page_title="scRNA-seq App", layout="wide")

st.markdown("""
<style>
html, body, .stApp {
    background-color: #2b2b2b !important;
    color: white !important;
}

/* Sidebar container */
section[data-testid="stSidebar"] {
    background-color: #1f1f1f !important;
    padding: 1rem 0.75rem 0.75rem 0.75rem;
    border-right: 1px solid #444;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: 100vh;
}

/* Sidebar title */
.sidebar-title {
    font-size: 18px;
    font-weight: bold;
    color: white;
    margin-bottom: 0.5rem;
    padding-left: 0.5rem;
}

/* Radio button group */
div[role="radiogroup"] {
    display: flex;
    flex-direction: column;
    gap: 0.1rem;
}

/* Each tab label (unselected) */
div[role="radiogroup"] > label {
    background-color: transparent !important;
    padding: 0.4rem 0.7rem;
    border-radius: 6px;
    font-size: 14px;
    color: white !important;
    display: flex;
    align-items: center;
    transition: background-color 0.2s ease;
}

/* Hover effect */
div[role="radiogroup"] > label:hover {
    background-color: #2b2b2b !important;
    color: black !important;
}

/* Selected tab */
div[role="radiogroup"] > label[data-selected="true"] {
    background-color: #ccc !important;
    color: black !important;
    font-weight: bold;
    text-decoration: underline;
}

/* Hide circle radio buttons */
div[role="radiogroup"] > label > div:first-child {
    display: none;
}
</style>
""", unsafe_allow_html=True)

pages = [
    "ℹ️ Πληροφορίες",
    "📥Μετατροπή CSV/TXT σε h5ad",
    "⚙️Προκαταρκτική Επεξεργασία",
    "🧬Ενοποίηση Δεδομένων",
    "🔗Scanorama Integration",
    "🔎Ανάθεση Κυτταρικών Τύπων",
    "📊DEG Ανάλυση",
    "🌋Volcano Plot",
    "🎯Εκφράσεις Γονιδίων",
    "📄Δήλωση"
]

st.sidebar.markdown('<div class="sidebar-title">🧬 Εργαλεία Ανάλυσης</div>', unsafe_allow_html=True)
page = st.sidebar.radio("Μενού", pages, label_visibility="collapsed")

if page == "ℹ️ Πληροφορίες":
    st.title("ℹ️ Πληροφορίες για την Εφαρμογή")
    st.write("""
             
    Η εφαρμογή αναπτύχθηκε με σκοπό την ανάλυση δεδομένων scRNA-seq, η οποία βασίζεται στη βιβλιοθήκη scanpy για Python.
    Παρέχει δυνατότητες όπως η ενοποίηση δεδομένων, η ανάλυση διαφορικής έκφρασης (DEG), και η οπτικοποίηση των αποτελεσμάτων, οι οποίες εξηγούνται παρακάτω.
    
 
    **Πληροφορίες Λειτουργιών:**

    **Μετατροπή CSV σε h5ad**: Εδώ μπορείτε να ανεβάσετε αρχεία CSV και να τα μετατρέψετε σε μορφή h5ad, η οποία είναι κατάλληλη για περαιτέρω ανάλυση με τη βιβλιοθήκη Scanpy.

    **Προκαταρκτική Επεξεργασία**: Σε αυτό το βήμα, η εφαρμογή εκτελεί διαδικασίες όπως φιλτράρισμα, κανονικοποίηση, PCA και clustering (Louvain) στα δεδομένα.

    **Ενοποίηση Δεδομένων**: Αυτή η λειτουργία συγχωνεύει πολλά αρχεία h5ad σε ένα ενιαίο dataset για ανάλυση.

    **Scanorama Integration**: Η εφαρμογή χρησιμοποιεί την τεχνική Scanorama για την ενοποίηση πολλών αρχείων h5ad με τη βοήθεια του Scanorama, που είναι ένα εργαλείο για την ενοποίηση δεδομένων scRNA-seq.

    **Ανάθεση Κυτταρικών Τύπων**: Αυτή η λειτουργία εκτελεί την ανάθεση κυτταρικών τύπων χρησιμοποιώντας το μοντέλο MLM (Markov Logic Networks) με βάση τα δεδομένα transcription factor.

    **DEG Ανάλυση**: Εδώ πραγματοποιείται η ανάλυση διαφορικής γονιδιακής έκφρασης (DEG) για να εντοπιστούν οι γονίδιοι που εμφανίζουν διαφορές ανάμεσα στις ομάδες κυττάρων.

    **Volcano Plot**: Δημιουργείται ένα volcano plot για την απεικόνιση των αποτελεσμάτων της DEG ανάλυσης, δείχνοντας τη σχέση fold-change vs. -log10(p-value).

    **Εκφράσεις Γονιδίων**: Εδώ μπορείτε να οπτικοποιήσετε την έκφραση ενός γονιδίου σε διαφορετικές ομάδες κυττάρων, χρησιμοποιώντας γραφήματα όπως το violin plot.
    """)

elif page == "📥Μετατροπή CSV/TXT σε h5ad":
    st.title("📥Μετατροπή CSV/TXT σε AnnData (.h5ad)")
    uploaded_file = st.file_uploader("📤 Ανέβασε το αρχείο CSV", type=["csv", "txt"])
  #Dropdown για επιλογή διαχωριστικού
    delimiter_options = {
        ",": "Κόμμα ( , )",
        ";": "Ερωτηματικό ( ; )",
        "	": "Tab",
        "|": "Pipe ( | )",
        ":": "Άνω-Κάτω Τελεία ( : )",
        "/": "Κάθετος ( / )",
        " ": "Κενό (space)",
        "--": "Διπλή Παύλα ( -- )"
    }


    delimiter_key = st.selectbox("Επιλέξτε διαχωριστικό:", options=list(delimiter_options.keys()),
                                 format_func=lambda x: delimiter_options[x])
    delimiter = delimiter_key.encode().decode("unicode_escape")

    if uploaded_file:
        try:
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8")) #Διαβαζει το CSV/TXT και το φορτώνει σε DataFrame
            df = pd.read_csv(stringio, delimiter=delimiter)

            st.write("📊Δείγμα Δεδομένων:")
            st.dataframe(df.head())

            if st.button("Μετατροπή σε AnnData (.h5ad)"): #Κουμπί μετατροπής
                try:
                    df.set_index(df.columns[0], inplace=True)

                    adata = ad.AnnData(X=df.values)  #Μετατροπή
                    adata.obs_names = df.index.astype(str)
                    adata.var_names = df.columns.astype(str)

                    export_filename = f"{datetime.now().strftime('%Y-%m-%dT%H-%M')}_converted.h5ad" # Αποθήκευση του αρχείου
                    adata.write_h5ad(export_filename)

                    with open(export_filename, "rb") as f:
                        st.download_button(
                            label="📥 Κατέβασμα αρχείου .h5ad",
                            data=f,
                            file_name=export_filename,
                            mime="application/octet-stream"
                        )

                    st.success("✅ Επιτυχής μετατροπή σε AnnData!")

                except Exception as e:
                    st.error(f"❌ Σφάλμα κατά την επεξεργασία: {str(e)}")

        except Exception as e:
            st.error(f"❌ Αποτυχία ανάγνωσης αρχείου: {str(e)}")



elif page == "⚙️Προκαταρκτική Επεξεργασία":
    st.title("⚙️Προκαταρκτική Επεξεργασία")
    st.write("Εδώ γίνεται φιλτράρισμα, κανονικοποίηση, PCA και clustering (Louvain).")

    file = st.file_uploader("📤 Ανέβασε αρχείο .h5ad")

    # Επιλογή thresholds από dropdown
    st.markdown("### Ρυθμίσεις Φιλτραρίσματος")
    col1, col2 = st.columns(2)
    with col1:
        min_genes = st.selectbox("Ελάχιστα γονίδια ανά κύτταρο", [50, 100, 200], index=2)
    with col2:
        min_cells = st.selectbox("Ελάχιστα κύτταρα ανά γονίδιο", [1, 3, 5], index=1)

    if file is not None:
        if st.button("Έναρξη Προεπεξεργασίας"):
            try:
                adata = sc.read_h5ad(file)

                # Φιλτράρισμα
                sc.pp.filter_cells(adata, min_genes=min_genes)
                sc.pp.filter_genes(adata, min_cells=min_cells)

                if adata.shape[0] == 0 or adata.shape[1] == 0:
                    st.error("❌ Τα δεδομένα είναι άδεια μετά το φιλτράρισμα. Δοκίμασε μικρότερα thresholds ή έλεγξε το αρχείο.")
                else:
                    # Προκαταρκτική επεξεργασία
                    sc.pp.normalize_total(adata)
                    sc.pp.log1p(adata)
                    sc.pp.pca(adata)
                    sc.pp.neighbors(adata)
                    sc.tl.louvain(adata)

                    out_file = f"preprocessed_{datetime.now().strftime('%Y-%m-%dT%H-%M')}.h5ad"
                    adata.write(out_file)

                    with open(out_file, "rb") as f:
                        st.download_button("📥 Κατέβασμα προεπεξεργασμένου αρχείου", f, file_name=out_file)

                    st.success("✅ Η επεξεργασία και ομαδοποίηση ολοκληρώθηκε.")
            except Exception as e:
                st.error(f"❌ Σφάλμα: {str(e)}")




elif page == "🧬Ενοποίηση Δεδομένων":
    st.title("🧬Ενοποίηση Δεδομένων")
    st.write("Συγχώνευση πολλαπλών αρχείων .h5ad σε ένα ενιαίο dataset.")

    files = st.file_uploader("📤 Ανέβασε πολλαπλά αρχεία .h5ad", type="h5ad", accept_multiple_files=True)

    if files:
        st.info("✅ Αρχεία φορτώθηκαν επιτυχώς. Πάτα 'Ενοποίηση' για να συνεχίσεις.")

        if st.button("🔁 Ενοποίηση"): #Κουμπί εκκινηση ενοποίησης
            try:
                adatas = [sc.read_h5ad(f) for f in files]
                merged = adatas[0].concatenate(adatas[1:])

                out_file = f"merged_{datetime.now().strftime('%Y-%m-%dT%H-%M')}.h5ad"
                merged.write(out_file)

                with open(out_file, "rb") as f: #Αποθήκευση αρχείου
                    st.download_button("📥 Κατέβασμα συγχωνευμένου αρχείου", f, file_name=out_file)

                st.success("✅ Η συγχώνευση ολοκληρώθηκε με επιτυχία.")
            except Exception as e:
                st.error(f"❌ Σφάλμα κατά την ενοποίηση: {str(e)}")



elif page == "🔗Scanorama Integration":
    st.title("🔗Ενσωμάτωση με Scanorama")
    st.write("Εκτελεί ενοποίηση πολλών αρχείων .h5ad με τη χρήση Scanorama.")

    uploaded_files = st.file_uploader("📤 Ανέβασε πολλαπλά αρχεία .h5ad", type="h5ad", accept_multiple_files=True)

    if uploaded_files and st.button("Εκκίνηση Ενοποίησης"):
        try:
            # Διαβάζουμε τα αρχεία .h5ad με τη βιβλιοθήκη scanpy
            adatas = [sc.read_h5ad(file) for file in uploaded_files]
            datasets = [adata.X for adata in adatas]  # Ανάκτηση των δεδομένων (X)
            genes_list = [adata.var_names.tolist() for adata in adatas]  # Ανάκτηση των ονομάτων των γονιδίων

            # Εκτέλεση ενοποίησης με scanorama
            integrated, genes = scanorama.integrate(datasets, genes_list)

            # Δημιουργούμε το ενοποιημένο AnnData από τα δεδομένα που επέστρεψε το Scanorama
            combined = integrated[0]
            for i in range(1, len(integrated)):
                combined = np.concatenate((combined, integrated[i]), axis=0)

            # Δημιουργία του AnnData αντικειμένου
            adata_combined = ad.AnnData(X=combined)  # Δημιουργία AnnData
            adata_combined.var_names = genes  # Ορισμός των γονιδίων

            # Αποθήκευση του ενοποιημένου αρχείου
            out_file = f"scanorama_integrated_{datetime.now().strftime('%Y-%m-%dT%H-%M')}.h5ad"
            adata_combined.write(out_file)

            with open(out_file, "rb") as f:
                st.download_button("📥 Κατέβασμα ενοποιημένου αρχείου", f, file_name=out_file)

            st.success("✅ Η ενοποίηση με Scanorama ολοκληρώθηκε.")
        except Exception as e:
            st.error(f"❌ Σφάλμα κατά την ενοποίηση: {str(e)}")




elif page == "🔎Ανάθεση Κυτταρικών Τύπων":
    st.title("🔎Ανάθεση Κυτταρικών Τύπων")
    st.write("Ανάθεση τύπων κυττάρων με βάση transcription factor signatures χρησιμοποιώντας το μοντέλο MLM.")

    h5ad_file = st.file_uploader("📤 Ανέβασε το αρχείο έκφρασης (.h5ad)", type=["h5ad"])
    tf_file = st.file_uploader("📤 Ανέβασε το αρχείο TF matrix (.csv)", type=["csv"])

    if h5ad_file and tf_file:
        if st.button("▶️ Εκκίνηση Ανάθεσης"):
            try:
                # Διαβάζει το .h5ad
                adata = sc.read_h5ad(h5ad_file)
                expr_df = pd.DataFrame(adata.X, index=adata.obs_names, columns=adata.var_names)

                # Διαβάζει το TF αρχείο
                tf_df = pd.read_csv(tf_file)

                # Έλεγχος σωστών στηλών
                required_cols = {'source', 'target', 'weight'}
                if not required_cols.issubset(tf_df.columns):
                    st.error("❌ Το αρχείο TF πρέπει να περιέχει τις στήλες: source, target, weight.")
                else:
                    shared_genes = set(tf_df['target']) & set(expr_df.columns)
                    st.info(f"🔍 Κοινά γονίδια μεταξύ TF και έκφρασης: {len(shared_genes)}")

                    if len(shared_genes) < 1:
                        st.error("❌ Δεν υπάρχουν κοινά γονίδια μεταξύ TF και dataset.")
                    else:
                        # Εκτέλεση MLM
                        result_scores, result_pvals = dc.run_mlm(
                            net=tf_df,
                            mat=expr_df,
                            source="source",
                            target="target",
                            weight="weight",
                            min_n=1
                        )

                        st.success("✅ Η εκτέλεση του MLM ολοκληρώθηκε.")
                        st.subheader("📊 Αποτελέσματα (Scores)")
                        st.dataframe(result_scores)

                        st.subheader("📊 P-values")
                        st.dataframe(result_pvals)

                        # Προαιρετικά: κουμπί για αποθήκευση
                        now = datetime.now().strftime("%Y-%m-%d_%H-%M")
                        scores_filename = f"mlm_scores_{now}.csv"
                        pvals_filename = f"mlm_pvals_{now}.csv"

                        st.download_button("📥 Κατέβασμα Scores", result_scores.to_csv().encode("utf-8"), scores_filename, "text/csv")
                        st.download_button("📥 Κατέβασμα P-values", result_pvals.to_csv().encode("utf-8"), pvals_filename, "text/csv")

            except Exception as e:
                st.error(f"❌ Σφάλμα κατά την εκτέλεση του MLM: {str(e)}")




elif page == "📊DEG Ανάλυση":
    st.title("📊Ανάλυση Διαφορικής Έκφρασης (DEG)")
    st.write("Ανάλυση διαφορικής γονιδιακής έκφρασης μεταξύ ομάδων κυττάρων.")

    file = st.file_uploader("📤 Ανέβασε αρχείο .h5ad")
    if file is not None:
        if st.button("Έναρξη Ανάλυσης"):
            try:
                adata = sc.read_h5ad(file)
                if 'louvain' in adata.obs:
                    sc.tl.rank_genes_groups(adata, 'louvain', method='t-test')
                    result_file = f"deg_{datetime.now().strftime('%Y-%m-%dT%H-%M')}.h5ad"
                    adata.write(result_file)

                    with open(result_file, "rb") as f:
                        st.download_button("📥 Κατέβασμα αποτελεσμάτων DEG", f, file_name=result_file)

                    st.success("✅ Η ανάλυση DEG ολοκληρώθηκε.")
                    sc.pl.rank_genes_groups(adata, sharey=False, show=True)
                else:
                    st.warning("Το πεδίο 'louvain' δεν υπάρχει στο αρχείο.")
            except Exception as e:
                st.error(f"❌ Σφάλμα: {str(e)}")


elif page == "🌋Volcano Plot":
    st.title("🌋Volcano Plot")
    st.write("Οπτικοποίηση DEG με volcano plot: fold change vs. -log10(p-value).")

    file = st.file_uploader("📤 Ανέβασε .h5ad")
    if file:
        if st.button("Έναρξη Volcano Plot"):
            try:
                adata = sc.read_h5ad(file)
                if 'rank_genes_groups' in adata.uns:
                    scores = adata.uns['rank_genes_groups']

                    group = scores['names'].dtype.names[0] if hasattr(scores['names'], 'dtype') else list(scores['names'].keys())[0]
                    gene_names = scores['names'][group]
                    pvals = scores['pvals'][group]
                    logfc = scores['logfoldchanges'][group]

                    df = pd.DataFrame({'gene': gene_names, 'pval': pvals, 'logfc': logfc})
                    df['neg_log_pval'] = -np.log10(df['pval'])

                    fig, ax = plt.subplots()
                    sns.scatterplot(data=df, x='logfc', y='neg_log_pval', hue='logfc', palette="coolwarm", ax=ax)
                    ax.set_title("Volcano Plot")
                    st.pyplot(fig)

                    # Save the plot
                    from io import BytesIO
                    buffer = BytesIO()
                    fig.savefig(buffer, format="png")
                    buffer.seek(0)

                    st.download_button(
                        label="📥 Κατέβασμα Volcano Plot (PNG)",
                        data=buffer,
                        file_name="volcano_plot.png",
                        mime="image/png"
                    )

                else:
                    st.warning("❗ Το αρχείο δεν περιέχει δεδομένα διαφορικής έκφρασης (rank_genes_groups).")
            except Exception as e:
                st.error(f"❌ Σφάλμα: {str(e)}")



elif page == "🎯Εκφράσεις Γονιδίων":
    import matplotlib.pyplot as plt
    from io import BytesIO

    st.title("🎯Εκφράσεις Γονιδίων")
    st.write("Οπτικοποίηση έκφρασης συγκεκριμένου γονιδίου σε ομάδες κυττάρων.")

    file = st.file_uploader("📤 Ανέβασε αρχείο .h5ad", type="h5ad")
    gene = st.text_input("🧬 Εισάγετε το όνομα του γονιδίου για προβολή")

    if file and gene:
        try:
            adata = sc.read_h5ad(file)

            if gene not in adata.var_names:
                st.error("❌ Το γονίδιο δεν υπάρχει στο dataset.")
            else:
                st.info(f"🔍 Προβάλλεται η έκφραση για το γονίδιο: `{gene}`")

                fig, ax = plt.subplots()
                if "louvain" in adata.obs.columns:
                    sc.pl.violin(adata, keys=gene, groupby="louvain", show=False, ax=ax)
                else:
                    st.warning("⚠️ Το πεδίο 'louvain' δεν υπάρχει. Προβάλλεται χωρίς ομαδοποίηση.")
                    sc.pl.violin(adata, keys=gene, show=False, ax=ax)

                st.pyplot(fig)

                buf = BytesIO()
                fig.savefig(buf, format="png")
                st.download_button(
                    label="📥 Κατέβασμα διαγράμματος",
                    data=buf.getvalue(),
                    file_name=f"gene_expression_{gene}.png",
                    mime="image/png"
                )

        except Exception as e:
            st.error(f"❌ Σφάλμα κατά την προβολή: {str(e)}")




elif page == "📄Δήλωση":
    st.title("📄Δήλωση Αυθεντικότητας")
    st.subheader("Δήλωση Αυθεντικότητας")
    st.markdown("""
    Την εργασία αυτή την έχω κάνει μόνος μου.  
    **Ονομάζομαι Σορτίκος Νικόλαος και ο Αριθμός Μητρώου μου είναι _inf2021207_**
    """)

if __name__ == "__main__":
    if "DOCKER" not in os.environ:
        print("✅ Η εφαρμογή είναι διαθέσιμη στο: http://localhost:8501")
    else:
        print("🚀 Streamlit σε Docker – άνοιξε τον browser και πήγαινε στο: http://localhost:8501")
