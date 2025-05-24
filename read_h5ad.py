import scanpy as sc
import pandas as pd

adata = sc.read_h5ad("C:/Users/nikso/OneDrive/Desktop/TexnologiaLogismikou/temp/scanpy-pbmc3k.h5ad")

print(" Διαβάστηκε το αρχείο με επιτυχία.")
print("\n Σχήμα X:", adata.X.shape)

print("\n .obs - metadata κυττάρων:")
print(adata.obs.head())

print("\n  .var - γονίδια:")
print(adata.var.head())

print("\n Κλειδιά διαθέσιμα στο .uns:")
print(adata.uns.keys())
