import pandas as pd
import numpy as np
import re
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tabulate import tabulate
def clean_price(price):
    if isinstance(price, str):
        price = re.sub(r'[^\d]', '', price)
    return int(price) if price else 0

data_sources = {
    "CPU": [
        ("amd Ryzen 5 3600 with Wraith Stealth Cooler (100-000000031) 3.6 Ghz Upto 4.2 GHz AM4 Socket 6 Cores 12 Threads 3 MB L2 32 MB L3 Desktop Processor", "â‚¹9,800"),
        ("amd Ryzen 9 5900X 3.7 GHz Upto 4.8 GHz AM4 Socket 12 Cores 24 Threads Desktop Processor", "â‚¹34,890"),
        ("Ultra 3.5 GHz LGA 1150 Intel Core i3 4150 3.5Ghz 4th gen Processor", "â‚¹760"),
        ("GIGASTAR 3.4 GHz LGA 1155 Intel i5-3570K For H61 Motherboard 3rd Gen Processor", "â‚¹1,890"),
    ],
    "GPU": [
        ("ASUS NVIDIA GeForce GT 730 2 GB GDDR5 Graphics Card", "â‚¹4,399"),
        ("ZEBRONICS NVIDIA ZEB-GT610 2 GB DDR3 Graphics Card", "â‚¹2,049"),
        ("GEONIX NVIDIA GX GT730 4GB D3 4 GB DDR3 Graphics Card", "â‚¹2,990"),
        ("GALAX NVIDIA GEFORCE GT 730 4GB DDR3 4 GB GDDR3 Graphics Card", "â‚¹6,870"),
    ],
    "RAM": [
        ("HyperX NA DDR3 8 GB (Dual Channel) Laptop, PC", "â‚¹1,285"),
        ("XPG RAM DDR4 16 GB PC DDR4", "â‚¹3,290"),
        ("Hynix DDR3 8 GB PC", "â‚¹760"),
        ("Hynix DDR3 4 GB PC", "â‚¹425"),
    ],
    "SSD": [
        ("WESTERN DIGITAL WD Green SATA 240 GB", "â‚¹1,349"),
        ("Crucial BX500 3D NAND 500 GB", "â‚¹2,183"),
        ("WD Blue NVMe SN570 500 GB", "â‚¹2,579"),
        ("ZEBRONICS ZEB-SD26 256 GB", "â‚¹1,099"),
    ],
    "PSU": [
        ("Ant Esports VS450L 450W", "â‚¹1,650"),
        ("ZEBRONICS SMPS ZEB-N450W", "â‚¹999"),
        ("Frontech PS-0005 SMPS", "â‚¹569"),
        ("ZEB-VS500Z 500W PSU", "â‚¹1,999"),
    ],
    "Motherboard": [
        ("ASRock B450M-HDV", "â‚¹5,678"),
        ("ZEBRONICS ZEB-H81M2", "â‚¹2,449"),
        ("ASUS PRIME H610M-E", "â‚¹8,950"),
        ("MSI B550M PRO-VDH WIFI", "â‚¹12,999"),
    ],
    "Cabinet": [
        ("Ant Esports ICE-120AG Mid Tower", "â‚¹2,629"),
        ("Ant Esports ICE-200TG Gaming Cabinet", "â‚¹3,125"),
        ("Deepcool MATREXX 40 Mid Tower", "â‚¹2,811"),
        ("Matrix MTXCC009 MINI TOWER", "â‚¹1,798"),
    ]
}
dfs = {}
for name, items in data_sources.items():
    df = pd.DataFrame(items, columns=['Product', 'Price'])
    df['Price'] = df['Price'].apply(clean_price)
    dfs[name] = df
builds = []
n_builds = 100

for _ in range(n_builds):
    build = {}
    total = 0
    desc = ""
    for comp, df in dfs.items():
        row = df.sample(1).iloc[0]
        build[comp] = row['Product']
        desc += row['Product'] + " "
        total += row['Price']
    build['TotalPrice'] = total
    build['Description'] = desc
    builds.append(build)

build_df = pd.DataFrame(builds)
tfidf = TfidfVectorizer(max_features=100)
X_tfidf = tfidf.fit_transform(build_df['Description'])
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_tfidf.toarray())

svd = TruncatedSVD(n_components=10)
X_svd = svd.fit_transform(X_tfidf)
X = np.hstack([X_pca, X_svd])
y = build_df['TotalPrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, c='blue', label="Predictions")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label="Ideal")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Gaming PC Price")
plt.legend()
plt.grid()
plt.show()
build_df_reset = build_df.reset_index(drop=True)
test_indices = y_test.index
test_builds = build_df_reset.loc[test_indices].reset_index(drop=True)
test_builds["Predicted Price"] = y_pred
test_builds["Actual Price"] = y_test.values
test_builds["Predicted Price"] = test_builds["Predicted Price"].apply(lambda x: f"Rs.{int(x):,}")
test_builds["Actual Price"] = test_builds["Actual Price"].apply(lambda x: f"Rs.{int(x):,}")
component_cols = list(data_sources.keys())
cols_to_display = component_cols + ["Actual Price", "Predicted Price"]
print("\n Sample Predictions (with selected components):\n")
print(tabulate(test_builds[cols_to_display].head(5), headers="keys", tablefmt="fancy_grid", showindex=True))