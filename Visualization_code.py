import pandas as pd
import numpy as np
import zipfile
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

# Extracting the ZIP files
zip_files = {
    "Electricity_Production": "1.Electricity production from renewable sources, excluding hydroelectric (kWh).zip",
    "GDP_Growth": "2. GDP growth (annual %).zip",
    "GDP_Per_Capita": "3. GDP per capita (current US$).zip",
    "Renewable_Energy_Consumption": "4. Renewable energy consumption (% of total final energy consumption).zip"
}
extract_dir = "extracted_data/"
os.makedirs(extract_dir, exist_ok=True)

for key, file in zip_files.items():
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

# Function to load and preprocess data
def load_and_preprocess(file_name, value_name):
    df = pd.read_csv(f"{extract_dir}/{file_name}", skiprows=4)
    df = df[df.columns.drop(list(df.filter(regex='Unnamed')))]  # Drop any unnamed columns
    df = df.melt(id_vars=["Country Name"], var_name="Year", value_name=value_name)
    df = df[df['Country Name'].isin(selected_countries)]
    df = df[df['Year'].apply(lambda x: x.isnumeric())]  # Keep only numeric 'Year' values
    df['Year'] = df['Year'].astype(int)
    df = df[df['Year'].between(2000, 2020)]  # Filter for the years 2000 to 2020
    return df

# Names of the extracted CSV files
file_names = {
    "Electricity_Production": "API_EG.ELC.RNWX.KH_DS2_en_csv_v2_6305428.csv",
    "GDP_Growth": "API_NY.GDP.MKTP.KD.ZG_DS2_en_csv_v2_6298243.csv",
    "GDP_Per_Capita": "API_NY.GDP.PCAP.CD_DS2_en_csv_v2_6298251.csv",
    "Renewable_Energy_Consumption": "API_EG.FEC.RNEW.ZS_DS2_en_csv_v2_6299944.csv"
}
selected_countries = ["United States", "China", "India", "Germany", "Brazil", "South Africa", "Australia", "Japan"]

# Load and preprocess the data
df_electricity_production = load_and_preprocess(file_names["Electricity_Production"], "Electricity Production")
df_gdp_growth = load_and_preprocess(file_names["GDP_Growth"], "GDP Growth")
df_gdp_per_capita = load_and_preprocess(file_names["GDP_Per_Capita"], "GDP per Capita")
df_renewable_energy = load_and_preprocess(file_names["Renewable_Energy_Consumption"], "Renewable Energy Consumption")

# Merging the datasets
df_merged = df_electricity_production.merge(df_gdp_growth, on=["Country Name", "Year"])
df_merged = df_merged.merge(df_gdp_per_capita, on=["Country Name", "Year"])
df_merged = df_merged.merge(df_renewable_energy, on=["Country Name", "Year"])

# Normalize the data
scaler = StandardScaler()
numeric_columns = ['Electricity Production', 'GDP Growth', 'GDP per Capita', 'Renewable Energy Consumption']
df_merged[numeric_columns] = scaler.fit_transform(df_merged[numeric_columns])

# Calculate the mean only for numeric columns and fill NaN values
df_means = df_merged[numeric_columns].mean()
df_merged.fillna(df_means, inplace=True)

# Clustering
kmeans = KMeans(n_clusters=4, random_state=0).fit(df_merged[numeric_columns])
df_merged['Cluster'] = kmeans.labels_

# Cluster Interpretation
# Calculate mean only for numeric columns in each cluster
cluster_averages = df_merged.groupby('Cluster')[numeric_columns].mean()

# Visualization
sns.pairplot(df_merged, hue="Cluster", vars=numeric_columns)
plt.show()

# Model Fitting within Clusters
def linear_model(x, a, b):
    return a * x + b

fitting_results = []
for cluster in sorted(df_merged['Cluster'].unique()):
    cluster_data = df_merged[df_merged['Cluster'] == cluster]
    popt, pcov = curve_fit(linear_model, cluster_data['Year'], cluster_data['Renewable Energy Consumption'], maxfev=10000)
    fitting_results.append({"Cluster": cluster, "Model Coefficients": popt, "Covariance": pcov})
    plt.scatter(cluster_data['Year'], cluster_data['Renewable Energy Consumption'], label=f"Cluster {cluster} Data")
    plt.plot(cluster_data['Year'], linear_model(cluster_data['Year'], *popt), color='red', label="Fitted Line")
    plt.title(f"Cluster {cluster} - Renewable Energy Consumption Trend")
    plt.xlabel("Year")
    plt.ylabel("Normalized Renewable Energy Consumption")
    plt.legend()
    plt.show()
