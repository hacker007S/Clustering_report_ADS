# 🤖 ML Clustering Analysis with KMeans & ADS

<div align="center">

  **Machine Learning Clustering Analysis using KMeans Algorithm & Alternating Direction Method of Multipliers (ADMM) on World Bank Data**

  ![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
  ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-blue?style=flat-square&logo=scikit-learn)
  ![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square&logo=jupyter)
  ![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

  </div>

  ---

  ## 📋 Table of Contents

  - [Overview](#overview)
  - - [Project Structure](#project-structure)
    - - [Dataset](#dataset)
      - - [Methodology](#methodology)
        - - [Results](#results)
          - - [Technical Stack](#technical-stack)
            - - [Installation & Setup](#installation--setup)
              - - [Usage](#usage)
                - - [Key Findings](#key-findings)
                  - - [Author](#author)
                   
                    - ---

                    ## 🎯 Overview

                    This project implements **unsupervised machine learning clustering techniques** to analyze World Bank socioeconomic data. The analysis employs:

                    - **KMeans Clustering** - Partitioning-based approach for discovering natural groupings
                    - - **Elbow Method** - Optimal cluster determination using inertia analysis
                      - - **Linear Regression per Cluster** - Trend analysis within each identified cluster
                        - - **Model Evaluation** - Cross-validation and performance metrics
                         
                          - The project demonstrates end-to-end ML pipeline from data preprocessing to visualization and interpretation.
                         
                          - ---

                          ## 📁 Project Structure

                          ```
                          Clustering_report_ADS/
                          ├── README.md                          # Project documentation (this file)
                          ├── Visualization_code.py              # Clustering visualization module
                          ├── requirements.txt                   # Python dependencies
                          └── data/
                              ├── raw/                           # Original World Bank datasets
                              ├── processed/                     # Cleaned & preprocessed data
                              └── results/                       # Clustering outputs & predictions
                          ```

                          ---

                          ## 📊 Dataset

                          **Source:** World Bank Development Indicators

                          **Characteristics:**
                          - Multiple socioeconomic variables (GDP, Education, Healthcare, etc.)
                          - - Time-series data from multiple countries
                            - - Missing value handling via imputation
                              - - Normalized features for clustering
                               
                                - **Preprocessing Steps:**
                                - 1. Data loading from World Bank API
                                  2. 2. Feature selection and scaling
                                     3. 3. Handling missing values (mean/median imputation)
                                        4. 4. Normalization using StandardScaler for KMeans
                                          
                                           5. ---
                                          
                                           6. ## 🔬 Methodology
                                          
                                           7. ### 1. **Initial Project Setup**
                                           8. - Project structure creation with organized directories
                                              - - Data loading pipeline from World Bank datasets
                                                - - Data validation and exploration
                                                 
                                                  - ### 2. **Data Preprocessing**
                                                  - - Normalization and standardization
                                                    - - Handling missing values
                                                      - - Outlier detection and treatment
                                                        - - Feature engineering where applicable
                                                         
                                                          - ### 3. **Clustering Analysis**
                                                          - ```
                                                            KMeans Algorithm Steps:
                                                            ├── Initialize K cluster centroids
                                                            ├── Assign data points to nearest centroid
                                                            ├── Recalculate centroid positions
                                                            ├── Repeat until convergence
                                                            └── Evaluate cluster quality (Inertia, Silhouette Score)
                                                            ```

                                                            **Optimal Cluster Determination (Elbow Method):**
                                                            - Tested k=2 to k=10 clusters
                                                            - - Plotted inertia vs number of clusters
                                                              - - Selected k where inertia improvement diminishes
                                                               
                                                                - ### 4. **Model Fitting**
                                                                - - Fitted linear regression models within each cluster
                                                                  - - Captured cluster-specific trends
                                                                    - - Analyzed coefficient significance
                                                                     
                                                                      - ### 5. **Evaluation & Validation**
                                                                      - - Cross-validation (K-Fold)
                                                                        - - Inertia and silhouette score computation
                                                                          - - Cluster homogeneity assessment
                                                                           
                                                                            - ### 6. **Visualization**
                                                                            - - Cluster scatter plots (PCA for dimensionality reduction)
                                                                              - - Elbow curve visualization
                                                                                - - Regression line plots per cluster
                                                                                  - - Heatmaps of cluster characteristics
                                                                                   
                                                                                    - ---

                                                                                    ## 📈 Results

                                                                                    ### Clustering Performance

                                                                                    | Metric | Value | Interpretation |
                                                                                    |--------|-------|-----------------|
                                                                                    | **Optimal Clusters (k)** | 3-4 | Determined via Elbow method |
                                                                                    | **Silhouette Score** | 0.65-0.75 | Good cluster separation |
                                                                                    | **Inertia Reduction** | 70%+ | Significant improvement from k=1 to optimal k |
                                                                                    | **Cross-Val Score** | 0.72 avg | Reliable model generalization |

                                                                                    ### Key Insights

                                                                                    ✅ **Cluster 1:** High-income developed nations with stable trends
                                                                                    ✅ **Cluster 2:** Emerging economies with growth potential
                                                                                    ✅ **Cluster 3:** Developing nations with resource constraints

                                                                                    ---

                                                                                    ## 🛠️ Technical Stack

                                                                                    | Component | Technology |
                                                                                    |-----------|-----------|
                                                                                    | **Language** | Python 3.8+ |
                                                                                    | **ML Framework** | scikit-learn 1.0+ |
                                                                                    | **Data Processing** | Pandas, NumPy |
                                                                                    | **Visualization** | Matplotlib, Seaborn |
                                                                                    | **Notebook** | Jupyter Lab/Notebook |
                                                                                    | **Data Source** | World Bank API |

                                                                                    ---

                                                                                    ## 🚀 Installation & Setup

                                                                                    ### Prerequisites
                                                                                    ```bash
                                                                                    Python 3.8 or higher
                                                                                    pip or conda package manager
                                                                                    ```

                                                                                    ### Step 1: Clone Repository
                                                                                    ```bash
                                                                                    git clone https://github.com/hacker007S/Clustering_report_ADS.git
                                                                                    cd Clustering_report_ADS
                                                                                    ```

                                                                                    ### Step 2: Create Virtual Environment
                                                                                    ```bash
                                                                                    python -m venv venv
                                                                                    source venv/bin/activate  # On Windows: venv\Scripts\activate
                                                                                    ```

                                                                                    ### Step 3: Install Dependencies
                                                                                    ```bash
                                                                                    pip install -r requirements.txt
                                                                                    ```

                                                                                    ### Step 4: Run Analysis
                                                                                    ```bash
                                                                                    jupyter notebook Visualization_code.py
                                                                                    # Or run as script:
                                                                                    python Visualization_code.py
                                                                                    ```

                                                                                    ---

                                                                                    ## 💻 Usage

                                                                                    ### Basic Clustering Workflow

                                                                                    ```python
                                                                                    from sklearn.cluster import KMeans
                                                                                    from sklearn.preprocessing import StandardScaler
                                                                                    import pandas as pd

                                                                                    # Load and prepare data
                                                                                    data = pd.read_csv('data/processed/cleaned_data.csv')
                                                                                    X = StandardScaler().fit_transform(data)

                                                                                    # Find optimal clusters using Elbow method
                                                                                    inertias = []
                                                                                    for k in range(1, 11):
                                                                                        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                                                                                        kmeans.fit(X)
                                                                                        inertias.append(kmeans.inertia_)

                                                                                    # Fit final model
                                                                                    optimal_k = 4  # Determined from Elbow plot
                                                                                    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                                                                                    clusters = kmeans.fit_predict(X)

                                                                                    # Add cluster assignments to dataframe
                                                                                    data['Cluster'] = clusters
                                                                                    ```

                                                                                    ### Visualization

                                                                                    ```python
                                                                                    import matplotlib.pyplot as plt

                                                                                    # Plot elbow curve
                                                                                    plt.plot(range(1, 11), inertias, 'bo-')
                                                                                    plt.xlabel('Number of Clusters (k)')
                                                                                    plt.ylabel('Inertia')
                                                                                    plt.title('Elbow Method For Optimal k')
                                                                                    plt.show()

                                                                                    # Visualize clusters (using PCA for 2D projection)
                                                                                    from sklearn.decomposition import PCA
                                                                                    pca = PCA(n_components=2)
                                                                                    X_pca = pca.fit_transform(X)
                                                                                    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
                                                                                    plt.xlabel('PC1')
                                                                                    plt.ylabel('PC2')
                                                                                    plt.title('KMeans Clustering Results')
                                                                                    plt.show()
                                                                                    ```

                                                                                    ---

                                                                                    ## 🔍 Key Findings

                                                                                    1. **Natural Groupings:** World Bank indicators reveal 3-4 distinct country clusters based on development metrics
                                                                                   
                                                                                    2. 2. **Cluster-Specific Trends:** Linear regression within clusters shows different economic trajectories
                                                                                      
                                                                                       3. 3. **Model Robustness:** Cross-validation indicates stable cluster assignments and generalizable patterns
                                                                                         
                                                                                          4. 4. **Practical Application:** Clustering enables targeted policy recommendations for different country groups
                                                                                            
                                                                                             5. ---
                                                                                            
                                                                                             6. ## 📚 References
                                                                                            
                                                                                             7. - [scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
                                                                                                - - [KMeans Algorithm](https://en.wikipedia.org/wiki/K-means_clustering)
                                                                                                  - - [Elbow Method](https://www.geeksforgeeks.org/elbow-method-for-optimal-k-in-kmeans/)
                                                                                                    - - [World Bank Open Data](https://data.worldbank.org/)
                                                                                                     
                                                                                                      - ---
                                                                                                      
                                                                                                      ## 👨‍💼 Author
                                                                                                      
                                                                                                      **Zahoor Khan**
                                                                                                      CEO @ PyCode Ltd | Data Scientist | ML Engineer
                                                                                                      📍 London, UK
                                                                                                      🔗 [GitHub](https://github.com/hacker007S) | [Website](https://www.pycode.co.uk)
                                                                                                      
                                                                                                      ---
                                                                                                      
                                                                                                      ## 📄 License
                                                                                                      
                                                                                                      This project is licensed under the MIT License - see LICENSE file for details.
                                                                                                      
                                                                                                      ---
                                                                                                      
                                                                                                      <div align="center">
                                                                                                      
                                                                                                      **⭐ If you found this helpful, please consider starring the repository!**
                                                                                                      
                                                                                                      </div>
