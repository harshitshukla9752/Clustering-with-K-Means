# K-Means Clustering Project

## Project Objective

Perform unsupervised learning using K-Means clustering to identify patterns and segments in a dataset.

## Tools Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

## Dataset

* Mall Customer Segmentation Dataset (or any suitable dataset)
* Columns: `CustomerID`, `Gender`, `Age`, `Annual Income (k$)`, `Spending Score (1-100)`

## Project Structure

```
Clustering-with-K-Means/
│
├── data/
│   └── mall_customers.csv      # Dataset file
│
├── outputs/
│   ├── clusters.png              # 2D PCA cluster visualization
│   ├── elbow_method.png          # Elbow method plot
│   ├── raw_data_pairplot.png     # Raw data pairplot
│   └── silhouette_score.txt      # Silhouette score
│
├── src/
│   └── kmeans_clustering.py    # Main Python script
│
└── README.md                  # Project documentation
```

## Steps Performed

1. **Load and Visualize Dataset**

   * Load dataset using Pandas.
   * Handle categorical columns (e.g., encode `Gender`).
   * Visualize distributions and relationships.

2. **Optional PCA for 2D Visualization**

   * Reduce dimensions for plotting (if dataset is high-dimensional).

3. **K-Means Clustering**

   * Fit K-Means to the data.
   * Assign cluster labels.
   * Store results in `outputs/clusters.png`.

4. **Elbow Method to Find Optimal K**

   * Plot inertia vs K.
   * Save plot as `outputs/elbow_method.png`.

5. **Visualize Clusters**

   * Color-code points based on cluster assignments.
   * Include cluster centroids.

6. **Evaluate Clustering**

   * Compute Silhouette Score.
   * Save results to `outputs/silhouette_score.txt`.

## Outputs

* **clusters.png**: Scatter plot of data points colored by cluster.
* **elbow_method.png**: Inertia plot to choose optimal K.
* **silhouette_score.txt**: Numeric score indicating clustering quality.

## Notes

* Ensure categorical columns are converted to numeric values before fitting K-Means.
* Always visualize clusters to validate results.
* Experiment with different K values and evaluate using Silhouette Score.
