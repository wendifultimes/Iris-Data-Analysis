# Iris Data Analysis
*Clustering analysis of the [Iris dataset](https://archive.ics.uci.edu/dataset/53/iris)*

## **Overview**

>This project provides an in-depth analysis of the Iris dataset, a classic dataset in the field of machine learning and statistics. The dataset consists of 150 samples from each of three species of Iris flowers (Iris setosa, Iris virginica, and Iris versicolor). Four features were measured from each sample: the >lengths and the widths of the sepals and petals.
>
>The primary objective of this analysis is to cluster the data into distinct groups based on the features and visualize the clusters in a 2D space using Principal Component Analysis (PCA).

---

## **Methodology**

### *Data Loading and Preprocessing:*
  
> The Iris dataset, available in the sklearn library, was loaded into a pandas DataFrame for easier manipulation and exploration.
>The dataset consists of 150 samples from each of three species of Iris flowers (Iris setosa, Iris virginica, and Iris versicolor). Four features were measured from each sample: the lengths and the widths of the sepals and petals.
>
>
```python
from sklearn.datasets import load_iris
import pandas as pd

# Load the Iris dataset and convert it into a pandas DataFrame
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = [iris.target_names[i] for i in iris.target]
```
### *Data Scaling:*

>The features of the dataset were standardized using the 'StandardScaler' from 'sklearn'. This ensures that each feature has a mean of 0 and a standard deviation of 1, which is essential for many machine learning algorithms.
>
```python
from sklearn.preprocessing import StandardScaler

# Standardize the dataset
scaler = StandardScaler()
scaled_data = scaler.fit_transform(iris_df.iloc[:, :-1])
```

### *KMeans Clustering:*
  
> KMeans clustering was applied to the standardized data to group the samples into three clusters, corresponding to the three species of Iris flowers.
>
> 
```python
from sklearn.cluster import KMeans

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_data)
iris_df['cluster'] = clusters
```

### *Dimensionality Reduction with PCA:*
  
> To visualize the clusters in a 2D space, the dimensionality of the data was reduced from 4D to 2D using PCA.
>
> 
```python
from sklearn.decomposition import PCA

# Reduce dimensionality for visualization
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_data)
```
  
### *Visualization:*
  
>   To generate a scatter plot using Seaborn to visualize the clusters in the 2D PCA space, with each species represented by a different color.
>
> 
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the Iris dataset and convert it into a pandas DataFrame
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = [iris.target_names[i] for i in iris.target]

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(iris_df.iloc[:, :-1])

# Apply PCA and reduce the data to 2 components
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# Create a DataFrame with the PCA components and species
pca_df = pd.DataFrame(data=pca_data, columns=['PC1', 'PC2'])
pca_df['species'] = iris_df['species']

# Visualize the clusters in 2D PCA space
plt.figure(figsize=(10, 6))
sns.scatterplot(x='PC1', y='PC2', hue='species', data=pca_df, palette='viridis', s=100, alpha=0.7)
plt.title('Clusters visualized in 2D PCA space')
plt.legend(title='Species')
plt.show()
```

![Iris Clusters visualized in a 2D PCA space](https://chat.noteable.io/origami/o/c8695e16e36c45e0988e2c9ea189ed74.png)

## **Assumptions Made**

*Number of Clusters:*

>Assumption: It was assumed that there would be three clusters, corresponding to the three species of Iris flowers.
>
>Reason: The Iris dataset is a well-known dataset in the data science community. It is commonly known that it contains three distinct species of Iris flowers. Therefore, it was logical to assume that the data would be grouped into three clusters.

*Features are Equally Important:*

>Assumption: By standardizing the dataset, it was assumed that all features are equally important for clustering.
>
>Reason: Without prior domain knowledge indicating otherwise, it's a common practice to treat all features with equal importance. Standardizing ensures that no feature dominates the clustering due to its scale.

## **Results**

>The KMeans clustering algorithm effectively grouped the samples into three distinct clusters. The visualization in the 2D PCA space showed clear separation between the clusters, indicating that the features of the Iris dataset are suitable for distinguishing between the three species of Iris flowers.

## **Code and Tools**

*The analysis was performed using Python, with the following libraries:*

 > **pandas**: For data manipulation and analysis.
>
>  **numpy**: For numerical operations.
>
>  **matplotlib and seaborn**: For data visualization.
>
>  **scikit-learn**: For machine learning algorithms and data preprocessing.

*The code for this analysis, along with the visualization, can be found in the accompanying files.*

----
## **Conclusion**

>This analysis demonstrates the power of clustering algorithms in grouping data into meaningful clusters based on their features. The Iris dataset, with its clear separations between species, serves as an excellent example of how data analytics techniques can be applied to derive insights from data.
----
## **Contact**

> I'm always open to discussing data analytics, project collaborations, or any other opportunities. Feel free to reach out to me through any of the channels below:
>
> LinkedIn: [Wendy Liang](https://www.linkedin.com/in/wendyliang38/)
>
> Email: wendybliang38@gmail.com
>
> Portfolio: [GitHub](https://github.com/wendifultimes/wendy-liang)
>
> If you have any questions or feedback regarding this project, please don't hesitate to open an issue or submit a pull request.
