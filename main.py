import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
# --- 1. Synthetic Data Generation ---
np.random.seed(42)  # for reproducibility
num_samples = 200
data = {
    'Sentiment': np.random.choice(['Positive', 'Negative', 'Neutral'], size=num_samples),
    'Engagement': np.random.randint(0, 100, size=num_samples), # Engagement score (e.g., likes, retweets)
    'Advocacy': np.random.randint(0, 2, size=num_samples) # 1 for advocate, 0 for non-advocate
}
df = pd.DataFrame(data)
# --- 2. Data Preprocessing ---
# Convert categorical sentiment to numerical using one-hot encoding
df = pd.get_dummies(df, columns=['Sentiment'], prefix=['Sentiment'])
# Scale numerical features for clustering
scaler = StandardScaler()
numerical_cols = ['Engagement']
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
# --- 3. Sentiment Clustering ---
# Apply KMeans clustering (assuming 3 clusters based on sentiment)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['Sentiment_Negative', 'Sentiment_Neutral', 'Sentiment_Positive', 'Engagement']])
# --- 4. Advocacy Prediction ---
# (Simplified prediction:  We'll just look at cluster-wise advocacy rates)
cluster_advocacy = df.groupby('Cluster')['Advocacy'].mean()
print("Cluster-wise Advocacy Rates:")
print(cluster_advocacy)
# --- 5. Visualization ---
# Cluster visualization (Engagement vs. Sentiment - simplified for demonstration)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Engagement', y='Sentiment_Positive', hue='Cluster', data=df, palette='viridis')
plt.title('Sentiment Clusters and Engagement')
plt.xlabel('Engagement (Scaled)')
plt.ylabel('Positive Sentiment (One-Hot Encoded)')
plt.savefig('sentiment_clusters.png')
print("Plot saved to sentiment_clusters.png")
# Bar plot of cluster advocacy rates
plt.figure(figsize=(8, 6))
sns.barplot(x=cluster_advocacy.index, y=cluster_advocacy.values)
plt.title('Advocacy Rate per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Average Advocacy (0-1)')
plt.savefig('advocacy_rates.png')
print("Plot saved to advocacy_rates.png")