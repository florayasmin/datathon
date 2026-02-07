import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
import matplotlib.pyplot as plt
import geopandas as gpd
import folium

# =========================
# 1. Load data
# =========================
df = pd.read_csv('accessibility.csv')

print("Initial data sample:")
print(df.head())
print("\nColumns:", df.columns.tolist())
print("\nMissing values:\n", df.isna().sum())

# =========================
# 2. Rename columns
# =========================
df = df.rename(columns={
    'geometry/coordinates/0': 'longitude',
    'geometry/coordinates/1': 'latitude',
    'properties/label_type': 'label_type',
    'properties/neighborhood': 'neighborhood',
    'properties/severity': 'severity',
    'properties/is_temporary': 'is_temporary'
})

# =========================
# 3. Clean data
# =========================
df = df.dropna(subset=['longitude', 'latitude', 'neighborhood', 'severity'])
df['severity_num'] = pd.to_numeric(df['severity'], errors='coerce')
df['is_temporary_num'] = df['is_temporary'].astype(int)
df = df.dropna(subset=['severity_num'])

# =========================
# 4. One-hot encode issue types
# =========================
label_dummies = pd.get_dummies(df['label_type'], prefix='label')
df = pd.concat([df, label_dummies], axis=1)

# =========================
# 5. DBSCAN clustering (issue-level)
# =========================
feature_cols = ['longitude', 'latitude', 'severity_num', 'is_temporary_num'] + list(label_dummies.columns)
X = df[feature_cols]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

dbscan = DBSCAN(eps=0.7, min_samples=5)
df['issue_cluster'] = dbscan.fit_predict(X_scaled)

print("\nIssue Cluster Counts:")
print(df['issue_cluster'].value_counts())

# =========================
# 6. Aggregate by neighborhood
# =========================
neighborhood_df = (
    df[df['issue_cluster'] != -1]  # ignore noise
    .groupby('neighborhood')
    .agg(
        issue_count=('issue_cluster', 'count'),
        cluster_count=('issue_cluster', 'nunique'),
        avg_severity=('severity_num', 'mean'),
        temp_issue_ratio=('is_temporary_num', 'mean')
    )
    .reset_index()
)

# =========================
# 7. Neighborhood clustering with KMeans
# =========================
neigh_features = neighborhood_df[['issue_count', 'cluster_count', 'avg_severity', 'temp_issue_ratio']]
scaler2 = StandardScaler()
neigh_scaled = scaler2.fit_transform(neigh_features)

kmeans = KMeans(n_clusters=4, random_state=42)
neighborhood_df['neighborhood_cluster'] = kmeans.fit_predict(neigh_scaled)

# Optional: label clusters descriptively
def label_cluster(row):
    if row['avg_severity'] > 3 and row['issue_count'] > 3000:
        return "High issues, high severity"
    elif row['avg_severity'] <= 3 and row['temp_issue_ratio'] > 0.02:
        return "Low severity, mostly temporary"
    elif row['issue_count'] < 500:
        return "Few issues"
    else:
        return "Medium issues"

neighborhood_df['cluster_label'] = neighborhood_df.apply(label_cluster, axis=1)

# =========================
# 8. Summary
# =========================
summary = neighborhood_df.groupby('neighborhood_cluster')[['issue_count','cluster_count','avg_severity','temp_issue_ratio']].mean()
print("\nNeighborhood Cluster Summary:")
print(summary)

# =========================
# 9. Save CSVs
# =========================
df.to_csv("issues_clustered.csv", index=False)
neighborhood_df.to_csv("neighborhoods_clustered.csv", index=False)

# =========================
# 10. Map visualization (Seattle)
# =========================
# Load neighborhood boundaries (GeoJSON)
neighborhood_shapes = gpd.read_file("seattle_neighborhoods.geojson")

# Check the neighborhood column in your GeoJSON
print("GeoJSON columns:", neighborhood_shapes.columns)

# Replace 'name' below with the correct column name from your GeoJSON
geo_col_name = 'S_HOOD'  # <- update this if needed
choropleth_df = neighborhood_shapes.merge(
    neighborhood_df,
    left_on=geo_col_name,
    right_on='neighborhood',
    how='left'
)

# Fill NaN clusters with -1
choropleth_df['neighborhood_cluster'] = choropleth_df['neighborhood_cluster'].fillna(-1)

# Create a folium map centered on Seattle
seattle_map = folium.Map(location=[47.6062, -122.3321], zoom_start=11)

# Add choropleth layer
folium.Choropleth(
    geo_data=choropleth_df,
    name="Accessibility Clusters",
    data=choropleth_df,
    columns=[geo_col_name, 'neighborhood_cluster'],
    key_on=f"feature.properties.{geo_col_name}",
    fill_color="YlOrRd",
    fill_opacity=0.7,
    line_opacity=0.3,
    legend_name="Accessibility Neighborhood Cluster"
).add_to(seattle_map)

# Add hover tooltips
folium.features.GeoJson(
    choropleth_df,
    name="Neighborhoods",
    tooltip=folium.features.GeoJsonTooltip(
        fields=[geo_col_name, "cluster_label", "issue_count", "avg_severity"],
        aliases=["Neighborhood","Cluster","Issue Count","Avg Severity"],
        localize=True
    )
).add_to(seattle_map)

# Save map
seattle_map.save("seattle_accessibility_map.html")
print("Map saved: seattle_accessibility_map.html")
