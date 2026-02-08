import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.cluster import DBSCAN, KMeans
import matplotlib.pyplot as plt
import geopandas as gpd
import folium

# load the files
df = pd.read_csv("accessibility.csv")

print("Initial data sample:")
print(df.head())
print("\nColumns:", df.columns.tolist())
print("\nMissing values:\n", df.isna().sum())

# rename all columns for readability
df = df.rename(columns={
    "geometry/coordinates/0": "longitude",
    "geometry/coordinates/1": "latitude",
    "properties/label_type": "label_type",
    "properties/neighborhood": "neighborhood",
    "properties/severity": "severity",
    "properties/is_temporary": "is_temporary"
})

# clean data
df = df.dropna(subset=["longitude", "latitude", "neighborhood", "severity"])

df["severity_num"] = pd.to_numeric(df["severity"], errors="coerce")

# boolean handling
df["is_temporary_num"] = (
    df["is_temporary"]
    .fillna(0)
    .astype(bool)
    .astype(int)
)

df = df.dropna(subset=["severity_num"])

# DBSCAN clustering of spatial data
spatial_X = df[["longitude", "latitude"]]

spatial_scaler = StandardScaler()
spatial_scaled = spatial_scaler.fit_transform(spatial_X)

dbscan = DBSCAN(eps=0.5, min_samples=10)
df["issue_cluster"] = dbscan.fit_predict(spatial_scaled)

print("\nIssue Cluster Counts:")
print(df["issue_cluster"].value_counts())

# safety check for DBSCAN results
clustered_points = (df["issue_cluster"] != -1).sum()
if clustered_points < 10:
    raise ValueError(
        "DBSCAN produced too few clustered points. "
        "Increase eps or reduce min_samples."
    )

# aggregate to neighborhood level
neighborhood_df = (
    df[df["issue_cluster"] != -1]
    .groupby("neighborhood")
    .agg(
        issue_count=("issue_cluster", "count"),
        cluster_count=("issue_cluster", "nunique"),
        avg_severity=("severity_num", "mean"),
        temp_issue_ratio=("is_temporary_num", "mean")
    )
    .reset_index()
)

print("\nNeighborhood-level data:")
print(neighborhood_df.head())

# neighborhood features
neigh_features = neighborhood_df[
    ["issue_count", "cluster_count", "avg_severity", "temp_issue_ratio"]
]

neigh_scaler = StandardScaler()
neigh_scaled = neigh_scaler.fit_transform(neigh_features)

# find optimal K
inertias = []
sil_scores = []

if len(neighborhood_df) < 3:
    raise ValueError("Not enough neighborhoods for KMeans clustering.")

K = range(2, min(9, len(neighborhood_df)))

for k in K:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(neigh_scaled)
    inertias.append(km.inertia_)

    if len(set(labels)) > 1:
        sil_scores.append(silhouette_score(neigh_scaled, labels))
    else:
        sil_scores.append(-1)

# save plots instead of showing them
plt.figure()
plt.plot(list(K), inertias, marker="o")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.savefig("elbow_method.png", dpi=150, bbox_inches='tight')
plt.close()

plt.figure()
plt.plot(list(K), sil_scores, marker="o")
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis")
plt.savefig("silhouette_analysis.png", dpi=150, bbox_inches='tight')
plt.close()

k_opt = list(K)[sil_scores.index(max(sil_scores))]
print("Optimal number of clusters:", k_opt)

# KMeans clustering
kmeans = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
neighborhood_df["neighborhood_cluster"] = kmeans.fit_predict(neigh_scaled)

# interpret clusters
cluster_profiles = (
    neighborhood_df
    .groupby("neighborhood_cluster")[[
        "issue_count", "cluster_count", "avg_severity", "temp_issue_ratio"
    ]]
    .mean()
)

print("\nCluster Profiles:")
print(cluster_profiles)

# calculate accessibility score (lower is better/safer/more accessible)
# higher issue count, more clusters, higher severity = worse accessibility
neighborhood_df["accessibility_score"] = (
    neighborhood_df["issue_count"] * 0.4 +
    neighborhood_df["cluster_count"] * 0.2 +
    neighborhood_df["avg_severity"] * 0.3 +
    neighborhood_df["temp_issue_ratio"] * 0.1
)

# create descriptive cluster labels
def create_cluster_label(row):
    # determine issue level based on issue_count and avg_severity
    if row["issue_count"] > neighborhood_df["issue_count"].quantile(0.66):
        issue_level = "High Issue"
    elif row["issue_count"] > neighborhood_df["issue_count"].quantile(0.33):
        issue_level = "Medium Issue"
    else:
        issue_level = "Low Issue"
    
    # check if mostly temporary
    if row["temp_issue_ratio"] > 0.5:
        issue_level += ", Mostly Temporary"
    
    return issue_level

neighborhood_df["cluster_label"] = neighborhood_df.apply(create_cluster_label, axis=1)

# find the most common issue type per neighborhood
main_issue_per_neighborhood = (
    df[df["issue_cluster"] != -1]
    .groupby("neighborhood")["label_type"]
    .agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else "Unknown")
    .reset_index()
    .rename(columns={"label_type": "main_issue_type"})
)

neighborhood_df = neighborhood_df.merge(
    main_issue_per_neighborhood,
    on="neighborhood",
    how="left"
)

neighborhood_df["main_issue_type"] = neighborhood_df["main_issue_type"].fillna("Unknown")


# predictive modeling: severity prediction
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

print("\n" + "="*50)
print("PREDICTIVE MODELING: SEVERITY PREDICTION")
print("="*50)

# add spatial context
df['issues_in_neighborhood'] = df.groupby('neighborhood')['severity_num'].transform('count')
df['avg_severity_in_neighborhood'] = df.groupby('neighborhood')['severity_num'].transform('mean')

# one-hot encode label_type
df_model = df.copy()
df_model = pd.get_dummies(df_model, columns=['label_type'], prefix='type', drop_first=False)

# select features for prediction
feature_cols = ['longitude', 'latitude', 'is_temporary_num', 
                'issues_in_neighborhood', 'avg_severity_in_neighborhood']
# add all the one-hot encoded type columns
type_cols = [col for col in df_model.columns if col.startswith('type_')]
feature_cols.extend(type_cols)

# prepare data
X = df_model[feature_cols].fillna(0)
y = df_model['severity_num']

# train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=None
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# train Random Forest model
rf_model = RandomForestRegressor(
    n_estimators=100, 
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# predict
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

# evaluate model
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
train_rmse = mean_squared_error(y_train, y_pred_train) ** 0.5
test_rmse = mean_squared_error(y_test, y_pred_test) ** 0.5
test_mae = mean_absolute_error(y_test, y_pred_test)

test_mape = mean_absolute_percentage_error(y_test, y_pred_test)
accuracy_pct = (1 - test_mape) * 100

# improvement over baseline logic
baseline_mae = abs(y_test - y_train.mean()).mean()
print(f"Improvement over baseline: {((baseline_mae - test_mae) / baseline_mae) * 100:.2f}%")

print("\n--- Model Performance ---")
print(f"Training R² Score: {train_r2:.4f}")
print(f"Test R² Score: {test_r2:.4f}")
print(f"Training RMSE: {train_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test MAE (Margin of Error): {test_mae:.4f}") # average point error
print(f"Model Accuracy (MAPE): {accuracy_pct:.2f}%") # percentage accuracy

# feature importance
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n--- Top 10 Most Important Features ---")
print(importance_df.head(10).to_string(index=False))

# save feature importance plot
plt.figure(figsize=(10, 6))
top_features = importance_df.head(15)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance')
plt.title('Top 15 Feature Importances for Severity Prediction')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches='tight')
plt.close()

# add predictions to original dataframe
df_model['predicted_severity'] = rf_model.predict(X)
df_model['prediction_error'] = abs(df_model['severity_num'] - df_model['predicted_severity'])

# save model results
model_results = df_model[['longitude', 'latitude', 'neighborhood', 'severity_num', 
                          'predicted_severity', 'prediction_error']].copy()
model_results.to_csv("severity_predictions.csv", index=False)

print("\n--- Prediction Statistics ---")
print(f"Mean actual severity: {df_model['severity_num'].mean():.2f}")
print(f"Mean predicted severity: {df_model['predicted_severity'].mean():.2f}")
print(f"Mean prediction error: {df_model['prediction_error'].mean():.2f}")

print("\n" + "="*50)
print("PREDICTIVE MODELING COMPLETE")
print("="*50 + "\n")

# save outputs
df.to_csv("issues_clustered.csv", index=False)
neighborhood_df.to_csv("neighborhoods_clustered.csv", index=False)

# map visualization
neighborhood_shapes = gpd.read_file("seattle_neighborhoods.geojson")
print("GeoJSON columns:", neighborhood_shapes.columns)

geo_col_name = "S_HOOD"  # update if needed
if geo_col_name not in neighborhood_shapes.columns:
    raise ValueError(f"{geo_col_name} not found in GeoJSON columns")


# normalize names to avoid merge failures
neighborhood_shapes[geo_col_name] = (
    neighborhood_shapes[geo_col_name].str.strip().str.lower()
)
neighborhood_df["neighborhood"] = (
    neighborhood_df["neighborhood"].str.strip().str.lower()
)

choropleth_df = neighborhood_shapes.merge(
    neighborhood_df,
    left_on=geo_col_name,
    right_on="neighborhood",
    how="left"
)

# for areas with no data, set to NaN (will appear without color)
choropleth_df["accessibility_score"] = choropleth_df["accessibility_score"].fillna(float('nan'))

# color in the predicted areas
# identify neighborhoods with no data
all_hoods = set(neighborhood_shapes[geo_col_name].unique())
data_hoods = set(neighborhood_df["neighborhood"].unique())
missing_hoods = list(all_hoods - data_hoods)

# create "synthetic" data for missing neighborhoods to run through the model
# use the centroid of the neighborhood to get lat/long for the prediction
prediction_records = []
for hood in missing_hoods:
    geom = neighborhood_shapes[neighborhood_shapes[geo_col_name] == hood].geometry.iloc[0]
    centroid = geom.centroid
    
    # created a feature row matching X_train columns
    # assume 0 existing issues for the 'neighborhood context' features
    row = {col: 0 for col in feature_cols}
    row['longitude'] = centroid.x
    row['latitude'] = centroid.y
    # assume a generic 'type' (e.g., the most common type from your training data)
    
    pred_severity = rf_model.predict(pd.DataFrame([row]))[0]
    
    # calculate predicted accessibility score
    pred_score = (0 * 0.4) + (0 * 0.2) + (pred_severity * 0.3) + (0 * 0.1)
    
    prediction_records.append({
        "neighborhood": hood,
        "accessibility_score": pred_score,
        "is_predicted": "PREDICTION (Modeled Estimation)"
    })

predict_df = pd.DataFrame(prediction_records)

# merge with the GeoJSON for a secondary layer
predicted_choropleth = neighborhood_shapes.merge(
    predict_df, left_on=geo_col_name, right_on="neighborhood", how="inner"
)

seattle_map = folium.Map(location=[47.6062, -122.3321], zoom_start=11)

folium.Choropleth(
    geo_data=predicted_choropleth,
    data=predicted_choropleth,
    columns=[geo_col_name, "accessibility_score"],
    key_on=f"feature.properties.{geo_col_name}",
    fill_color="RdYlGn_r", 
    fill_opacity=0.35,
    line_opacity=0.8,
    line_dash='5,5',
    legend_name="Predicted Accessibility Score (Est.)"
).add_to(seattle_map)

folium.GeoJson(
    predicted_choropleth,
    style_function=lambda x: {'fillOpacity': 0, 'color': 'gray', 'weight': 1},
    tooltip=folium.features.GeoJsonTooltip(
        fields=[geo_col_name, "is_predicted", "accessibility_score"],
        aliases=["PREDICTION", "Data Status:", "Estimated Score:"],
        localize=True
    )
).add_to(seattle_map)

# reversed Red-Yellow-Green:
# red = high accessibility_score = many issues = LEAST accessible
# green = low accessibility_score = few issues = MOST accessible
folium.Choropleth(
    geo_data=choropleth_df,
    data=choropleth_df,
    columns=[geo_col_name, "accessibility_score"],
    key_on=f"feature.properties.{geo_col_name}",
    fill_color="RdYlGn_r",  # red (bad) to green (good)
    fill_opacity=0.75,
    line_opacity=0,
    nan_fill_color="white",
    nan_fill_opacity=0.0, 
    legend_name="Actual Accessibility Score (Reported Data)"
).add_to(seattle_map)

choropleth_with_data = choropleth_df[choropleth_df['accessibility_score'].notna()]

folium.GeoJson(
    choropleth_with_data,
    style_function=lambda x: {
        'fillOpacity': 0,
        'color': 'blue',
        'weight': 2,
        'opacity': 0.5
    },
    tooltip=folium.features.GeoJsonTooltip(
        fields=[geo_col_name, "cluster_label", "main_issue_type", "issue_count", "avg_severity", "accessibility_score"],
        aliases=["Neighborhood", "Cluster Label", "Main Issue Present", "Issue Count", "Avg Severity", "Accessibility Score"],
        localize=True
    )
).add_to(seattle_map)

seattle_map.save("seattle_accessibility_map.html")
print("\n" + "="*50)
print("\nOutputs saved:")
print("="*50)
print("- seattle_accessibility_map.html (interactive map)")
print("- elbow_method.png (clustering diagnostic)")
print("- silhouette_analysis.png (clustering diagnostic)")
print("- feature_importance.png (predictive model)")
print("- issues_clustered.csv (individual issues with clusters)")
print("- neighborhoods_clustered.csv (neighborhood summaries)")
print("- severity_predictions.csv (model predictions)")
print("="*50)