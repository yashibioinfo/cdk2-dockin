# # QSAR2.py
# This script performs QSAR modeling using 2D molecular descriptors, 3D molecular descriptors and docking scores.

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.feature_selection import VarianceThreshold, RFE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Setting up directories
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

# 1. Loading and preparing data
print("Loading of data")
try:
    docking_results = pd.read_csv('best_docking_results.csv')
    print(f"Loaded {len(docking_results)} compounds")
except Exception as e:
    print(f"Error loading docking results: {e}")
    exit()

# 2. Calculation of  2D descriptors
def calculate_2d_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Failed to process SMILES: {smiles}")
        return None
    
    try:
        descriptors = {
            'MW': Descriptors.MolWt(mol),
            'LogP': Crippen.MolLogP(mol),
            'TPSA': Descriptors.TPSA(mol),
            'HBD': Lipinski.NumHDonors(mol),
            'HBA': Lipinski.NumHAcceptors(mol),
            'RotBonds': Lipinski.NumRotatableBonds(mol),
            'AromaticRings': Lipinski.NumAromaticRings(mol),
            'HeavyAtoms': Descriptors.HeavyAtomCount(mol),
            'FractionCSP3': Descriptors.FractionCSP3(mol),
            'RingCount': Descriptors.RingCount(mol),
            'LipinskiPass': int(all([
                Descriptors.MolWt(mol) <= 500,
                Crippen.MolLogP(mol) <= 5,
                Lipinski.NumHDonors(mol) <= 5,
                Lipinski.NumHAcceptors(mol) <= 10
            ]))
        }
        return descriptors
    except Exception as e:
        print(f"Error calculating descriptors for {smiles}: {e}")
        return None

print("Calculating 2D descriptors...")
descriptors_2d = []
valid_indices = []
for i, smiles in enumerate(docking_results['SMILES']):
    desc = calculate_2d_descriptors(smiles)
    if desc:
        descriptors_2d.append(desc)
        valid_indices.append(i)

df_2d = pd.DataFrame(descriptors_2d)
df_2d['Compound_ID'] = docking_results.loc[valid_indices, 'Compound_ID'].values
df_2d.to_csv('data/molecular_descriptors_2d.csv', index=False) # Save the initial df_2d
print(f"Calculated descriptors for {len(df_2d)} compounds and saved to data/molecular_descriptors_2d.csv")


# --- START: Outlier Rectification Section ---
print("\nApplying outlier rectification...")

# --- OPTION 1: Data Transformation (e.g., Log Transformation) ---

columns_to_transform = ['MW', 'TPSA', 'HBD', 'HBA'] 

for col in columns_to_transform:
    if col in df_2d.columns: 
        if (df_2d[col] <= 0).any():
            print(f"Warning: Column '{col}' contains non-positive values. Log transformation may not be appropriate. Skipping.")
            continue
        
        df_2d[f'{col}_log'] = np.log(df_2d[col])
        print(f"Applied log transformation to '{col}'. New column '{col}_log' created.")
        
    else:
        print(f"Column '{col}' not found in the descriptor DataFrame. Skipping transformation for this column.")

# ---or  OPTION 2: Winsorization (Capping) ---

columns_to_winsorize = {
    'LogP': {'lower': 0.01, 'upper': 0.99},
}    

for col, percentiles in columns_to_winsorize.items():
    if col in df_2d.columns: 
        lower_bound = df_2d[col].quantile(percentiles['lower'])
        upper_bound = df_2d[col].quantile(percentiles['upper'])
        
        # Apply winsorization
        df_2d[f'{col}_winsorized'] = df_2d[col].clip(lower=lower_bound, upper=upper_bound)
        print(f"Applied winsorization to '{col}' (capped at {percentiles['lower']*100}th and {percentiles['upper']*100}th percentiles). New column '{col}_winsorized' created.")
        
    else:
        print(f"Column '{col}' not found in the descriptor DataFrame. Skipping winsorization for this column.")

print("\nDataFrame after outlier rectification (head):")
print(df_2d.head())

# --- END: Outlier Rectification Section ---


# 3. Preparaion of feature matrix and target (after rectification)

feature_columns = [col for col in df_2d.columns if col not in ['Compound_ID']]

X = df_2d.drop(columns=['Compound_ID']) 
y = docking_results.loc[valid_indices, 'Docking_Score'].values

# 4. Data preprocessing
print("Preprocessing data...")

# Handling missing values
X = X.fillna(X.mean())

vt = VarianceThreshold(threshold=0.01) 
X = vt.fit_transform(X)

if not isinstance(X, pd.DataFrame):
    X = pd.DataFrame(X, columns=df_2d.drop(columns=['Compound_ID']).columns[vt.get_support()])

# Standardization of the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Training QSAR models
print("Training QSAR models...")
models = {
    'MLR': LinearRegression(),
    'PLS': PLSRegression(n_components=min(5, X_train.shape[1])),
    'RF': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'SVR': SVR(kernel='rbf')
}

results = []
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')
    
    results.append({
        'Model': name,
        'R2': r2,
        'RMSE': rmse,
        'MAE': mae,
        'CV_R2_Mean': np.mean(cv_scores),
        'CV_R2_Std': np.std(cv_scores)
    })
    
    # Saving the model
    joblib.dump(model, f'models/{name}_model.pkl')
    print(f"Saved {name} model to models/{name}_model.pkl")

# Saving the results
results_df = pd.DataFrame(results)
results_df.to_csv('models/model_performance.csv', index=False)
print("\nModel performance saved to models/model_performance.csv")

# 6. Feature importance analysis
print("\nAnalyzing feature importance...")
rf_model = models['RF']
importances = rf_model.feature_importances_

original_features_after_vt = df_2d.drop(columns=['Compound_ID']).columns[vt.get_support()]
features = original_features_after_vt

feat_imp = pd.DataFrame({'Feature': features, 'Importance': importances})
feat_imp = feat_imp.sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_imp.head(15))
plt.title('Top 15 Important Features (Random Forest)')
plt.tight_layout()
plt.savefig('visualizations/feature_importance.png')
print("Feature importance plot saved to visualizations/feature_importance.png")

# 7. Model comparison
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='R2', data=results_df.sort_values('R2', ascending=False))
plt.title('Model Performance Comparison (R²)')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('visualizations/model_performance.png')
print("Model comparison plot saved to visualizations/model_performance.png")

# 8. Generate predictions for all compounds
best_model_name = results_df.loc[results_df['R2'].idxmax()]['Model']
best_model = models[best_model_name]
all_predictions = best_model.predict(X_scaled)

results_all = pd.DataFrame({
    'Compound_ID': df_2d['Compound_ID'], # Use Compound_ID from the df_2d after descriptor calculation
    'SMILES': docking_results.loc[valid_indices, 'SMILES'].values, # Use SMILES from original docking_results aligned by valid_indices
    'Docking_Score': y,
    'QSAR_Prediction': all_predictions,
    'Residual': y - all_predictions
})
results_all.to_csv('data/qsar_predictions.csv', index=False)
print("\nQSAR predictions saved to data/qsar_predictions.csv")
print(f"\nBest performing model: {best_model_name} (R² = {results_df.loc[results_df['R2'].idxmax()]['R2']:.3f})")

print("\nQSAR modeling complete!")


# End of QSAR2.py