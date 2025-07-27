#This script fetches 3D molecular descriptors  for a list of compounds from PubChem.
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors3D, rdMolDescriptors
from rdkit.Chem import MACCSkeys
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Configuration
WORKING_DIR = r"C:\Users\bajpa\Desktop\week_3new\all_ligands"
OUTPUT_DIR = os.path.join(WORKING_DIR, "3d_analysis_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sample data 
COMPOUND_IDS = [
    53235510,445408,46930950,1318,7175,8362,660,19539,5161,7184,54682930,10251,8569,8365,54678486,5359485,16330,5280567,9458,323,3418,54688261,6873,161860,16523,55748,54689800,10286,60997,2741,10752,11833,4758425,11707110,5287969,6930950
]  

# 1. Fetching SMILES from PubChem
def fetch_smiles(cid):
    from pubchempy import get_compounds
    try:
        return get_compounds(cid, 'cid')[0].canonical_smiles
    except:
        print(f"Couldn't fetch SMILES for CID {cid}")
        return None

# 2. Generation of  3D Conformation
def generate_3d_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)  # For reproducibility
        try:
            AllChem.MMFFOptimizeMolecule(mol)
            return mol
        except:
            return None
    return None

# 3. Calculation of 3D Descriptors
def calculate_3d_descriptors(mol):
    if not mol or mol.GetNumConformers() == 0:
        return None
    
    return {
        'Volume': AllChem.ComputeMolVolume(mol),
        'SurfaceArea': rdMolDescriptors.CalcLabuteASA(mol),
        'RadiusOfGyration': Descriptors3D.RadiusOfGyration(mol),
        'Asphericity': Descriptors3D.Asphericity(mol),
        'PMI1': Descriptors3D.PMI1(mol),
        'PMI2': Descriptors3D.PMI2(mol),
        'PMI3': Descriptors3D.PMI3(mol),
        'InertialShapeFactor': Descriptors3D.InertialShapeFactor(mol),
        'Eccentricity': Descriptors3D.Eccentricity(mol),
        'SpherocityIndex': Descriptors3D.SpherocityIndex(mol)
    }

# 4. Calculation of Fingerprints
def calculate_fingerprints(mol):
    return {
        'ECFP4': AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048).ToBitString(),
        'MACCS': MACCSkeys.GenMACCSKeys(mol).ToBitString(),
        'AtomPairs': AllChem.GetHashedAtomPairFingerprintAsBitVect(mol).ToBitString()  # Alternative to pharmacophore
    }

# 5. Main Processing
def process_compounds():
    results_3d = []
    results_fp = []
    
    for cid in COMPOUND_IDS:
        smiles = fetch_smiles(cid)
        if not smiles:
            continue
            
        mol = generate_3d_mol(smiles)
        if not mol:
            continue
            
        # Calculate descriptors
        desc_3d = calculate_3d_descriptors(mol)
        if desc_3d:
            results_3d.append({'CID': cid, 'SMILES': smiles, **desc_3d})
            
        # Calculate fingerprints
        fps = calculate_fingerprints(mol)
        if fps:
            results_fp.append({'CID': cid, **fps})
    
    return pd.DataFrame(results_3d), pd.DataFrame(results_fp)

# 6. Generate Visualizations
def generate_plots(df_3d):
    plt.figure(figsize=(15, 10))
    
    # Descriptor distributions
    plt.subplot(2, 2, 1)
    sns.histplot(data=df_3d, x='Volume', kde=True)
    plt.title('Molecular Volume Distribution')
    
    plt.subplot(2, 2, 2)
    sns.scatterplot(data=df_3d, x='PMI1', y='PMI3', hue='RadiusOfGyration')
    plt.title('Principal Moments Analysis')
    
    plt.subplot(2, 2, 3)
    sns.boxplot(data=df_3d[['Asphericity', 'SpherocityIndex']])
    plt.title('Shape Descriptors')
    
    plt.subplot(2, 2, 4)
    sns.scatterplot(data=df_3d, x='SurfaceArea', y='Volume')
    plt.title('Surface Area vs Volume')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'descriptor_distribution_plots.png'))
    plt.close()

# 7. Execute Pipeline
print("Starting 3D descriptor calculation...")
df_3d, df_fp = process_compounds()

if not df_3d.empty:
    # Save results
    df_3d.to_csv(os.path.join(OUTPUT_DIR, 'molecular_descriptors_3d.csv'), index=False)
    df_fp.to_csv(os.path.join(OUTPUT_DIR, 'molecular_fingerprints.csv'), index=False)
    
    # Generate visualizations
    generate_plots(df_3d)
    
    print("\nSuccessfully generated:")
    print(f"- 3D descriptors for {len(df_3d)} compounds")
    print(f"- Fingerprints for {len(df_fp)} compounds")
    print(f"\nFiles saved to:\n{OUTPUT_DIR}")
    
    # Show sample output
    print("\nSample 3D descriptors:")
    print(df_3d.head().to_markdown(index=False))
    
    print("\nSample fingerprints (ECFP4 first 50 bits):")
    print(df_fp['CID'].head(), df_fp['ECFP4'].str[:50].head())
else:
    print("No compounds were processed successfully")

print("\nAnalysis complete!")