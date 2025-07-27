# This script fetches molecular descriptors 2 D for compounds using their PubChem IDs.
# It reads a CSV file containing PubChem IDs, retrieves the corresponding SMILES,

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen
import pubchempy as pcp
import re # For regular expressions to extract IDs

def calculate_descriptors_from_smiles(smiles, ligand_id):
    
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        # print(f"Could not generate molecule from SMILES for ID: {ligand_id}") 
        return None

    try:
        return {
            'ID': ligand_id,
            'MW': round(Descriptors.MolWt(mol), 2),
            'LogP': round(Crippen.MolLogP(mol), 2),
            'HBD': Lipinski.NumHDonors(mol),
            'HBA': Lipinski.NumHAcceptors(mol),
            'TPSA': round(Descriptors.TPSA(mol), 2),
            'RotBonds': Lipinski.NumRotatableBonds(mol),
            'AromaticRings': Lipinski.NumAromaticRings(mol),
            'HeavyAtoms': Descriptors.HeavyAtomCount(mol),
        }
    except Exception as e:
        # print(f"Error calculating descriptors for ID {ligand_id}: {str(e)}") 
        return None

def main():
    print("\n=== PubChem ID to Molecular Descriptors ===\n")

    # Loading the CSV file containing PubChem IDs
    try:
        # Reading the CSV file, assuming no header, and the IDs are in the first column
        df_ligands = pd.read_csv('ligandnametoget.csv', header=None)
        
        pubchem_ids = []
        # Iterating through the first column to extract IDs
        for entry in df_ligands.iloc[:, 0]:
            # Ensuring that the entry is a string before applying regex
            entry_str = str(entry) 
            match = re.search(r'=\s*(\d+)', entry_str)
            if match:
                pubchem_ids.append(match.group(1))

        # Removing duplicates if any
        pubchem_ids = sorted(list(set(pubchem_ids)), key=int) 
        print(f"Extracted {len(pubchem_ids)} unique PubChem IDs.")

    except Exception as e:
        print(f"Error reading ligandnametoget.csv or extracting IDs: {e}")
        return

    results = []
    # Setting a counter for processed compounds to provide progress updates
    processed_count = 0
    total_ids = len(pubchem_ids)

    for p_id in pubchem_ids:
        try:
            # Fetching the compound by CID
            compound = pcp.Compound.from_cid(p_id)
            
            # --- MODIFICATION START ---
            # Checking if 'smiles' attribute exists and is not None/empty
            if hasattr(compound, 'smiles') and compound.smiles:
                # print(f" Fetched SMILES for CID {p_id}") # Removed for cleaner output
                desc = calculate_descriptors_from_smiles(compound.smiles, p_id)
                if desc:
                    results.append(desc)
                else:
                    print(f"Skipping CID {p_id}: Descriptor calculation failed for retrieved SMILES.")
            else:
                print(f"✖ Could not fetch valid SMILES for CID: {p_id}. Skipping.")
            # --- MODIFICATION END ---

        except Exception as e:
            print(f"✖ Error fetching data for CID {p_id}: {e}")
        
        processed_count += 1
        # Print progress
        if processed_count % 10 == 0 or processed_count == total_ids:
            print(f"Processed {processed_count}/{total_ids} IDs.")

    # Save results
    if results:
        df_descriptors = pd.DataFrame(results)
        output_file = 'PubChem_ID_DESCRIPTORS.csv'
        df_descriptors.to_csv(output_file, index=False)
        print(f"\n✔ Success! Saved descriptors for {len(results)} molecules to {output_file}")
        print("\nSample results:")
        print(df_descriptors.head())
    else:
        print("\n✖ No valid descriptors processed. All compounds either lacked SMILES or failed descriptor calculation.")

if __name__ == '__main__':
    main()