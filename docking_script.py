import os
import subprocess
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# --- Configuration Section ---
VINA_PATH = r"C:\Users\bajpa\bio_env\vina\vina.exe"  # Path to your Vina executable
RECEPTOR = "receptor_cleaned.pdbqt"                 # Your prepared receptor file
CONFIG = "vina_config.txt"                          # Docking box configuration file
LIGANDS_DIR = "pdbqt_files"                         # Directory with your ligand PDBQT files
RESULTS_DIR = "docking_results"                     # Output directory for results
os.makedirs(RESULTS_DIR, exist_ok=True)             # Create output directory if needed

# --- Helper Functions ---
def parse_affinity(log_file):
    """Extracts binding affinity from Vina log file."""
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if line.strip().startswith('1 '):  # Look for Mode 1 result
                    parts = line.strip().split()
                    return float(parts[1])        # Return affinity (2nd column)
        print(f"  Warning: No affinity found in {log_file}")
    except Exception as e:
        print(f"  Error reading {log_file}: {e}")
    return None

def calculate_rmsd(ref_pdb, docked_pdbqt):
    """Calculates RMSD between reference and docked pose."""
    try:
        ref_mol = Chem.MolFromPDBFile(ref_pdb)
        docked_mol = Chem.MolFromPDBFile(docked_pdbqt)
        if ref_mol and docked_mol:
            return AllChem.GetBestRMS(ref_mol, docked_mol)
    except Exception as e:
        print(f"  RMSD calculation failed: {e}")
    return None

# --- Main Docking Workflow ---
results = []
print(f"\n{'='*50}\nStarting AutoDock Vina Docking\n{'='*50}")
print(f"Receptor: {RECEPTOR}")
print(f"Ligands Directory: {LIGANDS_DIR}")
print(f"Results Directory: {RESULTS_DIR}\n")

# --- Main Docking Workflow ---
results = []
print(f"\n{'='*50}\nStarting AutoDock Vina Docking\n{'='*50}")

# Process each PDBQT file
for lig_file in sorted(os.listdir(LIGANDS_DIR)):
    # Match files like lig_1.pdbqt, lig_2.pdbqt, etc.
    if lig_file.startswith("lig_") and lig_file.endswith(".pdbqt"):
        try:
            # Extract numeric ID (e.g., "1" from "lig_1.pdbqt")
            lig_id = lig_file.split('_')[1].split('.')[0]
            lig_path = os.path.join(LIGANDS_DIR, lig_file)
            
            # Output files
            out_pdbqt = os.path.join(RESULTS_DIR, f"lig_{lig_id}_docked.pdbqt")
            out_log = os.path.join(RESULTS_DIR, f"lig_{lig_id}_log.txt")
            
            # Vina command
            cmd = [
                VINA_PATH,
                "--receptor", RECEPTOR,
                "--ligand", lig_path,
                "--config", CONFIG,
                "--out", out_pdbqt,
                "--num_modes", "3"
            ]
            
            print(f"\nProcessing {lig_file}...")
            
            # Run Vina
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Save log
            with open(out_log, 'w') as f:
                f.write(result.stdout)
            
            # Parse results
            affinity = parse_affinity(out_log)
            rmsd = None
            
            # Optional RMSD calculation
            ref_pdb = os.path.join(LIGANDS_DIR, f"lig_{lig_id}_ref.pdb")
            if os.path.exists(ref_pdb):
                rmsd = calculate_rmsd(ref_pdb, out_pdbqt)
            
            results.append({
                "Ligand_ID": f"lig_{lig_id}",
                "Affinity_kcal_mol": affinity,
                "RMSD_angstrom": rmsd,
                "Docked_Pose": out_pdbqt,
                "Status": "Success"
            })
            
            print(f"  ✓ Success | Affinity: {affinity:.2f} kcal/mol")
            
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr.strip()
            results.append({
                "Ligand_ID": f"lig_{lig_id}",
                "Affinity_kcal_mol": None,
                "RMSD_angstrom": None,
                "Docked_Pose": None,
                "Status": f"Failed: {error_msg[:100]}"
            })
            print(f"  ✗ Failed | Error: {error_msg[:100]}...")
        except Exception as e:
            print(f"  ! Unexpected error with {lig_file}: {str(e)[:100]}")