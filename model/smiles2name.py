import json
import pickle
import sys
from pathlib import Path
from typing import List

sys.path.append('D:\software\chemdraw\ChemScript\Lib')
# noinspection PyUnresolvedReferences
import ChemScript22 as ChemScript
from ChemScript import *


def load_text(path) -> List[str]:
    lines = []
    with open(path, 'r') as f:
        for line in f:
            lines.append(line.strip())
    return lines


def smiles2formal_name(smiles):
    mol_name = None
    try:
        mol = ChemicalData.LoadData(smiles)
        mol_name = mol.ChemicalName()
    except:
        return None
    finally:
        return mol_name


def audit_smiles_list(smiles_list):
    y_num = 0
    x_num = 0
    failed_smiles = []
    for smiles in smiles_list:
        mol_name = smiles2formal_name(smiles)
        if mol_name:
            y_num += 1
        else:
            x_num += 1
            failed_smiles.append(smiles)
    print(f"Audit result: {y_num} valid, {x_num} invalid in total {len(smiles_list)}")
    print(f"Failed smiles: \n{json.dumps(failed_smiles, indent=2)}")
    print()
    print("Succeeded smiles:")
    for smiles in smiles_list:
        mol_name = smiles2formal_name(smiles)
        if mol_name:
            print(f"{smiles} -> {mol_name}")

if __name__ == '__main__':
    smiles = 'ClC(Cl)(Cl)Cl'
    mol_name = smiles2formal_name(smiles)
    print(mol_name)

