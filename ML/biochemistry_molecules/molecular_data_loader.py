import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator
from mordred import Calculator, descriptors
import requests
import gzip
import os
from typing import List, Dict, Tuple, Optional
from config import Config

class MolecularDataLoader:
    def __init__(self, config):
        self.config = config
        self.calculator = Calculator(descriptors, ignore_3D=True)
        
    def download_dataset(self, dataset_name: str) -> str:
        if dataset_name not in self.config.DATASETS:
            raise ValueError(f"Dataset {dataset_name} not found in config")
        
        dataset_info = self.config.DATASETS[dataset_name]
        url = dataset_info['url']
        
        data_path = self.config.DATA_DIR / f"{dataset_name}.csv"
        
        if os.path.exists(data_path):
            print(f"Dataset {dataset_name} already exists")
            return str(data_path)
        
        print(f"Downloading {dataset_name} dataset...")
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            if url.endswith('.gz'):
                with gzip.open(response.content, 'rt') as f:
                    content = f.read()
            else:
                content = response.text
            
            with open(data_path, 'w') as f:
                f.write(content)
            
            print(f"Dataset saved to {data_path}")
            return str(data_path)
            
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return None
    
    def load_dataset(self, dataset_name: str) -> pd.DataFrame:
        data_path = self.download_dataset(dataset_name)
        
        if data_path is None:
            raise FileNotFoundError(f"Could not load dataset {dataset_name}")
        
        df = pd.read_csv(data_path)
        print(f"Loaded {dataset_name} dataset with shape: {df.shape}")
        return df
    
    def clean_smiles(self, smiles: str) -> Optional[str]:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Chem.MolToSmiles(mol)
        except:
            return None
    
    def calculate_molecular_descriptors(self, smiles: str) -> Dict[str, float]:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}
            
            descriptors = {
                'MW': Descriptors.MolWt(mol),
                'LogP': Crippen.MolLogP(mol),
                'HBD': Lipinski.NumHDonors(mol),
                'HBA': Lipinski.NumHAcceptors(mol),
                'TPSA': Descriptors.TPSA(mol),
                'RotBonds': Descriptors.NumRotatableBonds(mol),
                'AromaticRings': Descriptors.NumAromaticRings(mol),
                'HeavyAtoms': Descriptors.HeavyAtomCount(mol),
                'FormalCharge': Chem.rdmolops.GetFormalCharge(mol),
                'NumHDonors': Descriptors.NumHDonors(mol),
                'NumHAcceptors': Descriptors.NumHAcceptors(mol),
                'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
                'NumSaturatedRings': Descriptors.NumSaturatedRings(mol),
                'NumAliphaticRings': Descriptors.NumAliphaticRings(mol),
                'NumAromaticRings': Descriptors.NumAromaticRings(mol),
                'NumSaturatedHeterocycles': Descriptors.NumSaturatedHeterocycles(mol),
                'NumAliphaticHeterocycles': Descriptors.NumAliphaticHeterocycles(mol),
                'NumAromaticHeterocycles': Descriptors.NumAromaticHeterocycles(mol),
                'NumSpiroAtoms': Descriptors.NumSpiroAtoms(mol),
                'NumBridgeheadAtoms': Descriptors.NumBridgeheadAtoms(mol)
            }
            
            return descriptors
            
        except Exception as e:
            print(f"Error calculating descriptors for {smiles}: {e}")
            return {}
    
    def calculate_fingerprints(self, smiles: str) -> Dict[str, np.ndarray]:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}
            
            fingerprints = {}
            
            morgan_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fingerprints['Morgan'] = np.array(morgan_fp)
            
            maccs_fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
            fingerprints['MACCS'] = np.array(maccs_fp)
            
            rdkit_fp = rdMolDescriptors.RDKFingerprint(mol)
            fingerprints['RDKit'] = np.array(rdkit_fp)
            
            ecfp4_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fingerprints['ECFP4'] = np.array(ecfp4_fp)
            
            ecfp6_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)
            fingerprints['ECFP6'] = np.array(ecfp6_fp)
            
            return fingerprints
            
        except Exception as e:
            print(f"Error calculating fingerprints for {smiles}: {e}")
            return {}
    
    def calculate_mordred_descriptors(self, smiles: str) -> Dict[str, float]:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}
            
            desc = self.calculator(mol)
            desc_dict = desc.asdict()
            
            numeric_desc = {}
            for key, value in desc_dict.items():
                if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                    numeric_desc[key] = value
            
            return numeric_desc
            
        except Exception as e:
            print(f"Error calculating Mordred descriptors for {smiles}: {e}")
            return {}
    
    def smiles_to_graph(self, smiles: str) -> Optional[Dict]:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            atoms = mol.GetAtoms()
            bonds = mol.GetBonds()
            
            node_features = []
            for atom in atoms:
                features = [
                    atom.GetAtomicNum(),
                    atom.GetDegree(),
                    atom.GetFormalCharge(),
                    atom.GetHybridization().real,
                    atom.GetIsAromatic(),
                    atom.GetTotalNumHs(),
                    atom.GetTotalValence()
                ]
                node_features.append(features)
            
            edge_index = []
            edge_features = []
            
            for bond in bonds:
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                
                edge_index.append([i, j])
                edge_index.append([j, i])
                
                bond_features = [
                    bond.GetBondType().real,
                    bond.GetIsAromatic(),
                    bond.GetIsConjugated(),
                    bond.IsInRing()
                ]
                edge_features.append(bond_features)
                edge_features.append(bond_features)
            
            return {
                'node_features': np.array(node_features, dtype=np.float32),
                'edge_index': np.array(edge_index, dtype=np.long).T,
                'edge_features': np.array(edge_features, dtype=np.float32)
            }
            
        except Exception as e:
            print(f"Error converting SMILES to graph for {smiles}: {e}")
            return None
    
    def process_dataset(self, df: pd.DataFrame, smiles_col: str = 'smiles', 
                      target_cols: List[str] = None) -> pd.DataFrame:
        print("Processing molecular dataset...")
        
        df_processed = df.copy()
        
        df_processed['smiles_clean'] = df_processed[smiles_col].apply(self.clean_smiles)
        df_processed = df_processed.dropna(subset=['smiles_clean'])
        
        print(f"Valid SMILES: {len(df_processed)}")
        
        descriptors_list = []
        fingerprints_list = []
        mordred_list = []
        graphs_list = []
        
        for idx, row in df_processed.iterrows():
            if idx % 1000 == 0:
                print(f"Processing molecule {idx}/{len(df_processed)}")
            
            smiles = row['smiles_clean']
            
            descriptors = self.calculate_molecular_descriptors(smiles)
            descriptors_list.append(descriptors)
            
            fingerprints = self.calculate_fingerprints(smiles)
            fingerprints_list.append(fingerprints)
            
            mordred_desc = self.calculate_mordred_descriptors(smiles)
            mordred_list.append(mordred_desc)
            
            graph = self.smiles_to_graph(smiles)
            graphs_list.append(graph)
        
        descriptors_df = pd.DataFrame(descriptors_list)
        df_processed = pd.concat([df_processed, descriptors_df], axis=1)
        
        for fp_name in ['Morgan', 'MACCS', 'RDKit', 'ECFP4', 'ECFP6']:
            fp_data = [fp_dict.get(fp_name, np.zeros(2048)) for fp_dict in fingerprints_list]
            fp_df = pd.DataFrame(fp_data, columns=[f'{fp_name}_{i}' for i in range(len(fp_data[0]))])
            df_processed = pd.concat([df_processed, fp_df], axis=1)
        
        mordred_df = pd.DataFrame(mordred_list)
        df_processed = pd.concat([df_processed, mordred_df], axis=1)
        
        df_processed['graph_data'] = graphs_list
        
        print(f"Processed dataset shape: {df_processed.shape}")
        return df_processed
    
    def create_molecular_dataset(self, dataset_name: str, target_cols: List[str] = None) -> pd.DataFrame:
        df = self.load_dataset(dataset_name)
        
        if target_cols is None:
            dataset_info = self.config.DATASETS[dataset_name]
            target_cols = dataset_info['tasks']
        
        df_processed = self.process_dataset(df, target_cols=target_cols)
        
        output_path = self.config.DATA_DIR / f"{dataset_name}_processed.csv"
        df_processed.to_csv(output_path, index=False)
        print(f"Processed dataset saved to {output_path}")
        
        return df_processed
