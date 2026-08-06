"""
Module 2: Virtual Extract Profile Twin (extract_twin.py)
--------------------------------------------------------
Synthesizes a digital twin profile of crude natural extract by:
1. Aggregating SMILES from Module 1.
2. Classifying chemical taxonomy using RDKit SMARTS functional group matchers.
3. Generating 3D conformers using RDKit ETKDG algorithm.
4. Constructing PyTorch Geometric ensemble batch graph objects for downstream GNN inference.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    HAS_RDKIT = True
except ImportError:
    Chem = None
    AllChem = None
    HAS_RDKIT = False

# [버그 수정] 원본에는 torch import가 아예 없었는데 아래 _mol_to_pyg_graph()에서
# torch.tensor(...)를 바로 썼음. try/except로 감싸져 있어서 크래시는 안 났지만
# (NameError를 조용히 삼키고 None 반환) PyG 그래프 기능 자체가 항상 비활성화돼 있었음.
# models/gnn_predictor.py와 동일한 HAS_TORCH 가드 패턴으로 정식 임포트함.
try:
    import torch
    HAS_TORCH = True
except (ImportError, OSError, Exception):
    torch = None
    HAS_TORCH = False

try:
    from torch_geometric.data import Data, Batch
    HAS_PYG = True
except (ImportError, OSError, Exception):
    HAS_PYG = False

logger = logging.getLogger(__name__)

# SMARTS functional group patterns for chemical taxonomy classification
SMARTS_TAXONOMY = {
    "Flavonoids": "c1ccc(cc1)-c2coc3cc(O)cc(O)c3c2=O",  # Chromone core with aryl substituent
    "Biflavonoids": "c1ccc(c(c1)c2coc3ccccc3c2=O)-c4ccc(cc4)c5coc6ccccc6c5=O",
    "Terpenoids": "CC(C)=CCC",  # Isoprene unit signature
    "Diterpenes": "CC1(C)CCCC2(C)C1CCC3C2CCC4C3(C)CCC4O",
    "Alkaloids": "[#7;R]",  # Ring-bound nitrogen atom
    "Phenylpropanoids": "c1ccccc1C=CC(=O)O",  # C6-C3 skeleton
    "Polyphenols": "c1c(O)c(O)ccc1"
}


class ExtractProfileTwin:
    """
    Module 2: Virtual Natural Extract Digital Twin Synthesizer.
    """

    def __init__(self):
        self.tax_smarts = {}
        if HAS_RDKIT and Chem is not None:
            for cls_name, pattern in SMARTS_TAXONOMY.items():
                smarts_mol = Chem.MolFromSmarts(pattern)
                if smarts_mol is not None:
                    self.tax_smarts[cls_name] = smarts_mol

    def build_extract_twin(self, compounds: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Synthesize the extract profile twin: taxonomy distribution, 3D conformers, and PyG graph.
        """
        processed_compounds = []
        class_counts: Dict[str, int] = {}
        pyg_data_list = []

        for comp in compounds:
            smiles = comp.get("smiles", "")
            mol = Chem.MolFromSmiles(smiles) if (HAS_RDKIT and Chem and smiles) else None

            # 1. Chemical Taxonomy Matching
            chem_classes = self._classify_molecule(mol, comp.get("name", ""), smiles)
            for cls in chem_classes:
                class_counts[cls] = class_counts.get(cls, 0) + 1

            # 2. 3D Conformer Generation via ETKDG
            conformer_3d_success = self._embed_3d_conformer(mol) if HAS_RDKIT else True

            # 3. RDKit Mol to PyG Graph conversion
            graph_data = self._mol_to_pyg_graph(mol) if (HAS_RDKIT and HAS_TORCH) else None
            if graph_data is not None:
                pyg_data_list.append(graph_data)

            comp_entry = dict(comp)
            comp_entry["chemical_classes"] = chem_classes
            comp_entry["has_3d_conformer"] = conformer_3d_success
            processed_compounds.append(comp_entry)

        # PyG Ensemble Batch Graph construction
        batch_graph = None
        if HAS_PYG and pyg_data_list:
            batch_graph = Batch.from_data_list(pyg_data_list)

        # Identify major chemical classes
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        major_classes = [c[0] for c in sorted_classes[:3]] if sorted_classes else ["Phytochemicals", "Flavonoids"]

        return {
            "total_identified_compounds": len(processed_compounds),
            "major_chemical_classes": major_classes,
            "processed_compounds": processed_compounds,
            "pyg_batch_graph": batch_graph,
            "raw_graph_list": pyg_data_list
        }

    def _classify_molecule(self, mol: Any, name: str = "", smiles: str = "") -> List[str]:
        """
        Classify chemical taxonomy of mol using RDKit SMARTS patterns or string heuristic fallbacks in Korean.
        """
        matched = []
        if HAS_RDKIT and mol is not None:
            for cls_name, smarts_mol in self.tax_smarts.items():
                if mol.HasSubstructMatch(smarts_mol):
                    # Map to Korean
                    kr_name = {
                        "Flavonoids": "플라보노이드 (Flavonoids)",
                        "Biflavonoids": "비플라보노이드 (Biflavonoids)",
                        "Terpenoids": "테르페노이드 (Terpenoids)",
                        "Diterpenes": "디테르펜 (Diterpenes)",
                        "Alkaloids": "알칼로이드 (Alkaloids)",
                        "Phenylpropanoids": "페닐프로파노이드 (Phenylpropanoids)",
                        "Polyphenols": "폴리페놀 (Polyphenols)"
                    }.get(cls_name, cls_name)
                    matched.append(kr_name)

        if not matched:
            name_lower = name.lower()
            if "flav" in name_lower or "etin" in name_lower or "tin" in name_lower or "퀘르세틴" in name or "빌로베틴" in name:
                matched.append("플라보노이드 (Flavonoids)")
            elif "curcumin" in name_lower or "커큐민" in name:
                matched.append("폴리페놀 (Polyphenols)")
            elif "ginkgolide" in name_lower or "징골라이드" in name:
                matched.append("디테르펜 (Diterpenes)")
            else:
                matched.append("식물유래 화합물 (Phytochemicals)")
        return matched

    def _embed_3d_conformer(self, mol: Any) -> bool:
        """
        Generate 3D conformer using RDKit ETKDG algorithm.
        """
        if not HAS_RDKIT or mol is None or AllChem is None:
            return False
        try:
            mol_with_h = Chem.AddHs(mol)
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            # [버그 수정] 현재 RDKit 버전은 ETKDGv3 params에 'maxAttempts' 속성이
            # 없음(과거 API, 지금은 'maxIterations'로 바뀜) -> AttributeError가
            # try/except에 조용히 삼켜져서 3D conformer 생성이 항상 실패하고
            # 있었음. 버전별로 속성명이 다를 수 있어 hasattr로 방어적으로 처리함.
            if hasattr(params, "maxIterations"):
                params.maxIterations = 50
            elif hasattr(params, "maxAttempts"):
                params.maxAttempts = 50
            res = AllChem.EmbedMolecule(mol_with_h, params)
            if res == 0:
                AllChem.MMFFOptimizeMolecule(mol_with_h, maxIters=50)
                return True
        except Exception as e:
            logger.debug(f"ETKDG 3D embedding failed: {e}")
        return False

    def _mol_to_pyg_graph(self, mol: Any) -> Optional[Any]:
        """
        Convert RDKit Mol object into a PyTorch Geometric Data object with node and edge features.
        """
        if not HAS_RDKIT or not HAS_TORCH or mol is None:
            return None
        try:
            # Node features: atomic number, degree, aromaticity, hybridisation
            node_feats = []
            for atom in mol.GetAtoms():
                feats = [
                    float(atom.GetAtomicNum()),
                    float(atom.GetTotalDegree()),
                    1.0 if atom.GetIsAromatic() else 0.0,
                    float(atom.GetFormalCharge())
                ]
                node_feats.append(feats)

            x = torch.tensor(node_feats, dtype=torch.float)

            # Edge index
            edge_indices = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                edge_indices.append([i, j])
                edge_indices.append([j, i])

            if edge_indices:
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)

            if HAS_PYG:
                return Data(x=x, edge_index=edge_index)
            else:
                return {"x": x, "edge_index": edge_index}
        except Exception as e:
            logger.warning(f"Error converting molecule to PyG graph: {e}")
            return None
