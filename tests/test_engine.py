"""
Verification Test Suite (tests/test_engine.py)
-----------------------------------------------
Tests all 4 modules, binomial names, tissue parts, and FastAPI schema compliance.
"""

import sys
import unittest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from miners.lit_miner import LitChemMiner
from pipeline.extract_twin import ExtractProfileTwin
from models.gnn_predictor import PanRNAHostPathogenGNN
from models.causal_moa import CausalMOASynergyEngine
from pipeline.orchestrator import LitPhytoPanRNAEngine


class TestLitPhytoPanRNAEngine(unittest.TestCase):

    def setUp(self):
        self.engine = LitPhytoPanRNAEngine(use_live_api=False)

    def test_module1_lit_miner(self):
        miner = LitChemMiner(use_live_api=False)
        compounds = miner.mine_plant_compounds("Ginkgo biloba", "H1N1", "Leaves")
        self.assertGreater(len(compounds), 0)
        first = compounds[0]
        self.assertIn("compound_id", first)
        self.assertIn("smiles", first)
        self.assertIn("citations", first)
        self.assertTrue(first["citations"][0]["url"].startswith("https://"))

    def test_module2_extract_twin(self):
        miner = LitChemMiner(use_live_api=False)
        twin_builder = ExtractProfileTwin()
        compounds = miner.mine_plant_compounds("Ginkgo biloba", "H1N1", "Leaves")
        twin = twin_builder.build_extract_twin(compounds)
        self.assertIn("total_identified_compounds", twin)
        self.assertIn("major_chemical_classes", twin)

    def test_module3_gnn_predictor(self):
        miner = LitChemMiner(use_live_api=False)
        twin_builder = ExtractProfileTwin()
        gnn = PanRNAHostPathogenGNN()

        compounds = miner.mine_plant_compounds("Ginkgo biloba", "H1N1", "Leaves")
        twin = twin_builder.build_extract_twin(compounds)
        leads, host_affinities = gnn.predict_leads_and_affinities(twin)

        self.assertGreater(len(leads), 0)
        self.assertLess(leads[0]["h1n1_pa_binding_affinity_kcal_mol"], 0)
        self.assertIn("lifecycle_affinities", leads[0])

    def test_module4_causal_moa(self):
        miner = LitChemMiner(use_live_api=False)
        twin_builder = ExtractProfileTwin()
        gnn = PanRNAHostPathogenGNN()
        causal = CausalMOASynergyEngine()

        compounds = miner.mine_plant_compounds("Ginkgo biloba", "H1N1", "Leaves")
        twin = twin_builder.build_extract_twin(compounds)
        leads, _ = gnn.predict_leads_and_affinities(twin)
        moa, graph = causal.analyze_causal_moa("Ginkgo biloba", leads, "H1N1")

        self.assertIn("moa_title", moa)
        self.assertIn("synergy_score", moa)
        self.assertGreater(graph.number_of_nodes(), 0)

    def test_full_pipeline_output_schema(self):
        result = self.engine.run_pipeline("Ginkgo biloba", "H1N1", "Leaves")

        self.assertEqual(result["query_resource"], "Ginkgo biloba")
        self.assertEqual(result["extract_part"], "Leaves")
        self.assertEqual(result["target_virus"], "H1N1")
        self.assertEqual(result["status"], "SUCCESS")
        self.assertIn("virtual_profile_summary", result)
        self.assertIn("predicted_leads", result)
        self.assertIn("discovered_moa", result)

        lead = result["predicted_leads"][0]
        self.assertIn("compound_name", lead)
        self.assertIn("h1n1_pa_binding_affinity_kcal_mol", lead)
        self.assertIn("pan_rna_host_target_affinity", lead)
        self.assertIn("citations", lead)


if __name__ == "__main__":
    unittest.main()
