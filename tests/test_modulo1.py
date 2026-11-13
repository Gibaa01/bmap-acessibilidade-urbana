"""Testes do Módulo 1 (sem rede). Rodar: python -m unittest discover -s tests -v

Os dicionários abaixo imitam o FORMATO da resposta da Overpass; são fixtures
de teste, não dados reais, e não são usados na análise."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modulo1_data_science"))
import coletar_dados_reais as coleta  # noqa: E402
import features_acessibilidade as fa  # noqa: E402


def _elementos():
    return [
        {"type": "node", "id": 1, "lat": -23.5614, "lon": -46.6559,
         "tags": {"amenity": "hospital", "name": "H #1", "wheelchair": "yes",
                  "toilets:wheelchair": "designated", "tactile_paving": "yes", "phone": "x"}},
        {"type": "way", "id": 2, "center": {"lat": -23.57, "lon": -46.64},
         "tags": {"amenity": "doctors", "name": "Dr. Fulano", "wheelchair": "unknown"}},
        {"type": "node", "id": 3, "lat": -23.56, "lon": -46.65, "tags": {"shop": "bakery"}},
        {"type": "node", "id": 4, "lat": -23.56, "lon": -46.65, "tags": {"healthcare": "laboratory"}},
    ]


class TestColeta(unittest.TestCase):
    def setUp(self):
        self.df = coleta.processar_estabelecimentos(_elementos())

    def test_filtra_categorias_de_interesse(self):
        self.assertEqual(len(self.df), 3)                       # padaria descartada
        self.assertEqual(set(self.df["categoria"]), {"hospital", "clinica"})

    def test_ausencia_vira_nao_informado(self):
        dr = self.df[self.df["osm_id"] == 2].iloc[0]
        self.assertEqual(dr["acesso_cadeirante_status"], "nao_informado")   # "unknown"
        self.assertEqual(dr["acesso_cadeirante_auditado"], 0)
        self.assertEqual(dr["informacao_acessibilidade"], "nao_informada")
        self.assertTrue(pd.isna(dr["toilets:wheelchair"]))                # bruto preservado como NaN

    def test_lgpd(self):
        dr = self.df[self.df["osm_id"] == 2].iloc[0]
        self.assertNotIn("Fulano", str(dr["name"]))
        self.assertNotIn("phone", self.df.columns)
        self.assertEqual(self.df[self.df["osm_id"] == 1].iloc[0]["tem_contato"], 1)

    def test_normalizacao(self):
        self.assertEqual(coleta.normalizar_valor("Designated"), "sim")
        self.assertEqual(coleta.normalizar_valor("limited;yes"), "limitado")
        self.assertEqual(coleta.normalizar_valor(np.nan), "nao_informado")

    def test_csv_com_metadados_ida_e_volta(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            caminho = Path(tmp) / "x.csv"
            fa.salvar_csv_com_metadados(self.df, caminho, {"fonte": "teste", "licenca": "ODbL"})
            df2, meta = fa.ler_csv_com_metadados(caminho)
            self.assertEqual(meta["fonte"], "teste")
            self.assertEqual(len(df2), len(self.df))
            self.assertIn("H #1", set(df2["name"]))                  # '#' no nome não quebra a leitura


class TestIndice(unittest.TestCase):
    def setUp(self):
        df = coleta.processar_estabelecimentos(_elementos())
        transp = pd.DataFrame({"latitude": [-23.5615], "longitude": [-46.6559], "acessivel_status": ["sim"]})
        self.df = fa.calcular_indice_infraestrutura(coleta.adicionar_proximidade_transporte(df, transp))

    def test_limites(self):
        self.assertTrue(self.df["indice_infraestrutura_inclusiva"].between(0, 100).all())

    def test_hospital_completo(self):
        h = self.df[self.df["osm_id"] == 1].iloc[0]
        # 30 entrada + 10 banheiro + 10 piso tátil + ~25 transporte (11 m)
        self.assertGreater(h["indice_infraestrutura_inclusiva"], 70)
        self.assertEqual(int(fa.calcular_apto_atendimento_autonomo(self.df).iloc[0]), 1)

    def test_bloqueio_de_vazamento(self):
        fa.verificar_sem_vazamento(fa.FEATURES_MODELO)
        with self.assertRaises(ValueError):
            fa.verificar_sem_vazamento(fa.FEATURES_MODELO + ["acesso_cadeirante_status"])


if __name__ == "__main__":
    unittest.main()
