"""Testes do Módulo 2. """
import contextlib
import io
import random
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dynamic-programming"))
import rotas_acessiveis as ra  # noqa: E402


class TestPerfis(unittest.TestCase):
    def setUp(self):
        self.g = ra.grafo_demo_offline()
        self.rot = ra.RoteadorAcessivel(self.g)

    def test_cada_perfil_escolhe_um_corredor(self):
        self.assertIn(11, self.rot.rota(1, 10, "padrao").caminho)              # atalho com escada
        self.assertIn(2, self.rot.rota(1, 10, "cadeirante").caminho)           # guias rebaixadas
        self.assertIn(6, self.rot.rota(1, 10, "deficiencia_visual").caminho)   # piso tátil

    def test_cadeirante_nunca_usa_escada(self):
        r = self.rot.rota(1, 10, "cadeirante")
        self.assertEqual(r.diagnostico()["escadas"], 0)

    def test_escada_intransponivel_sem_alternativa(self):
        g = ra.GrafoUrbano()
        g.adicionar_no(1, 0, 0)
        g.adicionar_no(2, 0, 0.001)
        g.adicionar_trecho(1, 2, highway="steps")
        self.assertFalse(ra.RoteadorAcessivel(g).rota(1, 2, "cadeirante").encontrada)
        self.assertTrue(ra.RoteadorAcessivel(g).rota(1, 2, "padrao").encontrada)

    def test_perfil_invalido(self):
        with self.assertRaises(ValueError):
            self.rot.rota(1, 10, "outro")


class TestMemoizacao(unittest.TestCase):
    def _grade(self, n=12, semente=3):
        rnd = random.Random(semente)
        g = ra.GrafoUrbano()
        for i in range(n):
            for j in range(n):
                g.adicionar_no(i * n + j, -23.56 + i * 1e-4, -46.65 + j * 1e-4)
        opcoes = [{}, {"highway": "steps"}, {"footway": "crossing", "kerb": "raised"},
                  {"footway": "crossing", "tactile_paving": "yes"}, {"surface": "sett"}]
        for i in range(n):
            for j in range(n):
                if j + 1 < n:
                    g.adicionar_trecho(i * n + j, i * n + j + 1, **rnd.choice(opcoes))
                if i + 1 < n:
                    g.adicionar_trecho(i * n + j, (i + 1) * n + j, **rnd.choice(opcoes))
        return g, n * n

    def test_memo_nao_altera_resultado(self):
        g, v = self._grade()
        com, sem = ra.RoteadorAcessivel(g, True), ra.RoteadorAcessivel(g, False)
        rnd = random.Random(0)
        for _ in range(200):
            o, d, p = rnd.randrange(4), rnd.randrange(v), rnd.choice(ra.PERFIS)
            self.assertAlmostEqual(com.rota(o, d, p).custo_ponderado, sem.rota(o, d, p).custo_ponderado)
        self.assertGreater(com.stats["consultas_do_memo"], 0)
        self.assertLess(com.stats["nos_expandidos"], sem.stats["nos_expandidos"])

    def test_consulta_repetida_vem_do_cache(self):
        rot = ra.RoteadorAcessivel(ra.grafo_demo_offline())
        self.assertFalse(rot.rota(1, 10, "padrao").veio_do_cache)
        self.assertTrue(rot.rota(1, 10, "padrao").veio_do_cache)

    def test_alterar_grafo_invalida_memo(self):
        g = ra.grafo_demo_offline()
        rot = ra.RoteadorAcessivel(g)
        antes = rot.rota(1, 10, "cadeirante").custo_ponderado
        g.adicionar_trecho(1, 10, 100, highway="footway")   # novo atalho acessível
        self.assertLess(rot.rota(1, 10, "cadeirante").custo_ponderado, antes)


    def test_estados_obsoletos_sao_descartados(self):
        g = ra.grafo_demo_offline()
        rot = ra.RoteadorAcessivel(g)
        rot.rota(1, 10, "padrao")
        g.adicionar_trecho(1, 10, 100, highway="footway")
        rot.rota(1, 10, "padrao")
        self.assertEqual(len(rot.memo_estado), 1)           # só o estado da versão atual


class TestGrafo(unittest.TestCase):
    def test_snapping_com_candidatos_vazio_nao_cai_no_grafo_inteiro(self):
        g = ra.grafo_demo_offline()
        with self.assertRaises(ValueError):                 # min() de conjunto vazio
            g.no_mais_proximo(-23.56, -46.65, candidatos=set())

    def test_snapping_escolhe_no_mais_proximo(self):
        g = ra.grafo_demo_offline()
        self.assertEqual(g.no_mais_proximo(-23.5614, -46.6559), 1)    # MASP


class TestOverpass(unittest.TestCase):
    def test_guia_vem_dos_nos_da_travessia(self):
        dados = {"elements": [
            {"type": "node", "id": 1, "lat": 0, "lon": 0, "tags": {"kerb": "lowered"}},
            {"type": "node", "id": 2, "lat": 0, "lon": 0.0002, "tags": {"kerb": "raised"}},
            {"type": "way", "id": 9, "nodes": [1, 2], "tags": {"highway": "footway", "footway": "crossing"}},
            {"type": "way", "id": 8, "nodes": [1, 2], "tags": {"highway": "service", "access": "private"}},
        ]}
        g = ra.grafo_de_overpass(dados)
        self.assertEqual(g.n_arestas, 2)                       # via privada ignorada
        self.assertEqual(g.adj[1][0].tag("kerb"), "raised")    # pior caso entre as pontas


RESPOSTA_MASP = [{"lat": "-23.5614", "lon": "-46.6559", "display_name": "MASP, Avenida Paulista, São Paulo"}]


class TestGeocodificacao(unittest.TestCase):
    def test_parametros_da_consulta(self):
        par = ra.montar_parametros_nominatim("Av. Paulista, 1578")
        self.assertEqual(par["q"], "Av. Paulista, 1578")
        self.assertEqual(par["limit"], 1)
        self.assertEqual(par["countrycodes"], "br")
        w, n, e, s = (float(v) for v in par["viewbox"].split(","))   # esquerda, topo, direita, base
        self.assertLess(w, e)
        self.assertGreater(n, s)

    def test_interpreta_resposta_valida(self):
        self.assertEqual(ra.interpretar_resposta_nominatim(RESPOSTA_MASP),
                         (-23.5614, -46.6559, "MASP, Avenida Paulista, São Paulo"))

    def test_resposta_vazia_ou_malformada_vira_erro(self):
        for ruim in ([], None, "texto", [{"lat": "x", "lon": "y"}], [{}]):
            with self.assertRaises(ra.ErroGeocodificacao):
                ra.interpretar_resposta_nominatim(ruim)

    def test_geocodificar_com_servico_simulado(self):
        chamadas = []

        def falso(params):
            chamadas.append(params)
            return RESPOSTA_MASP

        lat, lon, nome = ra.geocodificar("  MASP, São Paulo  ", consulta=falso)
        self.assertEqual((lat, lon), (-23.5614, -46.6559))
        self.assertEqual(chamadas[0]["q"], "MASP, São Paulo")           # espaços removidos

    def test_falha_de_rede_vira_erro_amigavel(self):
        def sem_rede(params):
            raise ConnectionError("sem internet")

        with self.assertRaises(ra.ErroGeocodificacao) as ctx:
            ra.geocodificar("Av. Paulista, 1578", consulta=sem_rede)
        self.assertIn("latitude e longitude", str(ctx.exception))

    def test_endereco_curto_ou_vazio_nao_consulta_o_servico(self):
        def nao_deveria_ser_chamado(params):
            raise AssertionError("não deveria consultar")

        for ruim in ("", "  ", "ab", None):
            with self.assertRaises(ra.ErroGeocodificacao):
                ra.geocodificar(ruim, consulta=nao_deveria_ser_chamado)


class TestResolverPonto(unittest.TestCase):
    PADRAO = ("Padrão", -23.0, -46.0)

    def test_coordenadas_tem_prioridade(self):
        r = ra.resolver_ponto(self.PADRAO, [-23.5, -46.6], "qualquer", geocoder=lambda e: self.fail("não deveria geocodificar"))
        self.assertEqual(r[1:], (-23.5, -46.6))

    def test_endereco_usa_o_geocoder(self):
        r = ra.resolver_ponto(self.PADRAO, None, "MASP", geocoder=lambda e: (-23.56, -46.65, "MASP completo"))
        self.assertEqual(r, ("MASP completo", -23.56, -46.65))

    def test_sem_nada_usa_o_ponto_padrao(self):
        self.assertEqual(ra.resolver_ponto(self.PADRAO, None, None), self.PADRAO)


class TestMainComEndereco(unittest.TestCase):
    def _executar(self, argv, geocoder):
        saida = io.StringIO()
        with mock.patch.object(ra, "geocodificar", geocoder), mock.patch.object(ra.time, "sleep"), \
                contextlib.redirect_stdout(saida):
            codigo = ra.main(argv)
        return codigo, saida.getvalue()

    def test_rota_entre_dois_enderecos(self):
        pontos = {"origem": (-23.5614, -46.6559, "MASP"), "destino": (-23.5685, -46.6485, "Brigadeiro")}
        fila = iter([pontos["origem"], pontos["destino"]])
        codigo, texto = self._executar(["--demo", "--origem-endereco", "MASP", "--destino-endereco", "Brigadeiro",
                                        "--consultas", "5"], lambda e: next(fila))
        self.assertEqual(codigo, 0)
        self.assertIn("Origem : MASP", texto)
        self.assertIn("Destino: Brigadeiro", texto)
        self.assertIn("Cadeirante", texto)

    def test_endereco_nao_encontrado_retorna_2_sem_traceback(self):
        def nao_acha(e):
            raise ra.ErroGeocodificacao("endereço não encontrado")

        codigo, texto = self._executar(["--demo", "--origem-endereco", "Rua Inexistente 999"], nao_acha)
        self.assertEqual(codigo, 2)
        self.assertIn("Erro: endereço não encontrado", texto)

    def test_ponto_longe_da_malha_gera_aviso(self):
        longe = (-23.65, -46.75, "Longe")
        codigo, texto = self._executar(["--demo", "--origem-endereco", "Longe", "--consultas", "5"], lambda e: longe)
        self.assertEqual(codigo, 0)
        self.assertIn("fora da área coberta", texto)

    def test_coordenadas_e_endereco_juntos_sao_rejeitados(self):
        with self.assertRaises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                ra.main(["--demo", "--origem", "-23.5", "-46.6", "--origem-endereco", "MASP"])


if __name__ == "__main__":
    unittest.main()
