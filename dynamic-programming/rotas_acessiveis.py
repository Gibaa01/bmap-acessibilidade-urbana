"""
rotas_acessiveis.py
===================

Módulo 2 · Dynamic Programming — rota mais ACESSÍVEL (não a mais curta) entre
dois pontos de um mesmo bairro, para três perfis de cidadão:
"padrao", "cadeirante" e "deficiencia_visual".

Técnica: grafo ponderado + Dijkstra adaptado (custo dinâmico por perfil) +
memoização em dois níveis (custo de aresta e estado do Dijkstra por origem).

FONTE DA MALHA
--------------
© OpenStreetMap contributors — Open Database License (ODbL) 1.0
https://www.openstreetmap.org/copyright — obtida via Overpass API.
Tags OSM usadas: highway, footway, sidewalk, kerb, tactile_paving, surface,
smoothness, incline, width, handrail, wheelchair, ramp:wheelchair,
traffic_signals:sound, crossing.
"""
from __future__ import annotations

import heapq
import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

INF = float("inf")


# --------------------------------------------------------------------------- #
# Utilidades geográficas
# --------------------------------------------------------------------------- #

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6_371_008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# --------------------------------------------------------------------------- #
# Estrutura do grafo
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Aresta:
    """Trecho caminhável entre dois nós. `atributos` guarda as tags OSM
    relevantes (já combinadas da via e dos nós das extremidades)."""
    id: int
    origem: int
    destino: int
    distancia_m: float
    atributos: Tuple[Tuple[str, str], ...]  # tupla ordenada → imutável e "hashável"

    def tag(self, chave: str, padrao: Optional[str] = None) -> Optional[str]:
        for k, v in self.atributos:
            if k == chave:
                return v
        return padrao


class GrafoUrbano:
    """Lista de adjacência: nó -> [arestas de saída]. Pedestres andam nos dois
    sentidos, então cada trecho vira duas arestas dirigidas (com ids distintos)."""

    def __init__(self, fonte: str = "desconhecida"):
        self.fonte = fonte
        self.nos: Dict[int, Tuple[float, float]] = {}
        self.rotulos: Dict[int, str] = {}
        self.adj: Dict[int, List[Aresta]] = {}
        self.versao = 0            # muda a cada alteração → invalida memos
        self._proximo_id = 0

    # -- construção ---------------------------------------------------------
    def adicionar_no(self, no: int, lat: float, lon: float, rotulo: Optional[str] = None) -> None:
        self.nos[no] = (lat, lon)
        self.adj.setdefault(no, [])
        if rotulo:
            self.rotulos[no] = rotulo
        self.versao += 1

    def adicionar_trecho(self, u: int, v: int, distancia_m: Optional[float] = None,
                         bidirecional: bool = True, **atributos: str) -> None:
        if u not in self.nos or v not in self.nos:
            raise KeyError(f"nó inexistente no trecho {u}-{v}")
        if distancia_m is None:
            distancia_m = haversine_m(*self.nos[u], *self.nos[v])
        attrs = tuple(sorted((k, str(val)) for k, val in atributos.items() if val is not None))
        pares = [(u, v), (v, u)] if bidirecional else [(u, v)]
        for a, b in pares:
            self.adj[a].append(Aresta(self._proximo_id, a, b, float(distancia_m), attrs))
            self._proximo_id += 1
        self.versao += 1

    # -- consultas ----------------------------------------------------------
    @property
    def n_nos(self) -> int:
        return len(self.nos)

    @property
    def n_arestas(self) -> int:
        return sum(len(v) for v in self.adj.values())

    def nome(self, no: int) -> str:
        return self.rotulos.get(no, f"nó {no}")

    def maior_componente(self) -> Set[int]:
        """Maior componente conexa (BFS). Evita "encaixar" a origem em um
        pedaço de calçada isolado do resto da malha. O(V + E)."""
        visitados: Set[int] = set()
        maior: Set[int] = set()
        for inicio in self.nos:
            if inicio in visitados:
                continue
            comp, fila = {inicio}, [inicio]
            visitados.add(inicio)
            while fila:
                atual = fila.pop()
                for ar in self.adj[atual]:
                    if ar.destino not in visitados:
                        visitados.add(ar.destino)
                        comp.add(ar.destino)
                        fila.append(ar.destino)
            if len(comp) > len(maior):
                maior = comp
        return maior

    def no_mais_proximo(self, lat: float, lon: float, candidatos: Optional[Set[int]] = None) -> int:
        """Encaixe (snapping) de uma coordenada no nó mais próximo. O(V)."""
        candidatos = self.nos if candidatos is None else candidatos
        return min(candidatos, key=lambda n: haversine_m(lat, lon, *self.nos[n]))


# --------------------------------------------------------------------------- #
# Perfis e modelo de custo dinâmico
# --------------------------------------------------------------------------- #
# Custo de uma aresta = distância × fator multiplicativo + penalidades fixas,
# expresso em "metros equivalentes de esforço". INF = intransponível.
# Todos os custos são ≥ 0 — requisito para a correção do Dijkstra.
#
# Princípio para dados ausentes: "desconhecido" é penalizado MENOS do que
# "barreira confirmada", mas MAIS do que "acessível confirmado" — a rota prefere
# trechos verificados sem descartar a malha não mapeada.

PERFIS = ("padrao", "cadeirante", "deficiencia_visual")

VIAS_DE_CARRO = {"primary", "primary_link", "secondary", "secondary_link", "tertiary",
                 "tertiary_link", "residential", "unclassified", "service", "living_street"}
SUPERFICIES_RUINS = {"sett", "cobblestone", "unhewn_cobblestone", "gravel", "fine_gravel",
                     "unpaved", "dirt", "ground", "grass", "sand", "mud", "pebblestone"}
LISURA_RUIM = {"bad", "very_bad", "horrible", "very_horrible", "impassable"}


@dataclass(frozen=True)
class ParametrosCusto:
    # cadeirante
    penal_guia_alta_m: float = 300.0        # travessia com guia alta (kerb=raised)
    penal_guia_indefinida_m: float = 60.0   # travessia sem informação de guia
    penal_guia_generica_m: float = 120.0    # kerb=yes / rolled (há degrau, altura incerta)
    fator_superficie_ruim_cad: float = 1.6
    fator_lisura_ruim_cad: float = 1.8
    fator_rampa_ingreme: float = 2.5        # inclinação > 8,33% (limite NBR 9050)
    fator_rampa_moderada: float = 1.5       # 5% a 8,33%
    fator_calcada_estreita: float = 1.5     # largura < 1,20 m (faixa livre NBR 9050)
    fator_sem_calcada_cad: float = 3.0
    fator_escada_com_rampa: float = 1.5
    # deficiência visual
    penal_travessia_sem_tatil_m: float = 150.0
    penal_travessia_tatil_indef_m: float = 80.0
    fator_tatil_confirmado: float = 0.90    # bônus (fator < 1, mas > 0 → custo ainda positivo)
    fator_escada_sem_corrimao: float = 1.5
    fator_escada_corrimao_indef: float = 1.2
    fator_sem_calcada_visual: float = 2.5
    fator_superficie_ruim_visual: float = 1.2
    # padrão
    fator_escada_padrao: float = 1.1


def _inclinacao_pct(valor: Optional[str]) -> Optional[float]:
    if not valor:
        return None
    m = re.search(r"-?\d+(?:[.,]\d+)?", valor)
    if m:
        num = abs(float(m.group().replace(",", ".")))
        return num if "%" in valor or num > 1 else num * 100
    return 6.0 if valor in ("up", "down") else None   # direção sem valor: assume moderada


def _eh_travessia(a: Aresta) -> bool:
    return a.tag("footway") == "crossing" or a.tag("highway") == "crossing" or a.tag("crossing") is not None


def custo_aresta(a: Aresta, perfil: str, p: ParametrosCusto = ParametrosCusto()) -> float:
    """Função de custo PURA (depende só da aresta e do perfil) → memoizável."""
    d = a.distancia_m
    highway = a.tag("highway")
    if a.tag("foot") == "no":
        return INF

    if perfil == "padrao":
        return d * (p.fator_escada_padrao if highway == "steps" else 1.0)

    travessia = _eh_travessia(a)
    sem_calcada = highway in VIAS_DE_CARRO and a.tag("sidewalk") in ("no", "none")

    if perfil == "cadeirante":
        if a.tag("wheelchair") == "no":
            return INF
        fator, penal = 1.0, 0.0
        if highway == "steps":
            if a.tag("ramp:wheelchair") == "yes" or a.tag("ramp") == "yes":
                fator *= p.fator_escada_com_rampa
            else:
                return INF                               # escada sem rampa: intransponível
        if travessia:
            kerb = a.tag("kerb")
            if kerb == "raised":
                penal += p.penal_guia_alta_m             # sem rebaixamento de guia
            elif kerb in ("yes", "rolled"):
                penal += p.penal_guia_generica_m
            elif kerb not in ("lowered", "flush", "no"):
                penal += p.penal_guia_indefinida_m
        if a.tag("surface") in SUPERFICIES_RUINS:
            fator *= p.fator_superficie_ruim_cad
        if a.tag("smoothness") in LISURA_RUIM:
            fator *= p.fator_lisura_ruim_cad
        inc = _inclinacao_pct(a.tag("incline"))
        if inc is not None:
            fator *= p.fator_rampa_ingreme if inc > 8.33 else (p.fator_rampa_moderada if inc > 5 else 1.0)
        try:
            if float(a.tag("width", "99").replace(",", ".").rstrip(" m")) < 1.2:
                fator *= p.fator_calcada_estreita
        except ValueError:
            pass
        if sem_calcada:
            fator *= p.fator_sem_calcada_cad
        return d * fator + penal

    if perfil == "deficiencia_visual":
        fator, penal = 1.0, 0.0
        tatil = a.tag("tactile_paving")
        if travessia:
            if a.tag("traffic_signals:sound") == "yes":
                pass                                     # semáforo sonoro compensa
            elif tatil == "no":
                penal += p.penal_travessia_sem_tatil_m   # travessia sem piso tátil
            elif tatil not in ("yes", "contrasted"):
                penal += p.penal_travessia_tatil_indef_m
        if tatil in ("yes", "contrasted"):
            fator *= p.fator_tatil_confirmado
        if highway == "steps":
            corrimao = a.tag("handrail")
            fator *= (1.0 if corrimao == "yes" else
                      p.fator_escada_sem_corrimao if corrimao == "no" else p.fator_escada_corrimao_indef)
        if a.tag("surface") in SUPERFICIES_RUINS:
            fator *= p.fator_superficie_ruim_visual
        if sem_calcada:
            fator *= p.fator_sem_calcada_visual
        return d * fator + penal

    raise ValueError(f"perfil desconhecido: {perfil!r} (use um de {PERFIS})")


# --------------------------------------------------------------------------- #
# Dijkstra adaptado + memoização (Programação Dinâmica)
# --------------------------------------------------------------------------- #
# Por que isso é Programação Dinâmica?
#   Subestrutura ótima (princípio de Bellman): se o melhor caminho s→t passa por
#   u, o trecho s→u também é ótimo. Logo
#       dist[v] = min_{(u,v) ∈ E} ( dist[u] + custo(u, v, perfil) )
#   O Dijkstra resolve essa recorrência na ordem certa (do menor dist para o
#   maior) e cada subproblema dist[u] é resolvido UMA vez.
#
# Memoização em dois níveis:
#   1. memo_custo[(id_aresta, perfil)] → custo da aresta (função pura).
#   2. memo_estado[(origem, perfil, versao_grafo)] → estado do Dijkstra (dist,
#      pred, fila de prioridade, finalizados). É um Dijkstra "retomável":
#        • destino já finalizado → resposta em O(k) (só reconstrói o caminho);
#        • destino ainda não alcançado → continua de onde parou, sem refazer
#          nenhum subproblema já resolvido.
#   A versão do grafo entra na chave: se a malha mudar, o memo antigo é ignorado.

@dataclass
class EstadoDijkstra:
    origem: int
    perfil: str
    dist: Dict[int, float]
    pred: Dict[int, Aresta]
    fila: List[Tuple[float, int]]
    finalizados: Set[int] = field(default_factory=set)


@dataclass
class ResultadoRota:
    perfil: str
    origem: int
    destino: int
    caminho: List[int]
    arestas: List[Aresta]
    custo_ponderado: float
    distancia_m: float
    tempo_ms: float
    veio_do_cache: bool

    @property
    def encontrada(self) -> bool:
        return math.isfinite(self.custo_ponderado)

    def diagnostico(self) -> Dict[str, int]:
        """Barreiras e incertezas presentes na rota escolhida (para o relatório)."""
        c = {"escadas": 0, "travessias": 0, "guia_alta": 0, "guia_sem_info": 0,
             "travessia_sem_tatil": 0, "travessia_tatil_sem_info": 0, "semaforo_sonoro": 0}
        for a in self.arestas:
            if a.tag("highway") == "steps":
                c["escadas"] += 1
            if _eh_travessia(a):
                c["travessias"] += 1
                kerb, tatil = a.tag("kerb"), a.tag("tactile_paving")
                c["guia_alta"] += kerb == "raised"
                c["guia_sem_info"] += kerb is None
                c["travessia_sem_tatil"] += tatil == "no"
                c["travessia_tatil_sem_info"] += tatil is None
                c["semaforo_sonoro"] += a.tag("traffic_signals:sound") == "yes"
        return c


class RoteadorAcessivel:
    def __init__(self, grafo: GrafoUrbano, usar_memo: bool = True,
                 parametros: ParametrosCusto = ParametrosCusto()):
        self.grafo = grafo
        self.usar_memo = usar_memo
        self.parametros = parametros
        self.memo_custo: Dict[Tuple[int, str], float] = {}
        self.memo_estado: Dict[Tuple[int, str, int], EstadoDijkstra] = {}
        self._versao_memo = grafo.versao
        self.stats = {"custos_calculados": 0, "custos_do_memo": 0,
                      "nos_expandidos": 0, "consultas": 0, "consultas_do_memo": 0}

    def limpar_memo(self) -> None:
        self.memo_custo.clear()
        self.memo_estado.clear()

    # -- nível 1: custo de aresta ---------------------------------------------
    def custo(self, a: Aresta, perfil: str) -> float:
        if self.usar_memo:
            chave = (a.id, perfil)
            valor = self.memo_custo.get(chave)
            if valor is not None:
                self.stats["custos_do_memo"] += 1
                return valor
            valor = custo_aresta(a, perfil, self.parametros)
            self.memo_custo[chave] = valor
        else:
            valor = custo_aresta(a, perfil, self.parametros)
        self.stats["custos_calculados"] += 1
        return valor

    # -- nível 2: estado do Dijkstra por (origem, perfil) -----------------------
    def _estado(self, origem: int, perfil: str) -> Tuple[EstadoDijkstra, bool]:
        if self.grafo.versao != self._versao_memo:
            # A malha mudou: estados antigos nunca mais serão consultados (a versão
            # está na chave). Descartá-los evita vazamento de memória. Os custos de
            # aresta continuam válidos: Aresta é imutável e seu id nunca é reutilizado.
            self.memo_estado.clear()
            self._versao_memo = self.grafo.versao
        chave = (origem, perfil, self.grafo.versao)
        if self.usar_memo and chave in self.memo_estado:
            return self.memo_estado[chave], True
        estado = EstadoDijkstra(origem, perfil, dist={origem: 0.0}, pred={}, fila=[(0.0, origem)])
        if self.usar_memo:
            self.memo_estado[chave] = estado
        return estado, False

    def _expandir_ate(self, est: EstadoDijkstra, destino: int) -> None:
        """Dijkstra com heap binária e parada antecipada; retomável."""
        while est.fila and destino not in est.finalizados:
            d_u, u = heapq.heappop(est.fila)
            if u in est.finalizados or d_u > est.dist.get(u, INF):
                continue                      # entrada obsoleta da heap (lazy deletion)
            est.finalizados.add(u)
            self.stats["nos_expandidos"] += 1
            for a in self.grafo.adj.get(u, ()):
                if a.destino in est.finalizados:
                    continue
                c = self.custo(a, est.perfil)
                if c == INF:
                    continue                  # trecho intransponível para este perfil
                novo = d_u + c
                if novo < est.dist.get(a.destino, INF):   # relaxamento (recorrência de Bellman)
                    est.dist[a.destino] = novo
                    est.pred[a.destino] = a
                    heapq.heappush(est.fila, (novo, a.destino))

    def rota(self, origem: int, destino: int, perfil: str = "padrao") -> ResultadoRota:
        if perfil not in PERFIS:
            raise ValueError(f"perfil desconhecido: {perfil!r}")
        t0 = time.perf_counter()
        self.stats["consultas"] += 1
        est, reutilizado = self._estado(origem, perfil)
        ja_resolvido = destino in est.finalizados
        if not ja_resolvido:
            self._expandir_ate(est, destino)
        if reutilizado and ja_resolvido:
            self.stats["consultas_do_memo"] += 1

        custo_total = est.dist.get(destino, INF) if destino in est.finalizados else INF
        caminho, arestas = [], []
        if math.isfinite(custo_total):
            no = destino
            while no != origem:                # reconstrução O(k) pelos predecessores
                a = est.pred[no]
                arestas.append(a)
                no = a.origem
            arestas.reverse()
            caminho = [origem] + [a.destino for a in arestas]
        return ResultadoRota(
            perfil=perfil, origem=origem, destino=destino, caminho=caminho, arestas=arestas,
            custo_ponderado=custo_total, distancia_m=sum(a.distancia_m for a in arestas),
            tempo_ms=(time.perf_counter() - t0) * 1000, veio_do_cache=reutilizado and ja_resolvido)


# --------------------------------------------------------------------------- #
# Carga da malha: (1) Overpass ao vivo → (2) cache real local → (3) demo offline
# --------------------------------------------------------------------------- #
RAIZ = Path(__file__).resolve().parents[1]
ARQ_CACHE_MALHA = RAIZ / "dados" / "malha_pedestre_bela_vista_osm.json"

# Recorte de Bela Vista ao longo da Av. Paulista (entre Consolação e Brigadeiro)
BBOX_MALHA_PADRAO = (-23.5725, -46.6625, -23.5555, -46.6440)  # (sul, oeste, norte, leste)

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]
VIAS_CAMINHAVEIS = ("footway|pedestrian|path|steps|corridor|living_street|residential|service|"
                    "unclassified|tertiary|tertiary_link|secondary|secondary_link|primary|primary_link")
# Tags da VIA copiadas para a aresta
TAGS_VIA = ("highway", "footway", "sidewalk", "surface", "smoothness", "incline", "width",
            "handrail", "wheelchair", "ramp", "ramp:wheelchair", "tactile_paving", "crossing",
            "foot", "kerb")
# Tags de NÓ que valem para a travessia (rebaixamento de guia fica nos nós)
TAGS_NO_TRAVESSIA = ("kerb", "tactile_paving", "traffic_signals:sound")
ORDEM_GUIA = {"raised": 4, "yes": 3, "rolled": 2, "lowered": 1, "flush": 0, "no": 0}


def montar_query_malha(bbox) -> str:
    s, w, n, e = bbox
    return f"""[out:json][timeout:180];
way["highway"~"^({VIAS_CAMINHAVEIS})$"]({s},{w},{n},{e});
(._;>;);
out body;"""


def baixar_malha_overpass(bbox=BBOX_MALHA_PADRAO) -> dict:
    import requests  # import tardio: o modo offline não exige a biblioteca
    query = montar_query_malha(bbox)
    erro = None
    for url in OVERPASS_ENDPOINTS:
        try:
            r = requests.post(url, data={"data": query}, timeout=200,
                              headers={"User-Agent": "CitizenTech-Rotas/1.0 (academico; ODbL)"})
            r.raise_for_status()
            dados = r.json()
            dados["_proveniencia"] = {
                "fonte": "OpenStreetMap contributors, via Overpass API", "endpoint": url,
                "licenca": "ODbL 1.0", "atribuicao": "(c) OpenStreetMap contributors",
                "data_extracao_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "bbox": list(bbox)}
            return dados
        except Exception as e:  # noqa: BLE001 — tenta o próximo endpoint
            erro = e
    raise RuntimeError(f"Overpass indisponível: {erro}")


def grafo_de_overpass(dados: dict) -> GrafoUrbano:
    """Converte a resposta da Overpass em GrafoUrbano. Cada par de nós
    consecutivos de uma via vira um trecho bidirecional."""
    prov = dados.get("_proveniencia", {})
    g = GrafoUrbano(fonte=f"OSM real ({prov.get('data_extracao_utc', 'data n/d')}) - ODbL")
    nos = {el["id"]: el for el in dados["elements"] if el["type"] == "node"}
    for via in (el for el in dados["elements"] if el["type"] == "way"):
        tv = via.get("tags", {})
        if tv.get("foot") == "no" or (tv.get("access") in ("private", "no") and tv.get("foot") != "yes"):
            continue
        refs = [r for r in via.get("nodes", []) if r in nos]
        travessia = tv.get("footway") == "crossing" or "crossing" in tv
        for u, v in zip(refs, refs[1:]):
            for n in (u, v):
                if n not in g.nos:
                    g.adicionar_no(n, nos[n]["lat"], nos[n]["lon"], nos[n].get("tags", {}).get("name"))
            attrs = {k: tv[k] for k in TAGS_VIA if k in tv}
            if travessia:
                # kerb / tactile_paving / semáforo sonoro vêm dos nós das pontas;
                # vale o PIOR caso de guia (ex.: um lado rebaixado e outro alto → alto)
                tags_nos = [nos[u].get("tags", {}), nos[v].get("tags", {})]
                guias = [t["kerb"] for t in tags_nos if "kerb" in t]
                if guias and "kerb" not in attrs:
                    attrs["kerb"] = max(guias, key=lambda k: ORDEM_GUIA.get(k, 2))
                for chave in ("tactile_paving", "traffic_signals:sound"):
                    valores = [t[chave] for t in tags_nos if chave in t]
                    if valores and chave not in attrs:
                        attrs[chave] = "no" if "no" in valores else valores[0]
            g.adicionar_trecho(u, v, **attrs)
    return g


def grafo_demo_offline() -> GrafoUrbano:
    """Subgrafo de DEMONSTRAÇÃO (último recurso, quando não há internet nem cache).

    ⚠ HONESTIDADE DOS DADOS: os pontos de referência (MASP e Estação Brigadeiro,
    Av. Paulista — Bela Vista) e as coordenadas são reais/aproximadas, mas a
    TOPOLOGIA, as DISTÂNCIAS e as TAGS de acessibilidade abaixo são ILUSTRATIVAS,
    montadas para exercitar o algoritmo (três corredores com barreiras diferentes).
    Para o fallback offline com dados REAIS, rode uma vez com internet
    (`--online`): a malha OSM é salva em `dados/` e passa a ser usada offline.
    """
    g = GrafoUrbano(fonte="DEMO ILUSTRATIVA (tags nao sao reais) - substitua por --online")
    pontos = {
        1: ("MASP (origem)", -23.5614, -46.6559),
        2: ("Corredor A - calçada 1", -23.5631, -46.6542),
        3: ("Corredor A - após travessia 1", -23.5633, -46.6540),
        4: ("Corredor A - calçada 2", -23.5657, -46.6514),
        5: ("Corredor A - após travessia 2", -23.5659, -46.6512),
        6: ("Corredor B - calçada 1", -23.5624, -46.6566),
        7: ("Corredor B - após travessia", -23.5626, -46.6564),
        8: ("Corredor B - calçada 2", -23.5660, -46.6526),
        11: ("Corredor C - topo da escadaria", -23.5629, -46.6551),
        12: ("Corredor C - base da escadaria", -23.5632, -46.6549),
        13: ("Corredor C - após travessia", -23.5634, -46.6547),
        10: ("Estação Brigadeiro (destino)", -23.5685, -46.6485),
    }
    for no, (rotulo, lat, lon) in pontos.items():
        g.adicionar_no(no, lat, lon, rotulo)
    calcada = dict(highway="footway", footway="sidewalk")
    # Corredor A: guias rebaixadas, SEM piso tátil (bom p/ cadeirante, ruim p/ visual) — 1040 m
    g.adicionar_trecho(1, 2, 300, **calcada)
    g.adicionar_trecho(2, 3, 20, highway="footway", footway="crossing", kerb="lowered", tactile_paving="no")
    g.adicionar_trecho(3, 4, 400, **calcada)
    g.adicionar_trecho(4, 5, 20, highway="footway", footway="crossing", kerb="lowered", tactile_paving="no")
    g.adicionar_trecho(5, 10, 300, **calcada)
    # Corredor B: piso tátil + semáforo sonoro, mas guia ALTA (bom p/ visual, ruim p/ cadeirante) — 980 m
    g.adicionar_trecho(1, 6, 150, tactile_paving="yes", **calcada)
    g.adicionar_trecho(6, 7, 20, highway="footway", footway="crossing", kerb="raised",
                       tactile_paving="yes", **{"traffic_signals:sound": "yes"})
    g.adicionar_trecho(7, 8, 500, tactile_paving="yes", **calcada)
    g.adicionar_trecho(8, 10, 310, tactile_paving="yes", **calcada)
    # Corredor C: atalho com ESCADARIA e travessia sem piso tátil (mais curto) — 900 m
    g.adicionar_trecho(1, 11, 250, **calcada)
    g.adicionar_trecho(11, 12, 40, highway="steps")
    g.adicionar_trecho(12, 13, 20, highway="footway", footway="crossing", kerb="raised", tactile_paving="no")
    g.adicionar_trecho(13, 10, 590, **calcada)
    return g


def carregar_grafo(modo: str = "auto", bbox=BBOX_MALHA_PADRAO) -> GrafoUrbano:
    """modo: 'online' (baixa e salva cache), 'cache', 'demo' ou 'auto'
    (cache real se existir; senão tenta online; senão demo)."""
    if modo == "demo":
        return grafo_demo_offline()
    if modo in ("cache", "auto") and ARQ_CACHE_MALHA.exists():
        return grafo_de_overpass(json.loads(ARQ_CACHE_MALHA.read_text(encoding="utf-8")))
    if modo == "cache":
        raise FileNotFoundError(f"{ARQ_CACHE_MALHA} não existe; rode com --online uma vez.")
    try:
        dados = baixar_malha_overpass(bbox)
        ARQ_CACHE_MALHA.parent.mkdir(parents=True, exist_ok=True)
        ARQ_CACHE_MALHA.write_text(json.dumps(dados), encoding="utf-8")
        return grafo_de_overpass(dados)
    except Exception as erro:  # noqa: BLE001
        if modo == "online":
            raise
        print(f"⚠ Sem malha real disponível ({erro.__class__.__name__}); usando DEMO ilustrativa.")
        return grafo_demo_offline()


# --------------------------------------------------------------------------- #
# Complexidade assintótica
# --------------------------------------------------------------------------- #
COMPLEXIDADE = """
Complexidade (V = nós, E = arestas dirigidas, P = nº de perfis, k = nós no caminho)
──────────────────────────────────────────────────────────────────────────────────
Dijkstra com heap binária (lazy deletion) .......... O((V + E) log V) tempo | O(V + E) memória
Custo de aresta ...................................... O(1) (nº fixo de tags por aresta)
1ª consulta (origem, perfil) — cache frio ............ O((V + E) log V)
Consulta repetida / destino já finalizado ............ O(1) busca no memo + O(k) reconstrução
Novo destino a partir de origem já em cache .......... só o trabalho que FALTA (retomada);
                                                       no total, cada (origem, perfil) custa no
                                                       máximo O((V + E) log V) somando todas as consultas
Q consultas com O origens distintas .................. O(O·P·(V + E) log V + Q·k)   (com memo)
                                                       vs. O(Q·(V + E) log V)       (sem memo)
Memória dos memos .................................... O(E·P) custos + O(O·P·(V + E)) estados
                                                       (dist/pred: V; heap com lazy deletion: até E entradas)
Encaixe (snapping) da coordenada e maior componente .. O(V) e O(V + E)
"""


# --------------------------------------------------------------------------- #
# Demonstração
# --------------------------------------------------------------------------- #
# Pontos reais do bairro Bela Vista (coordenadas aproximadas, usadas só para
# encaixar no nó mais próximo da malha).
PONTO_ORIGEM = ("MASP — Av. Paulista, 1578", -23.5614, -46.6559)
PONTO_DESTINO = ("Estação Brigadeiro (Metrô Linha 2-Verde)", -23.5685, -46.6485)

ROTULO_PERFIL = {"padrao": "Padrão", "cadeirante": "Cadeirante", "deficiencia_visual": "Def. visual"}


def _trechos(res: ResultadoRota) -> Set[frozenset]:
    return {frozenset((a.origem, a.destino)) for a in res.arestas}


def comparar_perfis(roteador: RoteadorAcessivel, origem: int, destino: int) -> Dict[str, ResultadoRota]:
    g = roteador.grafo
    resultados = {p: roteador.rota(origem, destino, p) for p in PERFIS}
    print(f"\n{'Perfil':<12} {'Distância':>10} {'Custo pond.':>12} {'Trechos':>8}  Barreiras na rota")
    print("─" * 92)
    for p, r in resultados.items():
        if not r.encontrada:
            print(f"{ROTULO_PERFIL[p]:<12} {'—':>10} {'sem rota':>12}")
            continue
        dg = r.diagnostico()
        barreiras = (f"escadas={dg['escadas']} guia_alta={dg['guia_alta']} "
                     f"trav_sem_tátil={dg['travessia_sem_tatil']} sem_info(guia/tátil)="
                     f"{dg['guia_sem_info']}/{dg['travessia_tatil_sem_info']} sonoro={dg['semaforo_sonoro']}")
        print(f"{ROTULO_PERFIL[p]:<12} {r.distancia_m:>8.0f} m {r.custo_ponderado:>12.1f} "
              f"{len(r.arestas):>8}  {barreiras}")

    if g.n_nos <= 50:  # grafo pequeno (demo): mostra o caminho nó a nó
        print()
        for p, r in resultados.items():
            print(f"  {ROTULO_PERFIL[p]:<12}: " + " → ".join(g.nome(n) for n in r.caminho))

    print("\nSobreposição entre rotas (Jaccard dos trechos; 1 = idênticas):")
    for i, p in enumerate(PERFIS):
        for q in PERFIS[i + 1:]:
            a, b = _trechos(resultados[p]), _trechos(resultados[q])
            jac = len(a & b) / len(a | b) if a | b else 1.0
            print(f"  {ROTULO_PERFIL[p]:<12} × {ROTULO_PERFIL[q]:<12} {jac:.2f}")
    return resultados


def benchmark_memo(grafo: GrafoUrbano, n_consultas: int = 300, n_origens: int = 5, semente: int = 7) -> None:
    import random
    rnd = random.Random(semente)
    comp = sorted(grafo.maior_componente())
    origens = rnd.sample(comp, min(n_origens, len(comp)))
    consultas = [(rnd.choice(origens), rnd.choice(comp), rnd.choice(PERFIS)) for _ in range(n_consultas)]

    tempos, roteadores = {}, {}
    for usar in (False, True):
        rot = RoteadorAcessivel(grafo, usar_memo=usar)
        t0 = time.perf_counter()
        custos = [rot.rota(o, d, p).custo_ponderado for o, d, p in consultas]
        tempos[usar] = (time.perf_counter() - t0, custos)
        roteadores[usar] = rot
    assert all((a == b) or abs(a - b) < 1e-6 for a, b in zip(tempos[False][1], tempos[True][1])), \
        "memoização alterou o resultado!"

    s, c = roteadores[False].stats, roteadores[True].stats
    t_sem, t_com = tempos[False][0], tempos[True][0]
    print(f"\n{n_consultas} consultas, {len(origens)} origens distintas × 3 perfis "
          f"(grafo: {grafo.n_nos} nós, {grafo.n_arestas} arestas)")
    print(f"{'':<30}{'sem memo':>14}{'com memo':>14}")
    print(f"{'tempo total':<30}{t_sem * 1000:>11.1f} ms{t_com * 1000:>11.1f} ms")
    print(f"{'nós expandidos':<30}{s['nos_expandidos']:>14,}{c['nos_expandidos']:>14,}")
    print(f"{'custos de aresta calculados':<30}{s['custos_calculados']:>14,}{c['custos_calculados']:>14,}")
    print(f"{'custos lidos do memo':<30}{s['custos_do_memo']:>14,}{c['custos_do_memo']:>14,}")
    print(f"{'consultas resolvidas no memo':<30}{s['consultas_do_memo']:>14,}{c['consultas_do_memo']:>14,}")
    print(f"→ ganho: {t_sem / t_com:.1f}× mais rápido | resultados idênticos ✔")

    # Mesma consulta repetida: cache frio vs. quente
    rot = RoteadorAcessivel(grafo)
    o, d, _ = consultas[0]
    fria = rot.rota(o, d, "cadeirante").tempo_ms
    quente = rot.rota(o, d, "cadeirante").tempo_ms
    print(f"Consulta repetida (cadeirante): 1ª = {fria:.3f} ms | 2ª (memo) = {quente:.4f} ms")


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Rotas acessíveis com Dijkstra + memoização (OSM).")
    grupo = ap.add_mutually_exclusive_group()
    grupo.add_argument("--online", action="store_true", help="baixar a malha real agora (e salvar cache)")
    grupo.add_argument("--cache", action="store_true", help="usar apenas o cache real salvo")
    grupo.add_argument("--demo", action="store_true", help="forçar o subgrafo ilustrativo offline")
    ap.add_argument("--origem", nargs=2, type=float, metavar=("LAT", "LON"))
    ap.add_argument("--destino", nargs=2, type=float, metavar=("LAT", "LON"))
    ap.add_argument("--consultas", type=int, default=300)
    args = ap.parse_args(argv)

    modo = "online" if args.online else "cache" if args.cache else "demo" if args.demo else "auto"
    grafo = carregar_grafo(modo)
    print(f"Malha: {grafo.fonte} | {grafo.n_nos} nós, {grafo.n_arestas} arestas dirigidas")
    if grafo.fonte.startswith("OSM"):
        print("Atribuição: © OpenStreetMap contributors (ODbL 1.0)")
    else:
        print("⚠ Malha ILUSTRATIVA: serve para demonstrar o algoritmo, não descreve calçadas reais.")

    componente = grafo.maior_componente()
    lat_o, lon_o = args.origem or PONTO_ORIGEM[1:]
    lat_d, lon_d = args.destino or PONTO_DESTINO[1:]
    origem = grafo.no_mais_proximo(lat_o, lon_o, componente)
    destino = grafo.no_mais_proximo(lat_d, lon_d, componente)
    print(f"Origem : {PONTO_ORIGEM[0] if not args.origem else (lat_o, lon_o)} → {grafo.nome(origem)} "
          f"({haversine_m(lat_o, lon_o, *grafo.nos[origem]):.0f} m do ponto)")
    print(f"Destino: {PONTO_DESTINO[0] if not args.destino else (lat_d, lon_d)} → {grafo.nome(destino)} "
          f"({haversine_m(lat_d, lon_d, *grafo.nos[destino]):.0f} m do ponto)")

    print("\n══ 1. Comparativo de rotas por perfil " + "═" * 54)
    comparar_perfis(RoteadorAcessivel(grafo), origem, destino)

    print("\n══ 2. Ganho de desempenho da memoização " + "═" * 52)
    benchmark_memo(grafo, n_consultas=args.consultas)

    print("\n══ 3. Complexidade assintótica " + "═" * 61)
    print(COMPLEXIDADE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
