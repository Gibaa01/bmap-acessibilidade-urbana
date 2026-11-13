"""
features_acessibilidade.py
==========================

Funções reutilizadas pela coleta (`coletar_dados_reais.py`) e pela análise
(`analise_acessibilidade.ipynb`):

* distâncias geodésicas (fórmula de Haversine, vetorizada com NumPy);
* leitura/escrita de CSV com bloco de metadados de proveniência no topo.

Fonte dos dados tratados: © OpenStreetMap contributors, licença ODbL 1.0
(https://www.openstreetmap.org/copyright).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

RAIO_TERRA_M = 6_371_008.8  # raio médio da Terra (IUGG), em metros

# --------------------------------------------------------------------------- #
# Geoespacial
# --------------------------------------------------------------------------- #

def haversine_m(lat1, lon1, lat2, lon2):
    """Distância de grande círculo, em metros. Aceita escalares ou arrays
    (com broadcasting do NumPy)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * RAIO_TERRA_M * np.arcsin(np.sqrt(a))


def distancia_ao_mais_proximo(lat, lon, lat_ref, lon_ref, bloco: int = 2000) -> np.ndarray:
    """Para cada ponto (lat, lon), distância em metros até o ponto de referência
    mais próximo. Processa em blocos para não estourar memória.

    Retorna NaN para todos os pontos se não houver nenhuma referência — a
    ausência de dado é preservada, nunca convertida em zero.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lat_ref = np.asarray(lat_ref, dtype=float)
    lon_ref = np.asarray(lon_ref, dtype=float)
    if lat_ref.size == 0:
        return np.full(lat.shape, np.nan)
    saida = np.empty(lat.shape)
    for i in range(0, lat.size, bloco):
        d = haversine_m(lat[i:i + bloco, None], lon[i:i + bloco, None],
                        lat_ref[None, :], lon_ref[None, :])
        saida[i:i + bloco] = d.min(axis=1)
    return saida


def contar_vizinhos(lat, lon, raio_m: float) -> np.ndarray:
    """Quantos OUTROS pontos do próprio conjunto estão a até `raio_m` metros.
    O(n²) — adequado para alguns milhares de pontos (escala de um bairro)."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    d = haversine_m(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    return (d <= raio_m).sum(axis=1) - 1  # desconta o próprio ponto


# --------------------------------------------------------------------------- #
# CSV com metadados de proveniencia
# --------------------------------------------------------------------------- #


def salvar_csv_com_metadados(df: pd.DataFrame, caminho: Path, metadados: Dict[str, object]) -> None:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", encoding="utf-8", newline="") as f:
        for chave, valor in metadados.items():
            texto = str(valor).replace("\n", " ")
            f.write(f"# {chave}: {texto}\n")
        df.to_csv(f, index=False)
    # Cópia legível por máquina dos mesmos metadados (sidecar JSON)
    caminho.with_suffix(".meta.json").write_text(
        json.dumps(metadados, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def ler_csv_com_metadados(caminho: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    caminho = Path(caminho)
    metadados: Dict[str, str] = {}
    n_linhas_meta = 0
    with open(caminho, encoding="utf-8") as f:
        for linha in f:
            if not linha.startswith("# "):
                break
            n_linhas_meta += 1
            chave, _, valor = linha[2:].rstrip("\n").partition(": ")
            metadados[chave] = valor
    df = pd.read_csv(caminho, skiprows=n_linhas_meta, low_memory=False)
    return df, metadados


# --------------------------------------------------------------------------- #
# Variáveis-alvo (critérios técnicos objetivos)
# --------------------------------------------------------------------------- #


PONTUACAO = {
    # atributo: {status: pontos}
    "acesso_cadeirante":  {"sim": 30, "limitado": 15},
    "banheiro_acessivel": {"sim": 10, "limitado": 5},
    "rampa":              {"sim": 10, "limitado": 5},
    "piso_tatil":         {"sim": 10, "limitado": 5},
    "laco_inducao":       {"sim": 8,  "limitado": 4},
    "braille":            {"sim": 7,  "limitado": 3},
}
PONTOS_TRANSPORTE_MAX = 25
RAIO_TRANSPORTE_M = 500.0       # ~6–8 min a pé em velocidade reduzida
RAIO_APTO_AUTONOMO_M = 500.0

ATRIBUTOS_FISICOS = ("acesso_cadeirante", "banheiro_acessivel", "rampa")
ATRIBUTOS_COMUNICACIONAIS = ("piso_tatil", "laco_inducao", "braille")


def _pontos(df: pd.DataFrame, atributo: str) -> pd.Series:
    return df[f"{atributo}_status"].map(PONTUACAO[atributo]).fillna(0.0)


def componente_transporte(dist_m: pd.Series) -> pd.Series:
    fracao = (1 - dist_m / RAIO_TRANSPORTE_M).clip(lower=0, upper=1)
    return (PONTOS_TRANSPORTE_MAX * fracao).fillna(0.0)  # sem parada acessível conhecida -> 0


def calcular_indice_infraestrutura(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona as três dimensões e o índice final (0–100)."""
    df = df.copy()
    df["dim_fisica"] = sum(_pontos(df, a) for a in ATRIBUTOS_FISICOS)
    df["dim_comunicacional"] = sum(_pontos(df, a) for a in ATRIBUTOS_COMUNICACIONAIS)
    df["dim_transporte"] = componente_transporte(df["dist_transporte_acessivel_m"])
    df["indice_infraestrutura_inclusiva"] = (
        df["dim_fisica"] + df["dim_comunicacional"] + df["dim_transporte"]
    ).clip(0, 100).round(2)
    return df


def calcular_apto_atendimento_autonomo(df: pd.DataFrame, exigir_recurso_extra: bool = False) -> pd.Series:
    """1 quando uma pessoa com deficiência consegue CHEGAR e ENTRAR sem ajuda:
    entrada acessível comprovada (wheelchair=yes) E transporte acessível a até
    RAIO_APTO_AUTONOMO_M. Com `exigir_recurso_extra=True`, exige também banheiro
    acessível ou ao menos um recurso comunicacional comprovado (critério mais rígido)."""
    entrada = df["acesso_cadeirante_status"].eq("sim")
    transporte = df["dist_transporte_acessivel_m"].le(RAIO_APTO_AUTONOMO_M)  # NaN -> False
    apto = entrada & transporte
    if exigir_recurso_extra:
        extra = df["banheiro_acessivel_status"].eq("sim")
        for a in ATRIBUTOS_COMUNICACIONAIS:
            extra |= df[f"{a}_status"].eq("sim")
        apto &= extra
    return apto.astype(int)


# --------------------------------------------------------------------------- #
# Features de CONTEXTO (prevenção de vazamento do alvo)
# --------------------------------------------------------------------------- #

# Referência geográfica: MASP, Av. Paulista 1578 (coordenada aproximada)
REF_CENTRO_PAULISTA = (-23.5614, -46.6559)

FEATURES_NUMERICAS = ["latitude", "longitude", "dist_centro_paulista_m",
                      "densidade_estab_300m", "n_tags_contexto", "building_levels"]
FEATURES_BINARIAS = ["tem_horario", "tem_contato", "tem_operador"]
FEATURES_CATEGORICAS = ["categoria", "osm_tipo"]
FEATURES_MODELO = FEATURES_NUMERICAS + FEATURES_BINARIAS + FEATURES_CATEGORICAS

# Colunas que NUNCA podem entrar no X (vazamento direto ou indireto do alvo)
PREFIXOS_PROIBIDOS = ("acesso_cadeirante", "banheiro_acessivel", "rampa", "piso_tatil",
                      "laco_inducao", "braille", "dim_", "dist_transporte", "indice_", "apto_",
                      "n_tags_acess", "informacao_acessibilidade")


def adicionar_features_contexto(df: pd.DataFrame) -> pd.DataFrame:
    """Features geográficas não supervisionadas (não usam o alvo): podem ser
    calculadas antes do split sem vazamento."""
    df = df.copy()
    df["dist_centro_paulista_m"] = haversine_m(df["latitude"], df["longitude"],
                                               *REF_CENTRO_PAULISTA).round(1)
    df["densidade_estab_300m"] = contar_vizinhos(df["latitude"], df["longitude"], 300.0)
    return df


def verificar_sem_vazamento(colunas) -> None:
    proibidas = [c for c in colunas if c.startswith(PREFIXOS_PROIBIDOS)]
    if proibidas:
        raise ValueError(f"Vazamento do alvo: colunas proibidas nas features: {proibidas}")
