"""
coletar_dados_reais.py
======================

Coleta de DADOS PÚBLICOS REAIS sobre acessibilidade de estabelecimentos
(hospitais, clínicas, bancos, farmácias e serviços públicos) no entorno da
Av. Paulista / Bela Vista / Aclimação — São Paulo/SP.

FONTE OFICIAL
-------------
* Base: OpenStreetMap (https://www.openstreetmap.org)
* Acesso: Overpass API (https://overpass-api.de) — consulta pública, sem chave
* Licença: Open Database License (ODbL) 1.0
  https://opendatacommons.org/licenses/odbl/1-0/
* Atribuição obrigatória: "© OpenStreetMap contributors"
  https://www.openstreetmap.org/copyright

LGPD (Lei 13.709/2018)
----------------------
* Só são coletados dados de ESTABELECIMENTOS (pontos de interesse), nunca de
  pessoas. Não usamos `out meta`, então nomes de usuário/editores do OSM e
  carimbos de edição não são baixados.
* Minimização: telefone, e-mail e site NÃO são gravados — apenas indicadores
  booleanos de "possui contato" (tem_contato).
* Consultórios individuais (amenity=doctors|dentist) costumam ter o nome de
  uma pessoa física; por padrão, o nome é omitido nesses casos.

USO
---
    python data-science/coletar_dados_reais.py
    python data-science/coletar_dados_reais.py --bbox -23.585 -46.665 -23.552 -46.622
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features_acessibilidade import distancia_ao_mais_proximo, salvar_csv_com_metadados  # noqa: E402

# --------------------------------------------------------------------------- #
# Configuração
# --------------------------------------------------------------------------- #

RAIZ = Path(__file__).resolve().parents[1]
DIR_DADOS = RAIZ / "dados"
ARQ_ESTABELECIMENTOS = DIR_DADOS / "estabelecimentos_reais_sp.csv"
ARQ_TRANSPORTE = DIR_DADOS / "transporte_publico_sp.csv"
DIR_BRUTO = DIR_DADOS / "bruto"

VERSAO_SCRIPT = "1.0.0"

# Instâncias públicas da Overpass API (tentadas em ordem, em caso de falha).
OVERPASS_ENDPOINTS: List[str] = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]

# Política de uso da Overpass: identifique-se e evite consultas repetidas.
USER_AGENT = "CitizenTech-Acessibilidade/1.0 (projeto academico; dados ODbL)"

# Bounding box (sul, oeste, norte, leste) — Av. Paulista, Bela Vista e Aclimação.
BBOX_PADRAO: Tuple[float, float, float, float] = (-23.585, -46.665, -23.552, -46.622)

# Margem extra (graus) para buscar paradas/estações ao redor da área: evita que
# estabelecimentos na borda pareçam "longe" de transporte só por efeito de corte.
MARGEM_TRANSPORTE_GRAUS = 0.006  # ~650 m

# amenity=* de interesse -> categoria do projeto
AMENITY_PARA_CATEGORIA: Dict[str, str] = {
    "hospital": "hospital",
    "clinic": "clinica",
    "doctors": "clinica",
    "dentist": "clinica",
    "pharmacy": "farmacia",
    "bank": "banco",
    "townhall": "servico_publico",
    "police": "servico_publico",
    "fire_station": "servico_publico",
    "post_office": "servico_publico",
    "courthouse": "servico_publico",
    "library": "servico_publico",
    "social_facility": "servico_publico",
    "community_centre": "servico_publico",
}

# healthcare=* (esquema de saúde do OSM) -> categoria
HEALTHCARE_PARA_CATEGORIA: Dict[str, str] = {
    "hospital": "hospital",
    "pharmacy": "farmacia",
}  # demais valores (clinic, doctor, laboratory, centre, rehabilitation...) -> "clinica"


# --------------------------------------------------------------------------- #
# Consultas Overpass QL
# --------------------------------------------------------------------------- #

def _bbox_str(bbox: Tuple[float, float, float, float]) -> str:
    s, w, n, e = bbox
    return f"{s},{w},{n},{e}"


def montar_query_estabelecimentos(bbox) -> str:
    """Nós, vias e relações (nwr) com as tags de interesse. `out tags center`
    devolve só as tags + o centroide (sem geometria completa, sem metadados de
    editores)."""
    b = _bbox_str(bbox)
    amenities = "|".join(sorted(AMENITY_PARA_CATEGORIA))
    return f"""[out:json][timeout:180];
(
  nwr["amenity"~"^({amenities})$"]({b});
  nwr["healthcare"]({b});
  nwr["office"="government"]({b});
);
out tags center;"""


def montar_query_transporte(bbox) -> str:
    """Paradas de ônibus, plataformas, estações e acessos de metrô/trem."""
    s, w, n, e = bbox
    m = MARGEM_TRANSPORTE_GRAUS
    b = _bbox_str((s - m, w - m, n + m, e + m))
    return f"""[out:json][timeout:180];
(
  node["highway"="bus_stop"]({b});
  node["public_transport"="platform"]({b});
  nwr["public_transport"="station"]({b});
  nwr["railway"~"^(station|halt)$"]({b});
  node["railway"="subway_entrance"]({b});
);
out tags center;"""


def consultar_overpass(query: str, tentativas_por_endpoint: int = 3, pausa_s: float = 5.0) -> Tuple[dict, str]:
    """Envia a consulta e devolve (json, endpoint_usado).

    Trata os erros típicos de instâncias públicas: 429 (limite de taxa) e
    504 (servidor ocupado) com espera crescente (backoff linear) e troca de
    endpoint se um servidor estiver indisponível.
    """
    ultimo_erro: Exception | None = None
    for url in OVERPASS_ENDPOINTS:
        for tentativa in range(1, tentativas_por_endpoint + 1):
            try:
                resp = requests.post(url, data={"data": query},
                                     headers={"User-Agent": USER_AGENT}, timeout=200)
                if resp.status_code == 200:
                    return resp.json(), url
                if resp.status_code in (429, 502, 503, 504):
                    print(f"  [{url}] HTTP {resp.status_code}; aguardando...", file=sys.stderr)
                    time.sleep(pausa_s * tentativa)
                    continue
                resp.raise_for_status()
            except (requests.RequestException, ValueError) as erro:
                ultimo_erro = erro
                print(f"  [{url}] falha ({erro}); tentativa {tentativa}", file=sys.stderr)
                time.sleep(pausa_s * tentativa)
    raise RuntimeError(f"Nenhum endpoint Overpass respondeu. Último erro: {ultimo_erro}")


def coordenadas(elemento: dict) -> Tuple[float | None, float | None]:
    """Nós têm lat/lon; vias e relações trazem o centroide em `center`."""
    if "lat" in elemento and "lon" in elemento:
        return elemento["lat"], elemento["lon"]
    centro = elemento.get("center")
    if centro:
        return centro["lat"], centro["lon"]
    return None, None


# --------------------------------------------------------------------------- #
# Normalização das tags de acessibilidade
# --------------------------------------------------------------------------- #
# Em bases colaborativas, a AUSÊNCIA de uma tag não significa "não acessível":
# significa "ninguém verificou". Por isso cada atributo vira:
#   <atributo>_status   ∈ {sim, limitado, nao, nao_informado}
#   <atributo>_auditado ∈ {1, 0}   (1 = alguém registrou um valor válido)
# e mantemos também a coluna com o valor bruto original da tag OSM.

# atributo do projeto -> chaves OSM (em ordem de prioridade)
TAGS_ACESSIBILIDADE: Dict[str, List[str]] = {
    "acesso_cadeirante": ["wheelchair"],
    "banheiro_acessivel": ["toilets:wheelchair"],
    "rampa": ["ramp:wheelchair", "ramp"],
    "piso_tatil": ["tactile_paving"],
    "laco_inducao": ["hearing_loop"],
    "braille": ["tactile_writing:braille", "tactile_writing", "braille"],
}
COLUNAS_BRUTAS_OSM = sorted({k for chaves in TAGS_ACESSIBILIDADE.values() for k in chaves})

MAPA_VALORES = {
    "yes": "sim", "designated": "sim", "true": "sim",
    "limited": "limitado", "partial": "limitado", "incorrect": "limitado",
    "no": "nao", "false": "nao",
}

STATUS_VALIDOS = ("sim", "limitado", "nao")


def normalizar_valor(valor) -> str:
    """'yes' -> 'sim'; ausente/vazio/'unknown'/valor fora do padrão -> 'nao_informado'.
    Valores múltiplos ('yes;limited') usam o primeiro."""
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return "nao_informado"
    texto = str(valor).strip().lower().split(";")[0].strip()
    return MAPA_VALORES.get(texto, "nao_informado")


def classificar_categoria(tags: Dict[str, str]) -> str | None:
    amenity = tags.get("amenity")
    if amenity in AMENITY_PARA_CATEGORIA:
        return AMENITY_PARA_CATEGORIA[amenity]
    if tags.get("healthcare"):
        return HEALTHCARE_PARA_CATEGORIA.get(tags["healthcare"], "clinica")
    if tags.get("office") == "government":
        return "servico_publico"
    return None


def _primeira_tag(tags: Dict[str, str], chaves: List[str]):
    for chave in chaves:
        if tags.get(chave) not in (None, ""):
            return tags[chave]
    return None


def _para_float(valor):
    try:
        return float(str(valor).replace(",", "."))
    except (TypeError, ValueError):
        return np.nan


CHAVES_CONTATO = ("phone", "contact:phone", "website", "contact:website",
                  "email", "contact:email", "contact:instagram", "contact:facebook")
CHAVES_EXCLUIDAS_DA_CONTAGEM = set(COLUNAS_BRUTAS_OSM) | {"name"}


def processar_estabelecimentos(elementos: List[dict], omitir_nome_consultorio: bool = True) -> pd.DataFrame:
    """Converte a resposta da Overpass em uma linha por estabelecimento."""
    linhas = []
    for el in elementos:
        tags = el.get("tags", {})
        categoria = classificar_categoria(tags)
        lat, lon = coordenadas(el)
        if categoria is None or lat is None:
            continue

        nome = tags.get("name")
        if omitir_nome_consultorio and tags.get("amenity") in ("doctors", "dentist"):
            nome = "[consultorio - nome omitido (LGPD)]" if nome else None

        linha = {
            "osm_tipo": el["type"],
            "osm_id": el["id"],
            "osm_url": f"https://www.openstreetmap.org/{el['type']}/{el['id']}",
            "name": nome,
            "categoria": categoria,
            "amenity": tags.get("amenity"),
            "healthcare": tags.get("healthcare"),
            "office": tags.get("office"),
            "latitude": lat,
            "longitude": lon,
        }
        # Valores brutos (NaN quando a tag não existe — ausência preservada)
        for chave in COLUNAS_BRUTAS_OSM:
            linha[chave] = tags.get(chave, np.nan)
        # Status normalizado + indicador de auditoria
        for atributo, chaves in TAGS_ACESSIBILIDADE.items():
            status = normalizar_valor(_primeira_tag(tags, chaves))
            linha[f"{atributo}_status"] = status
            linha[f"{atributo}_auditado"] = int(status in STATUS_VALIDOS)
        # Atributos de CONTEXTO (não descrevem acessibilidade; usados como features)
        linha["tem_horario"] = int("opening_hours" in tags)
        linha["tem_contato"] = int(any(k in tags for k in CHAVES_CONTATO))  # só o booleano (LGPD)
        linha["tem_operador"] = int(any(k in tags for k in ("operator", "brand", "network")))
        linha["n_tags_contexto"] = sum(1 for k in tags if k not in CHAVES_EXCLUIDAS_DA_CONTAGEM)
        linha["building_levels"] = _para_float(tags.get("building:levels"))
        linhas.append(linha)

    df = pd.DataFrame(linhas)
    if df.empty:
        return df
    df = df.drop_duplicates(subset="osm_url").reset_index(drop=True)
    colunas_aud = [f"{a}_auditado" for a in TAGS_ACESSIBILIDADE]
    df["n_tags_acess_auditadas"] = df[colunas_aud].sum(axis=1)
    df["informacao_acessibilidade"] = np.where(df["n_tags_acess_auditadas"] > 0,
                                               "auditada", "nao_informada")
    return df


def processar_transporte(elementos: List[dict]) -> pd.DataFrame:
    linhas = []
    for el in elementos:
        tags = el.get("tags", {})
        lat, lon = coordenadas(el)
        if lat is None:
            continue
        status = normalizar_valor(tags.get("wheelchair"))
        linhas.append({
            "osm_tipo": el["type"],
            "osm_id": el["id"],
            "osm_url": f"https://www.openstreetmap.org/{el['type']}/{el['id']}",
            "name": tags.get("name"),
            "modo": ("metro_trem" if (tags.get("railway") or tags.get("station") in ("subway", "train"))
                     else "onibus"),
            "wheelchair": tags.get("wheelchair", np.nan),
            "acessivel_status": status,
            "latitude": lat,
            "longitude": lon,
        })
    df = pd.DataFrame(linhas)
    return df.drop_duplicates(subset="osm_url").reset_index(drop=True) if not df.empty else df


def adicionar_proximidade_transporte(estab: pd.DataFrame, transp: pd.DataFrame) -> pd.DataFrame:
    """Distância (m) ao transporte público mais próximo — qualquer um e apenas
    os marcados como acessíveis (wheelchair=yes/designated) no OSM."""
    estab = estab.copy()
    if transp.empty:
        estab["dist_transporte_qualquer_m"] = np.nan
        estab["dist_transporte_acessivel_m"] = np.nan
        return estab
    acess = transp[transp["acessivel_status"] == "sim"]
    estab["dist_transporte_qualquer_m"] = distancia_ao_mais_proximo(
        estab["latitude"], estab["longitude"], transp["latitude"], transp["longitude"]).round(1)
    estab["dist_transporte_acessivel_m"] = distancia_ao_mais_proximo(
        estab["latitude"], estab["longitude"], acess["latitude"], acess["longitude"]).round(1)
    return estab


# --------------------------------------------------------------------------- #
# Execução
# --------------------------------------------------------------------------- #

def _metadados(endpoint: str, bbox, query: str, n: int, descricao: str) -> Dict[str, object]:
    return {
        "titulo": descricao,
        "fonte": "OpenStreetMap contributors, via Overpass API",
        "endpoint": endpoint,
        "licenca": "Open Database License (ODbL) 1.0 - https://opendatacommons.org/licenses/odbl/1-0/",
        "atribuicao": "(c) OpenStreetMap contributors - https://www.openstreetmap.org/copyright",
        "data_extracao_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bbox_sul_oeste_norte_leste": ",".join(map(str, bbox)),
        "query_sha256": hashlib.sha256(query.encode()).hexdigest()[:16],
        "n_registros": n,
        "lgpd": "Somente estabelecimentos; sem dados pessoais; sem telefones/e-mails; sem metadados de editores",
        "ausencia_de_tag": "nao_informado = tag inexistente no OSM (nao verificado), NAO significa inacessivel",
        "script": f"coletar_dados_reais.py v{VERSAO_SCRIPT}",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Coleta dados reais de acessibilidade (OSM/Overpass).")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("SUL", "OESTE", "NORTE", "LESTE"),
                        default=BBOX_PADRAO)
    parser.add_argument("--manter-nome-consultorio", action="store_true",
                        help="não omitir nomes de amenity=doctors|dentist")
    args = parser.parse_args(argv)
    bbox = tuple(args.bbox)

    print("1/3 Consultando estabelecimentos na Overpass API...")
    q_estab = montar_query_estabelecimentos(bbox)
    bruto_estab, endpoint = consultar_overpass(q_estab)

    print("2/3 Consultando paradas e estações de transporte público...")
    q_transp = montar_query_transporte(bbox)
    time.sleep(2)  # cortesia com o servidor público
    bruto_transp, _ = consultar_overpass(q_transp)

    DIR_BRUTO.mkdir(parents=True, exist_ok=True)
    (DIR_BRUTO / "overpass_estabelecimentos.json").write_text(json.dumps(bruto_estab), encoding="utf-8")
    (DIR_BRUTO / "overpass_transporte.json").write_text(json.dumps(bruto_transp), encoding="utf-8")

    print("3/3 Processando e salvando CSVs...")
    estab = processar_estabelecimentos(bruto_estab.get("elements", []),
                                       omitir_nome_consultorio=not args.manter_nome_consultorio)
    transp = processar_transporte(bruto_transp.get("elements", []))
    if estab.empty:
        print("Nenhum estabelecimento encontrado na área informada.", file=sys.stderr)
        return 1
    estab = adicionar_proximidade_transporte(estab, transp)

    salvar_csv_com_metadados(estab, ARQ_ESTABELECIMENTOS, _metadados(
        endpoint, bbox, q_estab, len(estab),
        "Estabelecimentos reais (saude, bancos, farmacias, servicos publicos) - Paulista/Bela Vista/Aclimacao, SP"))
    salvar_csv_com_metadados(transp, ARQ_TRANSPORTE, _metadados(
        endpoint, bbox, q_transp, len(transp), "Paradas e estacoes de transporte publico - entorno da area"))

    # Resumo da qualidade da informação (subnotificação)
    print(f"\nEstabelecimentos: {len(estab)}  |  Paradas/estações: {len(transp)}")
    print(estab["categoria"].value_counts().to_string())
    print("\n% com informação auditada por atributo:")
    for atributo in TAGS_ACESSIBILIDADE:
        print(f"  {atributo:<20} {100 * estab[f'{atributo}_auditado'].mean():5.1f}%")
    print(f"\nArquivos salvos em: {ARQ_ESTABELECIMENTOS} e {ARQ_TRANSPORTE}")
    print("Fonte: (c) OpenStreetMap contributors, ODbL 1.0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
