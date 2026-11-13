# %% [markdown]
# # CitizenTech — Acessibilidade Urbana e Inclusão de PCD
# ## Módulo 1 · Data Science & Statistical Computing
#
# **Fonte dos dados:** © OpenStreetMap contributors, via Overpass API — licença
# ODbL 1.0 (<https://www.openstreetmap.org/copyright>). Dados gerados por
# `coletar_dados_reais.py` (área: Av. Paulista / Bela Vista / Aclimação — São Paulo/SP).
#
# **LGPD:** a base contém apenas estabelecimentos (pontos de interesse), sem dados
# pessoais, sem telefones/e-mails e sem metadados de editores do OSM.
#
# Roteiro:
# 1. Carga e auditoria de qualidade (subnotificação)
# 2. Limpeza e construção das variáveis-alvo
# 3. Split **antes** de qualquer pré-processamento (prevenção de data leakage)
# 4. Regressão Linear → `indice_infraestrutura_inclusiva` (MAE, RMSE, R²)
# 5. Regressão Logística → `apto_atendimento_autonomo` (matriz de confusão, precisão, recall, F1)
# 6. Coeficientes, impacto social, recomendações e limitações

# %%
import os
import sys
import warnings
from pathlib import Path

import matplotlib
if "ipykernel" not in sys.modules:      # rodando como script: sem janela gráfica
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _achar_raiz() -> Path:
    inicio = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
    for pasta in [inicio, *inicio.parents]:
        if (pasta / "modulo1_data_science").is_dir():
            return pasta
    return inicio


RAIZ = _achar_raiz()
sys.path.insert(0, str(RAIZ / "modulo1_data_science"))

from features_acessibilidade import (  # noqa: E402
    FEATURES_BINARIAS, FEATURES_CATEGORICAS, FEATURES_MODELO, FEATURES_NUMERICAS,
    adicionar_features_contexto, calcular_apto_atendimento_autonomo,
    calcular_indice_infraestrutura, ler_csv_com_metadados, salvar_csv_com_metadados,
    verificar_sem_vazamento,
)

SEMENTE = 42
ARQ_CSV = Path(os.environ.get("CITIZENTECH_CSV", RAIZ / "dados" / "estabelecimentos_reais_sp.csv"))
DIR_SAIDAS = RAIZ / "saidas"
DIR_SAIDAS.mkdir(exist_ok=True)
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 30)

if not ARQ_CSV.exists():
    raise FileNotFoundError(
        f"{ARQ_CSV} não encontrado. Rode antes: python modulo1_data_science/coletar_dados_reais.py")

# %% [markdown]
# ## 1. Carga e auditoria de qualidade dos dados reais
# O bloco de metadados no topo do CSV documenta fonte, licença e data da extração.

# %%
df_bruto, META = ler_csv_com_metadados(ARQ_CSV)
for chave in ("fonte", "licenca", "data_extracao_utc", "bbox_sul_oeste_norte_leste", "n_registros"):
    print(f"{chave:>28}: {META.get(chave)}")
print(f"\nDimensões: {df_bruto.shape}")
print(df_bruto["categoria"].value_counts().rename("n").to_frame())

# %% [markdown]
# ### Subnotificação: quanto de cada atributo foi realmente verificado?
# Em bases colaborativas, **tag ausente ≠ inacessível**. A tabela abaixo mostra a
# fração de estabelecimentos com informação *auditada* (alguém registrou sim/limitado/não).

# %%
ATRIBUTOS = ["acesso_cadeirante", "banheiro_acessivel", "rampa",
             "piso_tatil", "laco_inducao", "braille"]
cobertura = pd.DataFrame({
    "% auditado": [100 * df_bruto[f"{a}_auditado"].mean() for a in ATRIBUTOS],
    "% sim (entre auditados)": [
        100 * df_bruto.loc[df_bruto[f"{a}_auditado"] == 1, f"{a}_status"].eq("sim").mean()
        if df_bruto[f"{a}_auditado"].sum() else np.nan
        for a in ATRIBUTOS],
}, index=ATRIBUTOS).round(1)
print(cobertura)
print("\nPor categoria — % com ao menos um atributo auditado:")
print((100 * df_bruto.groupby("categoria")["n_tags_acess_auditadas"]
       .apply(lambda s: (s > 0).mean())).round(1).to_string())

fig, ax = plt.subplots(figsize=(8, 4))
cobertura["% auditado"].sort_values().plot.barh(ax=ax, color="#4C72B0")
ax.set_xlabel("% de estabelecimentos com a tag preenchida")
ax.set_title("Cobertura das tags de acessibilidade no OSM\nFonte: © OpenStreetMap contributors (ODbL)")
fig.tight_layout()
fig.savefig(DIR_SAIDAS / "cobertura_tags.png", dpi=150)
plt.show()

# %% [markdown]
# ## 2. Limpeza e construção das variáveis-alvo
# * remove duplicatas e registros sem coordenada;
# * valores fisicamente impossíveis de `building_levels` viram NaN (a imputação
#   acontece **depois** do split, dentro do Pipeline);
# * calcula o **índice (0–100)** e o **alvo binário** a partir de critérios técnicos
#   (ver `features_acessibilidade.py` — pesos documentados em um único lugar).

# %%
df = (df_bruto
      .drop_duplicates(subset="osm_url")
      .dropna(subset=["latitude", "longitude", "categoria"])
      .reset_index(drop=True))
df.loc[(df["building_levels"] < 0) | (df["building_levels"] > 60), "building_levels"] = np.nan

df = calcular_indice_infraestrutura(df)
df["apto_atendimento_autonomo"] = calcular_apto_atendimento_autonomo(df)
df = adicionar_features_contexto(df)   # geográficas, não supervisionadas → sem vazamento

print(df[["dim_fisica", "dim_comunicacional", "dim_transporte",
          "indice_infraestrutura_inclusiva"]].describe().round(2))
print("\nalvo binário:", df["apto_atendimento_autonomo"].value_counts().to_dict())

fig, ax = plt.subplots(figsize=(8, 4))
for rotulo, grupo in df.groupby("informacao_acessibilidade"):
    ax.hist(grupo["indice_infraestrutura_inclusiva"], bins=20, alpha=0.6, label=rotulo)
ax.set_xlabel("indice_infraestrutura_inclusiva (0–100)")
ax.set_ylabel("estabelecimentos")
ax.legend(title="informação de acessibilidade")
ax.set_title("Distribuição do índice — auditados vs. não informados")
fig.tight_layout()
fig.savefig(DIR_SAIDAS / "distribuicao_indice.png", dpi=150)
plt.show()

# %% [markdown]
# ### Recorte de modelagem
# Estabelecimentos **sem nenhuma** tag de acessibilidade têm índice baixo por
# *falta de informação*, não por falta de acessibilidade. Treinar com eles
# ensinaria o modelo a prever "não mapeado". Por isso a modelagem usa apenas
# os estabelecimentos **auditados**; os não informados são pontuados no final
# para **priorizar vistorias em campo**.

# %%
MIN_AUDITADOS = 40
auditados = df["informacao_acessibilidade"].eq("auditada")
if auditados.sum() >= MIN_AUDITADOS:
    df_modelo = df[auditados].reset_index(drop=True)
    df_sem_info = df[~auditados].reset_index(drop=True)
else:
    warnings.warn(f"Apenas {auditados.sum()} auditados (< {MIN_AUDITADOS}); usando a base inteira. "
                  "Interprete os resultados com cautela.")
    df_modelo, df_sem_info = df.copy(), df.iloc[0:0].copy()
print(f"Modelagem: {len(df_modelo)} auditados | Para priorização: {len(df_sem_info)} sem informação")

# %% [markdown]
# ## 3. Split ANTES do pré-processamento (prevenção de data leakage)
# Duas fontes de vazamento são tratadas:
# 1. **Vazamento de pré-processamento:** a mediana do imputer, a média/desvio do
#    scaler e as categorias do one-hot são aprendidas **somente no treino**, porque
#    tudo fica dentro de um `Pipeline` que é ajustado depois do `train_test_split`.
# 2. **Vazamento do alvo:** as colunas usadas para *calcular* o alvo (tags de
#    acessibilidade, distância ao transporte acessível) são **proibidas** como features.

# %%
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report, confusion_matrix,
                             f1_score, mean_absolute_error, mean_squared_error, precision_score,
                             r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

verificar_sem_vazamento(FEATURES_MODELO)

X = df_modelo[FEATURES_MODELO]
y_reg = df_modelo["indice_infraestrutura_inclusiva"]
y_clf = df_modelo["apto_atendimento_autonomo"]

# Estratifica pelo alvo binário quando possível: mantém a proporção de classes
pode_estratificar = y_clf.nunique() == 2 and y_clf.value_counts().min() >= 2
X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = train_test_split(
    X, y_reg, y_clf, test_size=0.25, random_state=SEMENTE,
    stratify=y_clf if pode_estratificar else None)
print(f"treino: {len(X_tr)} | teste: {len(X_te)} | estratificado: {pode_estratificar}")


def construir_preprocessador() -> ColumnTransformer:
    """Novo ColumnTransformer (sem estado) — cada Pipeline recebe o seu."""
    numerico = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),  # + flag "estava ausente"
        ("scaler", StandardScaler()),
    ])
    binario = Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))])
    categorico = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="desconhecido")),
        # drop="first" evita colinearidade perfeita com o intercepto (coeficientes interpretáveis);
        # categorias inéditas no teste viram vetor de zeros (= categoria de referência)
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numerico, FEATURES_NUMERICAS),
        ("bin", binario, FEATURES_BINARIAS),
        ("cat", categorico, FEATURES_CATEGORICAS),
    ])


# %% [markdown]
# ## 4. Regressão Linear — `indice_infraestrutura_inclusiva`
# Comparamos com um **baseline** que sempre prevê a média do treino: um modelo só
# é útil se errar menos do que "chutar a média".

# %%
modelo_reg = Pipeline([("pre", construir_preprocessador()), ("modelo", LinearRegression())])
modelo_reg.fit(X_tr, yr_tr)
pred_reg = modelo_reg.predict(X_te)
pred_base = DummyRegressor(strategy="mean").fit(X_tr, yr_tr).predict(X_te)


def metricas_regressao(y_true, y_pred) -> dict:
    return {"MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": r2_score(y_true, y_pred)}


m_reg, m_base = metricas_regressao(yr_te, pred_reg), metricas_regressao(yr_te, pred_base)
print(pd.DataFrame({"Regressão Linear": m_reg, "Baseline (média)": m_base}).round(3))

# Validação cruzada no TREINO (o Pipeline é reajustado em cada fold → sem vazamento)
cv = cross_validate(Pipeline([("pre", construir_preprocessador()), ("modelo", LinearRegression())]),
                    X_tr, yr_tr, cv=KFold(5, shuffle=True, random_state=SEMENTE),
                    scoring={"MAE": "neg_mean_absolute_error", "R2": "r2"})
print(f"\nCV 5-fold (treino): MAE = {-cv['test_MAE'].mean():.2f} ± {cv['test_MAE'].std():.2f} | "
      f"R² = {cv['test_R2'].mean():.3f} ± {cv['test_R2'].std():.3f}")


def interpretar_regressao(m: dict, base: dict) -> str:
    ganho = 100 * (1 - m["MAE"] / base["MAE"]) if base["MAE"] else 0.0
    linhas = [
        f"• MAE = {m['MAE']:.2f} pontos: em média, a previsão erra {m['MAE']:.1f} pontos na escala 0–100 "
        f"({ganho:+.1f}% vs. baseline).",
        f"• RMSE = {m['RMSE']:.2f}: penaliza erros grandes. Razão RMSE/MAE = {m['RMSE'] / m['MAE']:.2f} "
        + ("(> 1,4: há estabelecimentos com erro muito alto — casos atípicos)."
           if m["RMSE"] / m["MAE"] > 1.4 else "(≤ 1,4: erros relativamente homogêneos)."),
        f"• R² = {m['R2']:.3f}: fração da variância do índice explicada só pelo CONTEXTO do estabelecimento.",
    ]
    if m["R2"] < 0.1:
        linhas.append("  → Contexto explica pouco: acessibilidade depende de decisões de cada local "
                      "(reforma, gestão), não de tipo/localização — o que justifica vistoria caso a caso.")
    elif m["R2"] < 0.4:
        linhas.append("  → Sinal moderado: o perfil do estabelecimento ajuda a priorizar, mas não substitui vistoria.")
    else:
        linhas.append("  → Sinal forte: o perfil do estabelecimento é um bom indicador para priorizar fiscalização.")
    return "\n".join(linhas)


print("\n" + interpretar_regressao(m_reg, m_base))

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(yr_te, pred_reg, alpha=0.6)
lim = [0, max(100, float(np.nanmax(pred_reg)))]
ax.plot(lim, lim, "k--", lw=1)
ax.set_xlabel("índice real (teste)")
ax.set_ylabel("índice previsto")
ax.set_title("Regressão Linear — previsto vs. real")
fig.tight_layout()
fig.savefig(DIR_SAIDAS / "regressao_previsto_vs_real.png", dpi=150)
plt.show()

# %% [markdown]
# ### Demonstração: o que acontece COM vazamento do alvo
# Só para fins didáticos: se as colunas que compõem o índice entrassem como features,
# o R² seria quase perfeito — mas o modelo estaria apenas decorando a fórmula.

# %%
cols_vazadas = [f"{a}_status" for a in ATRIBUTOS] + ["dist_transporte_acessivel_m"]
pre_vazado = ColumnTransformer([
    ("num", SimpleImputer(strategy="constant", fill_value=10_000), ["dist_transporte_acessivel_m"]),
    ("cat", OneHotEncoder(handle_unknown="ignore"), [f"{a}_status" for a in ATRIBUTOS]),
])
Xv_tr, Xv_te = df_modelo.loc[X_tr.index, cols_vazadas], df_modelo.loc[X_te.index, cols_vazadas]
r2_vazado = r2_score(yr_te, Pipeline([("pre", pre_vazado), ("m", LinearRegression())])
                     .fit(Xv_tr, yr_tr).predict(Xv_te))
print(f"R² COM vazamento = {r2_vazado:.3f}  vs.  R² do modelo correto = {m_reg['R2']:.3f}")
print("→ R² alto aqui NÃO é mérito: é a fórmula do alvo reaparecendo nas features.")

# %% [markdown]
# ## 5. Regressão Logística — `apto_atendimento_autonomo`
# Em dados reais a classe positiva costuma ser minoritária. Se a classe minoritária
# tiver menos de 40% dos casos, usamos `class_weight="balanced"`, que pondera cada
# exemplo pelo inverso da frequência da sua classe — sem isso, o modelo tende a
# prever sempre "não apto" e o recall da classe 1 desaba.

# %%
modelo_clf = None
if yc_tr.nunique() < 2:
    print("⚠ O treino tem uma única classe — não é possível treinar a Regressão Logística.\n"
          "  Revise o critério do alvo binário (ex.: RAIO_APTO_AUTONOMO_M) ou amplie a área de coleta.")
else:
    prop_minoritaria = yc_tr.value_counts(normalize=True).min()
    peso = "balanced" if prop_minoritaria < 0.40 else None
    print(f"proporção da classe minoritária (treino): {prop_minoritaria:.1%} → class_weight={peso!r}")

    modelo_clf = Pipeline([
        ("pre", construir_preprocessador()),
        ("modelo", LogisticRegression(class_weight=peso, max_iter=2000, random_state=SEMENTE)),
    ])
    modelo_clf.fit(X_tr, yc_tr)
    proba_te = modelo_clf.predict_proba(X_te)[:, 1]
    pred_clf = (proba_te >= 0.5).astype(int)

    cm = confusion_matrix(yc_te, pred_clf, labels=[0, 1])
    print("\nMatriz de confusão (linhas = real, colunas = previsto):")
    print(pd.DataFrame(cm, index=["real 0 (não apto)", "real 1 (apto)"],
                       columns=["prev 0", "prev 1"]))

    m_clf = {
        "Precisão": precision_score(yc_te, pred_clf, zero_division=0),
        "Recall": recall_score(yc_te, pred_clf, zero_division=0),
        "F1-Score": f1_score(yc_te, pred_clf, zero_division=0),
    }
    if yc_te.nunique() == 2:
        m_clf["ROC-AUC"] = roc_auc_score(yc_te, proba_te)
    base_clf = DummyClassifier(strategy="stratified", random_state=SEMENTE).fit(X_tr, yc_tr).predict(X_te)
    m_base_clf = {"Precisão": precision_score(yc_te, base_clf, zero_division=0),
                  "Recall": recall_score(yc_te, base_clf, zero_division=0),
                  "F1-Score": f1_score(yc_te, base_clf, zero_division=0)}
    print("\n" + pd.DataFrame({"Logística": m_clf, "Baseline (aleatório estratificado)": m_base_clf})
          .round(3).to_string())
    print("\n" + classification_report(yc_te, pred_clf, labels=[0, 1],
                                       target_names=["não apto", "apto"], zero_division=0))

    if pode_estratificar:
        cv_clf = cross_validate(
            Pipeline([("pre", construir_preprocessador()),
                      ("modelo", LogisticRegression(class_weight=peso, max_iter=2000))]),
            X_tr, yc_tr, cv=StratifiedKFold(5, shuffle=True, random_state=SEMENTE), scoring="f1")
        print(f"CV 5-fold (treino): F1 = {cv_clf['test_score'].mean():.3f} ± {cv_clf['test_score'].std():.3f}")

    tn, fp, fn, tp = cm.ravel()
    print(f"""
Leitura das métricas (classe positiva = apto ao atendimento autônomo):
• Precisão = {m_clf['Precisão']:.2f}: dos locais que o modelo indicou como aptos, essa fração realmente é.
  {fp} falso(s) positivo(s) = locais apontados como aptos que NÃO são — o erro mais grave para o
  cidadão, que pode se deslocar e encontrar uma barreira.
• Recall = {m_clf['Recall']:.2f}: dos locais realmente aptos, essa fração foi encontrada ({fn} perdido(s)).
• F1 = {m_clf['F1-Score']:.2f}: média harmônica entre precisão e recall.
→ Para recomendar locais a PCD, priorize PRECISÃO (subir o limiar de 0,5); para triagem de
  fiscalização (achar quem NÃO é apto), priorize o recall da classe 0.""")

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["não apto", "apto"]).plot(ax=ax, colorbar=False)
    ax.set_title("Regressão Logística — matriz de confusão")
    fig.tight_layout()
    fig.savefig(DIR_SAIDAS / "matriz_confusao.png", dpi=150)
    plt.show()

# %% [markdown]
# ## 6. Interpretação dos coeficientes e impacto social
# As features numéricas foram padronizadas, então os coeficientes são comparáveis:
# * **Linear:** variação esperada no índice (pontos) para +1 desvio-padrão da
#   feature (ou para a categoria vs. a categoria de referência).
# * **Logística:** `exp(coef)` = *odds ratio*. OR > 1 aumenta a chance de o local
#   ser apto; OR < 1 diminui.
#
# ⚠ Coeficientes indicam **associação**, não causalidade.

# %%
def tabela_coeficientes(pipeline: Pipeline) -> pd.DataFrame:
    nomes = pipeline.named_steps["pre"].get_feature_names_out()
    coefs = np.ravel(pipeline.named_steps["modelo"].coef_)
    tab = pd.DataFrame({"feature": [n.split("__", 1)[-1] for n in nomes], "coef": coefs})
    return tab.reindex(tab["coef"].abs().sort_values(ascending=False).index).reset_index(drop=True)


coef_reg = tabela_coeficientes(modelo_reg)
cats_ref = modelo_reg.named_steps["pre"].named_transformers_["cat"].named_steps["onehot"].categories_
print("Categorias de referência (absorvidas no intercepto):",
      {col: list(c)[0] for col, c in zip(FEATURES_CATEGORICAS, cats_ref)})
print("Regressão Linear — efeito no índice (pontos):")
print(coef_reg.round(3).to_string(index=False))

if modelo_clf is not None:
    coef_clf = tabela_coeficientes(modelo_clf)
    coef_clf["odds_ratio"] = np.exp(coef_clf["coef"])
    print("\nRegressão Logística — odds ratio de ser apto:")
    print(coef_clf.round(3).to_string(index=False))

DESCRICAO_FEATURE = {
    "tem_operador": "pertencer a rede/marca/operador",
    "tem_horario": "ter horário de funcionamento mapeado",
    "tem_contato": "ter canal de contato publicado",
    "n_tags_contexto": "estar bem documentado no OSM",
    "densidade_estab_300m": "estar em área com muitos serviços",
    "dist_centro_paulista_m": "estar mais longe do eixo da Av. Paulista",
    "building_levels": "estar em edifício com mais andares",
    "osm_tipo_way": "estar mapeado como edificação/área (way), não como ponto",
    "osm_tipo_relation": "estar mapeado como relação (complexo/multipolígono)",
}


def descrever(feature: str) -> str:
    if feature.startswith("categoria_"):
        return f"ser da categoria '{feature.removeprefix('categoria_')}' (vs. referência)"
    if feature.startswith("missingindicator_"):
        return f"não ter '{feature.removeprefix('missingindicator_')}' informado"
    return DESCRICAO_FEATURE.get(feature, feature)


sinal = coef_reg[~coef_reg["feature"].isin(["latitude", "longitude"])]
print("\nFatores mais associados a MAIOR índice:")
for _, l in sinal[sinal["coef"] > 0].head(3).iterrows():
    print(f"  + {descrever(l.feature)}: {l.coef:+.2f} pts")
print("Fatores mais associados a MENOR índice:")
for _, l in sinal[sinal["coef"] < 0].head(3).iterrows():
    print(f"  − {descrever(l.feature)}: {l.coef:+.2f} pts")

# %% [markdown]
# ### Priorização de vistorias: estabelecimentos SEM informação de acessibilidade
# Usamos os modelos treinados nos auditados para estimar o índice e a probabilidade
# de aptidão dos não informados. Locais **essenciais** (hospitais, serviços públicos)
# com **menor previsão** vão para o topo da fila de vistoria.
# A saída mantém a atribuição da fonte (ODbL) e é uma **estimativa**, não um laudo.

# %%
if len(df_sem_info):
    fila = df_sem_info[["osm_url", "name", "categoria", "latitude", "longitude"]].copy()
    fila["indice_previsto"] = modelo_reg.predict(df_sem_info[FEATURES_MODELO]).clip(0, 100).round(1)
    if modelo_clf is not None:
        fila["prob_apto_prevista"] = modelo_clf.predict_proba(df_sem_info[FEATURES_MODELO])[:, 1].round(3)
    fila["essencial"] = fila["categoria"].isin(["hospital", "servico_publico", "clinica"]).astype(int)
    fila = fila.sort_values(["essencial", "indice_previsto"], ascending=[False, True]).reset_index(drop=True)
    fila.insert(0, "prioridade", np.arange(1, len(fila) + 1))
    salvar_csv_com_metadados(fila, DIR_SAIDAS / "prioridade_vistoria.csv", {
        "fonte": META.get("fonte", "OpenStreetMap contributors"),
        "licenca": META.get("licenca", "ODbL 1.0"),
        "atribuicao": "(c) OpenStreetMap contributors",
        "data_extracao_utc": META.get("data_extracao_utc"),
        "aviso": "Valores previstos por modelo estatistico; nao substituem vistoria tecnica (NBR 9050)",
    })
    print(fila.head(10).to_string(index=False))
else:
    print("Não há estabelecimentos sem informação para priorizar.")

# %% [markdown]
# ## 7. Recomendações práticas
#
# **Para o poder público (Prefeitura/SMPED, subprefeituras):**
# 1. **Vistoria dirigida:** usar `saidas/prioridade_vistoria.csv` para ordenar a
#    fiscalização — serviços essenciais sem informação e com baixa previsão primeiro.
# 2. **Dados abertos de acessibilidade:** publicar o resultado das vistorias em
#    formato aberto e, quando possível, devolvê-lo ao OSM (`wheelchair`,
#    `toilets:wheelchair`, `tactile_paving`), reduzindo a subnotificação.
# 3. **Transporte:** a dimensão de transporte depende de paradas marcadas como
#    acessíveis; integrar o GTFS público da SPTrans/Metrô (`wheelchair_boarding`)
#    melhora o critério e revela "desertos" de transporte acessível.
# 4. **Mapatonas:** apoiar mutirões de mapeamento com coletivos de PCD — quem
#    usa o espaço sabe onde estão as barreiras.
#
# **Para os estabelecimentos:**
# 1. Declarar no OSM/Google o status real de acessibilidade (inclusive "limitado" e "não").
# 2. Priorizar o que mais pesa no índice e na autonomia: **entrada sem degraus**
#    (30 pts) e **banheiro acessível** (10 pts); depois recursos comunicacionais
#    de baixo custo (piso tátil, sinalização em braille, laço de indução no balcão).
# 3. Informar a rota acessível desde a parada/estação mais próxima.
#
# ## 8. Limitações (honestidade metodológica)
# * **Subnotificação:** a maioria das tags de acessibilidade está ausente no OSM.
#   Ausência ≠ inacessível; o índice dos não auditados é um **piso**, não a realidade.
# * **Viés de seleção:** quem é mapeado com detalhe (redes, hospitais grandes) não
#   representa o pequeno comércio. O modelo treinado nos auditados herda esse viés.
# * **Qualidade e atualização:** dados colaborativos podem estar desatualizados
#   (reformas, mudanças) e não seguem necessariamente os critérios da NBR 9050.
# * **Alvo construído:** índice e alvo binário dependem dos pesos definidos neste
#   projeto; outros pesos mudam o ranking. Faça análise de sensibilidade.
# * **Autocorrelação espacial:** vizinhos se parecem; o split aleatório tende a ser
#   otimista. Uma validação mais rigorosa usaria blocos espaciais (`GroupKFold` por grade).
# * **Amostra pequena** em um único recorte de bairro: métricas com alta variância
#   (veja o desvio-padrão da validação cruzada) e baixa generalização para a cidade.
# * **Associação ≠ causalidade:** os coeficientes não indicam o efeito de intervenções.
# * **Transporte:** distância em linha reta ≠ rota acessível real (o Módulo 2 trata isso).
