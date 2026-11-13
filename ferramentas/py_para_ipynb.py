"""
Converte um script no formato "percent" (células delimitadas por `# %%`,
compatível com VS Code, Spyder e Jupytext) em um notebook Jupyter (.ipynb).

O .py é a fonte da verdade (diffs legíveis no Git); o .ipynb é gerado:
    python ferramentas/py_para_ipynb.py data-science/analise_acessibilidade.py
Sem dependências além da biblioteca padrão.
"""
import json
import sys
from pathlib import Path


def converter(caminho_py: Path) -> Path:
    celulas, atual, tipo = [], [], None

    def fechar():
        if tipo is None:
            return
        linhas = list(atual)
        while linhas and not linhas[-1].strip():
            linhas.pop()
        if tipo == "markdown":
            linhas = [l[2:] if l.startswith("# ") else l.lstrip("#") for l in linhas]
        fonte = [l + "\n" for l in linhas]
        if fonte:
            fonte[-1] = fonte[-1].rstrip("\n")
        cel = {"cell_type": tipo, "metadata": {}, "source": fonte}
        if tipo == "code":
            cel.update(execution_count=None, outputs=[])
        celulas.append(cel)

    for linha in caminho_py.read_text(encoding="utf-8").splitlines():
        if linha.startswith("# %%"):
            fechar()
            tipo, atual = ("markdown" if "[markdown]" in linha else "code"), []
        else:
            atual.append(linha)
    fechar()

    nb = {"cells": celulas, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}}
    destino = caminho_py.with_suffix(".ipynb")
    destino.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return destino


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print("gerado:", converter(Path(arg)))
