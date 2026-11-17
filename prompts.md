Registro de uso de IA (prompts.md) — Dynamic Programming

Documento exigido pelo regulamento da Global Solution: registra como a IA foi usada como copiloto na entrega de Dynamic Programming. A equipe é responsável por todo o código e pelos textos entregues: tudo foi lido, executado e é explicável pelos integrantes.

1. Resumo
Item	Informação
Projeto	BMap — CitizenTech: Acessibilidade Urbana e Inclusão PCD (Global Solution FIAP)
Disciplina	Dynamic Programming (dynamic-programming/rotas_acessiveis.py)
Ferramenta de IA	Claude (Anthropic)
Papel da IA	Copiloto: revisão do código, correção de erros de execução, conferência dos critérios do professor e rascunho do README
Decisões finais	Da equipe. A IA sugeriu; a equipe executou, testou e escolheu o que entrou
2. Registro de interações

Entrada 001 — 12/11/2026 — Revisão do código de Dynamic Programming

Prompt (resumo fiel): pediu revisão do código de Dynamic Programming (erros lógicos, bugs e boas práticas), validação da estrutura de grafos com memoização e a complexidade assintótica (Big-O) de tempo e espaço, com a versão corrigida do que fosse necessário, e a redação do README.

Resposta da IA (resumo): confirmou a estrutura (grafo em lista de adjacência, Dijkstra com heapq e memoização em dois níveis) e propôs ajustes: imports movidos para o topo do arquivo, tratamento explícito de conjunto vazio em no_mais_proximo, descarte de estados de memoização de versões antigas do grafo e correção da nota de espaço (heap com lazy deletion: O(O·P·(V+E))). Incluiu testes para esses pontos.

**O que a equipe adotou / modificou / rejeitou:** a equipe adotou todas as correções propostas, substituindo
`rotas_acessiveis.py` e `tests/test_modulo2.py` pelas versões revisadas, e rodou os testes até passarem.
Também renomeou a pasta do módulo para `dynamic-programming/` e atualizou os caminhos nos arquivos.
Nenhuma sugestão foi rejeitada.

Como foi validado: execução dos 11 testes automatizados de Dynamic Programming e da demonstração --demo.

Entrada 003 — 14/11/2026 — Resolução de erros de execução

Prompts (resumo fiel): a equipe colou as mensagens de erro do terminal e pediu ajuda: os testes falharam com ModuleNotFoundError: No module named 'rotas_acessiveis' e o número de testes executados não batia com o esperado.

Resposta da IA (resumo): identificou que o arquivo de teste ainda apontava para o nome antigo da pasta (modulo2_programacao_dinamica) e indicou a substituição por dynamic-programming; forneceu a versão atualizada de rotas_acessiveis.py e test_modulo2.py.

Como foi validado: a equipe substituiu os arquivos e os testes passaram (Ran 19 tests ... OK na etapa; 20 no projeto completo).

Entrada 004 — 16/11/2026 — README e regras do professor

Prompt (resumo fiel): a equipe enviou as regras de Dynamic Programming do professor (critérios de correção e padrão de entrega) e pediu a conferência do código e a adaptação do README.

Resposta da IA (resumo): conferiu o código contra os critérios (grafos, memoização, heapq, try/except, benchmark); atualizou o rascunho do README com a Ficha da CitizenTech, a lista de funcionalidades e a seção de qualidade de código; orientou a estrutura do ZIP de entrega.



3. O que a IA não fez
Não inventou dados: a malha vem do OpenStreetMap (ODbL), coletada por script.
Não substituiu a validação: rotas e benchmark foram obtidos executando o código.
Não definiu os pesos do custo por perfil como verdade: são decisões de projeto, documentadas no README.
Não incluiu dados pessoais.
4. Outras disciplinas

O uso de IA na entrega de Data Science está descrito na seção "Uso de IA" do relatório de Data Science.

5. Validação pela equipe
[X] Rodei python -m unittest discover -s tests -v e os testes passaram
[X] Rodei python dynamic-programming/rotas_acessiveis.py --online e conferi as rotas e o benchmark

