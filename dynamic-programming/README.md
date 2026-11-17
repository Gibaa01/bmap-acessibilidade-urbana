
Projeto	BMap — Acessibilidade Urbana e Inclusão de PCD
Problema escolhido	Falta de informação confiável e de rotas que considerem acessibilidade: a pessoa com deficiência não sabe por onde consegue ir, e o poder público não sabe onde vistoriar primeiro
Público afetado	Pessoas com deficiência (cadeirantes e com deficiência visual), idosos e famílias com carrinho; poder público municipal
Solução proposta	Dados abertos do OpenStreetMap → índice de infraestrutura inclusiva (Data Science) + roteador que calcula a rota mais acessível por perfil (Dynamic Programming)
Papel deste algoritmo	Responder "por onde eu consigo ir?" e apontar as barreiras de cada rota
Dados	Públicos (OpenStreetMap, ODbL 1.0), sem nenhum dado pessoal
Equipe
Nome completo	RM
<Felipe Ferrari Sumida>	<RM563082>
<Jean Pierre Andrade Feltran>	<RM566534>
<Giovanni Tarzoni Picccin>	<RM564014>


Turma: <2ESPV-2026> · Repositório: <[link do GitHub](https://github.com/Gibaa01/bmap-acessibilidade-urbana)>

1. Objetivo da solução

A rota mais curta quase nunca é a rota mais acessível. Uma escadaria economiza 100 m para quem anda, mas fecha o caminho de quem usa cadeira de rodas; uma travessia sem piso tátil e sem semáforo sonoro é um risco para quem tem deficiência visual.

rotas_acessiveis.py calcula, entre dois pontos de um bairro de São Paulo (Av. Paulista / Bela Vista), a rota de menor esforço para cada perfil de cidadão:

Perfil	O que pesa no custo do trecho
padrao	distância (escadas custam um pouco mais)
cadeirante	escada sem rampa = intransponível; guia alta; calçada estreita (< 1,20 m); inclinação > 8,33 % (limite da NBR 9050); piso ruim; via sem calçada
deficiencia_visual	travessia sem piso tátil / sem semáforo sonoro; escada sem corrimão; piso tátil confirmado reduz o custo; via sem calçada

A saída compara os três perfis (distância, custo ponderado, barreiras encontradas na rota e sobreposição entre as rotas) e mede o ganho da memoização.

Impacto social: o mesmo mapa passa a responder "por onde eu consigo ir?", e o relatório de barreiras da rota indica onde a prefeitura deve intervir primeiro.

2. Dados utilizados (reais)
Fonte: © OpenStreetMap contributors, via Overpass API — licença ODbL 1.0 (https://www.openstreetmap.org/copyright). Atribuição obrigatória mantida nos arquivos e na saída.
Recorte: Bela Vista, ao longo da Av. Paulista (entre Consolação e Brigadeiro).
Tags OSM usadas: highway, footway, sidewalk, kerb, tactile_paving, surface, smoothness, incline, width, handrail, wheelchair, ramp:wheelchair, traffic_signals:sound, crossing.
LGPD: só geometria e atributos de vias/calçadas; nenhum dado pessoal.
Ausência de tag ≠ inacessível. "Desconhecido" recebe penalidade menor que "barreira confirmada" e maior que "acessível confirmado" — a rota prefere trechos verificados sem descartar a malha não mapeada.
O modo --demo usa um subgrafo ilustrativo (só os pontos de referência são reais) e avisa isso na tela. Use --online para gerar os resultados do relatório.
3. Algoritmos utilizados
3.1 Modelagem como grafo
Nó = esquina / ponto de uma via (id OSM + latitude/longitude).
Aresta = trecho caminhável entre dois nós consecutivos de uma via, com distância (Haversine) e as tags OSM relevantes. Pedestres andam nos dois sentidos, então cada trecho vira duas arestas dirigidas.
Estrutura: lista de adjacência (Dict[nó, List[Aresta]]) — ideal para grafos esparsos como malhas viárias (E ≈ 2–3·V).
As tags de travessia (kerb, tactile_paving, traffic_signals:sound) ficam nos nós das pontas no OSM; ao montar a aresta, vale o pior caso entre as duas pontas (um lado rebaixado e o outro alto → alto).
3.2 Custo dinâmico por perfil
custo(aresta, perfil) = distância × fator(perfil, tags) + penalidade_fixa(perfil, tags)      [em "metros equivalentes de esforço"]
custo = ∞  →  trecho intransponível para aquele perfil

Todos os custos são ≥ 0 (inclusive o bônus do piso tátil, que é fator 0,90 e não negativo). Isso é o requisito de correção do Dijkstra: com pesos negativos o algoritmo falharia. Os parâmetros ficam em um único lugar (ParametrosCusto), o que permite análise de sensibilidade.

3.3 Dijkstra como Programação Dinâmica

Subestrutura ótima (princípio de Bellman): se o melhor caminho de s até t passa por u, então o trecho s → u também é o melhor possível. Isso gera a recorrência

dist[s] = 0
dist[v] = min  sobre (u,v) ∈ E  de  ( dist[u] + custo(u, v, perfil) )

O Dijkstra resolve essa recorrência na ordem certa (do menor dist para o maior, usando uma heap binária). Como os custos são não negativos, quando um nó sai da heap seu dist é definitivo — cada subproblema dist[u] é resolvido uma única vez e reaproveitado por todos os vizinhos (sobreposição de subproblemas). Duas otimizações: parada antecipada (para quando o destino é finalizado) e lazy deletion (entradas obsoletas da heap são ignoradas ao sair, em vez de removidas).

Nota conceitual: o Dijkstra é guloso na ordem de escolha e programação dinâmica na formulação (resolve a recorrência de Bellman). A memoização abaixo é a parte de PD explícita do projeto.

3.4 Memoização em dois níveis
Nível	Chave	O que guarda	Por que é seguro
1 · custo de aresta	(id_aresta, perfil)	resultado de custo_aresta	função pura (depende só da aresta e do perfil) e Aresta é imutável
2 · estado do Dijkstra	(origem, perfil, versão_do_grafo)	dist, pred, heap e conjunto de finalizados	a versão do grafo na chave invalida o cache quando a malha muda

O nível 2 é um Dijkstra retomável:

destino já finalizado → resposta em O(1) de busca + O(k) para reconstruir o caminho pelos predecessores;
destino ainda não alcançado → continua de onde parou, sem refazer nenhum subproblema já resolvido.

Quando a malha é alterada, os estados de versões antigas são descartados (evita vazamento de memória).

4. Análise de complexidade

V = nós · E = arestas dirigidas · P = nº de perfis (3) · k = nós no caminho · Q = consultas · O = origens distintas

Operação	Tempo	Espaço
Dijkstra (heap binária, lazy deletion)	O((V + E) log V)	O(V + E)
Custo de uma aresta	O(1) (nº fixo de tags)	—
1ª consulta de (origem, perfil) (cache frio)	O((V + E) log V)	O(V + E) por estado
Consulta repetida (destino já finalizado)	O(1) de busca + O(k) de reconstrução	—
Novo destino com origem já em cache	só o trabalho que falta (retomada)	—
Q consultas, O origens distintas, com memo	O(O·P·(V + E) log V + Q·k)	O(E·P) + O(O·P·(V + E))
Q consultas sem memo	O(Q·(V + E) log V)	O(V + E)
Snapping da coordenada / maior componente conexa	O(V) / O(V + E)	O(V)

Notas: (1) a heap com lazy deletion pode guardar até E entradas, daí o termo (V + E) no espaço dos estados; como log E ≤ 2 log V, a heap não altera o fator logarítmico. (2) Em malhas viárias E = O(V), então na prática o custo é O(V log V). (3) A memoização transforma o fator Q (nº de consultas) em O·P (nº de pares origem/perfil), e o ganho cresce quanto mais consultas repetirem a origem. O programa mede isso (benchmark_memo): resultados idênticos com e sem memo e contagem de nós expandidos nos dois modos.

5. Como rodar

Requisito: Python ≥ 3.10.

bash
# na raiz do repositório
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1ª vez, com internet: baixa a malha real do OSM e salva o cache em dados/
python dynamic-programming/rotas_acessiveis.py --online

# depois funciona offline, usando o cache real
python dynamic-programming/rotas_acessiveis.py

# sem internet e sem cache: subgrafo ILUSTRATIVO (não use no relatório)
python dynamic-programming/rotas_acessiveis.py --demo

# outros pontos (latitude longitude) e nº de consultas do benchmark
python dynamic-programming/rotas_acessiveis.py --origem -23.5614 -46.6559 --destino -23.5685 -46.6485 --consultas 500

# testes automatizados
python -m unittest discover -s tests -v

A execução imprime: (1) comparativo de rotas por perfil, (2) benchmark com/sem memoização e (3) a tabela de complexidade.

6. Funcionalidades implementadas
Construção do grafo a partir da malha pedestre do OpenStreetMap: online (Overpass), cache local ou subgrafo ilustrativo (--demo).
Custo por perfil (padrao, cadeirante, deficiencia_visual), com trechos intransponíveis e penalidades configuráveis.
Rota mais acessível entre dois pontos (latitude/longitude) com Dijkstra e memoização em dois níveis.
Comparação entre perfis: distância, custo ponderado, barreiras encontradas na rota e sobreposição (Jaccard) entre rotas.
Encaixe da coordenada no grafo (nó mais próximo, restrito à maior componente conexa).
Benchmark com e sem memoização, verificando que os resultados são idênticos.
Interface de linha de comando (--online, --demo, --origem, --destino, --consultas).
Testes automatizados (11 testes de Dynamic Programming).
Qualidade de código
Fila de prioridade com heapq (sem pop(0)); a busca de componente conexa usa pilha com pop() em O(1).
Funções puras e com retorno (haversine_m, custo_aresta), separadas da interface (main e funções de impressão).
try/except na rede (Overpass, com troca de servidor) e na leitura do cache, com fallback para o modo demo.
Estruturas imutáveis (Aresta é frozen) e dataclasses, o que torna a memoização segura.
7. Estrutura do código (rotas_acessiveis.py)
Bloco	Responsabilidade
Aresta, GrafoUrbano	grafo (lista de adjacência), versão do grafo, componente conexa, snapping
ParametrosCusto, custo_aresta	modelo de custo por perfil (função pura)
RoteadorAcessivel	Dijkstra retomável + memoização em dois níveis + estatísticas
baixar_malha_overpass, grafo_de_overpass, carregar_grafo	Overpass → grafo; cache local; fallback demo
comparar_perfis, benchmark_memo, main	demonstração, benchmark e CLI
8. Limitações
Dados colaborativos: tags ausentes ou desatualizadas; não substituem laudo técnico (NBR 9050).
Muitas vias de carro no OSM não têm a tag sidewalk; o modelo só penaliza quando ela é explicitamente no/none — o ideal seria mapear calçadas como vias próprias.
Os pesos do custo são decisões de projeto (a mudança deles altera as rotas); faça análise de sensibilidade.
Não considera tempo de semáforo, lotação, obras temporárias nem transporte público (extensão natural: GTFS).
A distância entre ponto e nó mais próximo (snapping) é impressa para o usuário avaliar o encaixe.
9. Uso de IA

O uso de IA como copiloto está documentado no arquivo prompts.md, na raiz do projeto.