# Avaliação de publicabilidade — "When Does Joint Admission Control and Routing Pay?" (v3, out/2026)

Data: 2026-10-06 · Alvo no ficheiro: IEEE TNSM · Fontes: LaTeX v3 completo (inclui CLAIMS AUDIT), Crossref, arXiv, pesquisa web.
Limitações desta pesquisa: OpenAlex/Semantic Scholar sem quota (429), IEEE Xplore/Scholar não consultados, grafo de citações **não** construído. A cobertura do estado da arte é por isso de "boa amostragem", não exaustiva. Quartil do TNSM não verificado no Scimago nesta sessão.

## Veredicto
**Publicável em Q1 de nível TNSM / Computer Networks — mas não na forma atual.** A ideia central (o valor de aprender routing é governado por uma estatística do substrato, calculável sem treino) continua sem concorrente direto nas fontes que consultei. Contudo, o estado da arte entretanto mudou (ver abaixo) e o paper tem três fragilidades que um revisor Q1 vai atacar. Com 6–10 semanas de trabalho dirigido, torna-se uma submissão forte; submetido hoje, o resultado provável é "major revision" ou rejeição.

## O que mudou no estado da arte (relevante para o posicionamento)
- OACR2, IEEE TMC 2025 (Li, Hao, Yang, Zhao) — AC + reserva de recursos com DRL em slices 5G. DOI 10.1109/tmc.2025.3548767
- Ahmadi, Moayyedi, Sulaiman, Salahuddin, IEEE TNSM 2024 — slicing/AC RAN/MEC **generalizável** (GNN). DOI 10.1109/tnsm.2024.3437217. Ataca a mesma agenda de generalização entre substratos que o paper invoca.
- Sulaiman, Sun, Salahuddin, Boutaba — AC online baseado em preços, com *competitive ratio*. arXiv 2501.10285 (2025) e a versão estendida OPA, arXiv 2608.03954 (ago 2026), que reporta +32.2% de receita sobre DRL "state of the art" numa topologia real. Já citam/duplicam o vosso [sulaiman2025datadriven]; a nova versão é um comparador mais forte do que o vosso "aggregate-state DQN".
- DePSAC (arXiv 2510.08769, 2025) — AC DRL com recompensa de atraso; AWaRe-SAC (arXiv 2601.05978, 2026) — AC sob incerteza de capacidade em x-haul mmWave; Verifying Behavior of RL Agents for Network Slice AC (CNSM 2025, DOI 10.23919/cnsm67658.2025.11297450) — verificação/explicabilidade de agentes RL para AC.
- RL para slice embedding em SDM-EON (arXiv 2608.29444, ago 2026) — PPO com ação conjunta nó+caminho em k-shortest paths; mostra que "ação conjunta caminho+admissão" é prática corrente, não novidade.
- Survey: Neurocomputing 2026, DRL para routing em redes (DOI 10.1016/j.neucom.2025.132263).
- Clássico em falta no paper: Marbach, Mihatsch, Tsitsiklis, "Call admission control and routing in integrated services networks using neuro-dynamic programming", IEEE JSAC 2000 (DOI 10.1109/49.824797). Já formulava AC+routing conjunto com RL. Não encontrei nenhuma citação a este trabalho nem à literatura de *alternate routing / trunk reservation* (Kelly, Gibbens, Mitra) no .tex. O ρ é próximo de noções clássicas de ganho de encaminhamento alternativo; sem esse enquadramento, a "novidade" arrisca ser lida como redescoberta.

Não encontrei, nas pesquisas feitas, trabalho que defina uma estatística de substrato análoga a ρ nem que mostre sinal negativo do routing conjunto. A contribuição permanece diferenciada, mas a lacuna "ninguém faz routing no espaço de ação" já não se sustenta.

## Pontos fracos que um revisor Q1 vai atacar (por gravidade)
1. **A "lei" é quase mecânica.** ρ é definido como a fração de chegadas admissíveis fora do caminho mais curto; um AC-only fixado no caminho mais curto perde exatamente esse alcance. O texto admite-o ("mechanical explanation"). Sem um limite teórico (p.ex. gap ≤ f(ρ)) ou previsão fora de amostra, soa a verificação de sanidade, não a lei. r²=0.979 assenta em 5 pontos × 2 seeds.
2. **O agente conjunto não bate o greedy na família controlada.** Tabela 'law': ρ=12.91% → conjunto 0.4843 vs greedy 0.4849; ρ=14.42% → 0.4776 vs 0.4774. O que a lei mostra é o valor de ter *flexibilidade de caminho*, não de *aprender* routing. Falta o baseline óbvio: AC-only aprendido + routing greedy/bottleneck (e ECMP). Se este empatar com o conjunto, a recomendação "factored agent" cai.
3. **Comparadores e agentes fracos/estreitos.** Só DQN duelling; "aggregate-state DQN" é um *stand-in* construído por vós, não os métodos publicados; sem PPO/GNN, sem OPA/OACR2. Um único topo real, tráfego sintético estacionário, simulador a nível de fluxo, sem perda de pacotes (já assumido no CLAIMS AUDIT). Vários efeitos com n=5 seeds e t≈2 ("não significativos"), n=3 em alta carga.

## Inconsistências internas a corrigir antes de qualquer submissão
- Resumo e contribuições: "−1.7% em ρ=0". A tabela da lei dá −0.58% em ρ=0 e −1.65% em ρ=0.70%.
- Texto diz que o ganho "rises monotonically" em ρ; na tabela vai de −0.58% para −1.65% nos dois primeiros níveis.
- Acceptance do greedy: Tabela da família (0.4739–0.4838, "dentro de 1 pp") vs coluna Greedy da tabela da lei (0.4708–0.4849, ≈1.4 pp). Explicar (seeds diferentes?) ou harmonizar.
- Modelo previsto pela reta vs medido nos substratos reais: operador 0.97·11.0−1.82 ≈ +8.9% (medido +7.4%); Waxman −1.8% (medido −2.3%). Reportar o erro de previsão, não só "caem perto da reta". Os dois pontos reais são n=2 e um deles tem ρ=0.

## Trabalho necessário (priorizado)
**Obrigatório para Q1 (≈6–10 semanas, tudo dentro do simulador atual):**
1. Baselines: AC-only + routing greedy/ECMP; PPO (com máscara) conjunto; um agente GNN ou pelo menos uma variante invariante a permutações; reproduzir OPA/preços (Sulaiman) como comparador "estado da arte não-DRL".
2. Reforçar a lei: ≥10 seeds por nível, IC a 95%, mais níveis de ρ (≥8) e **segunda família** (variar K, tamanho, distribuição de capacidades); previsão *out-of-sample* com substratos reais novos (Topology Zoo, SNDlib — não verifiquei links nesta sessão) e erro de previsão reportado. Idealmente um limite superior formal para o gap em função de ρ.
3. Reescrever o estado da arte/posicionamento: incluir OACR2, Ahmadi et al. TNSM 2024, OPA, Marbach et al. JSAC 2000 e a literatura clássica de alternate routing; reenquadrar ρ como estatística que torna *operacional* um efeito conhecido.
4. Corrigir as inconsistências numéricas acima e libertar código + gerador de substratos.

**Reforça muito (se houver tempo):** endpoints multi-homed (já listado como limitação), procura não estacionária, e uma verificação em emulação (OVS) de pelo menos um resultado.

## Portefólio (estratégia)
- **Aposta segura:** itens 1–4 acima → TNSM / Computer Networks / IEEE TCOM-nível Q1, submissão em ~2–3 meses.
- **Aposta estratégica:** lei + limite teórico + multi-homing + não estacionário → hipótese de JSAC/ToN, 6–9 meses.
- **Moonshot:** ligar ρ ao *competitive ratio*/pseudo-preços do OPA (porque o routing "paga" ⇔ preços por caminho divergem). Alto impacto, risco alto.
Premissa: assumi simulador e GPUs atuais; não confirmei orçamento, prazo nem alvo de venue — dizer-me altera o ranking.

## Referências (resolvidas nesta sessão via Crossref/arXiv)
- 10.1109/tmc.2025.3548767 · 10.1109/tnsm.2024.3437217 · 10.23919/cnsm67658.2025.11297450 · 10.1016/j.neucom.2025.132263 · 10.1109/49.824797
- arXiv: 2501.10285, 2608.03954, 2510.08769, 2601.05978, 2608.29444
