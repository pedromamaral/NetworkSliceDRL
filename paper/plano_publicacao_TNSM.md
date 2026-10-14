# Plano para tornar o paper publicável — TNSM, ~12 semanas

Data: 2026-10-06 · Decisões tomadas: alvo **IEEE TNSM**, prazo **~3 meses**, recursos **Pedro + GPU, código pronto**, narrativa **ρ como ferramenta de decisão** (o DRL é o veículo, não a contribuição).
Base: `publishability_assessment_rho_law.md` (avaliação do estado da arte, Xplore/Scholar incluídos).

---

## Parte A — Motivação e posicionamento

### A1. A pergunta do paper (o "hook")
Um operador que vai pôr um controlador de admissão aprendido no core IP/MPLS tem de decidir **antes de treinar** se o routing entra no espaço de ação. A literatura assume implicitamente que sim (formulações conjuntas em RAN, VNE, SFC e, em 2026, slice embedding em redes ópticas com ação conjunta nó+caminho) e avalia cada proposta num único substrato. Mostramos que o valor de dar escolha de caminho ao controlador é **condicional**, pode ser **negativo**, e é **previsível a partir de uma estatística do substrato, ρ, que não requer treino**.

Frase-âncora para a introdução: *the question is not whether joint control is better, but when, and whether this can be decided before deployment.* (já está no v3 — manter.)

### A2. Três razões pelas quais a pergunta importa (argumentos para a introdução)
1. **Custo de engenharia e de treino.** O espaço de ação conjunto cresce com K; o agente unificado degrada de K=3 para K=8 (−2.6%, significativo). Saber à partida que o routing não paga evita esse custo.
2. **Risco de piorar.** Em ρ≈0 o agente conjunto perde para o AC-only (−2.3%, t=5.97). Um operador que siga a literatura "joint is better" perde aceitação.
3. **Lacuna de generalização reconhecida.** A literatura recente ataca a transferência de *políticas* entre substratos (Ahmadi et al., TNSM 2024; FlagVNE). Nós atacamos uma pergunta anterior e complementar: *se vale a pena aprender esta arquitetura neste substrato*.

### A3. Ligação à teoria clássica (obrigatório — senão o revisor fá-la por nós)
A pergunta "quando é que a flexibilidade de routing compensa" tem raízes em redes de perdas. Posicionar ρ como a versão **operacional e calculável sem treino** dessa pergunta, para controladores aprendidos e slices multi-conexão:
- Kelly, *Loss Networks*, Ann. Appl. Probab. 1991 — DOI 10.1214/aoap/1177005872
- Gibbens & Kelly, *Dynamic routing in fully connected networks*, IMA J. Math. Control Inf. 1990 — DOI 10.1093/imamci/7.1.77
- Gibbens, Kelly & Turner, *Dynamic routing in multiparented networks*, IEEE/ACM ToN 1993 — DOI 10.1109/90.222932 (**multi-homing** — liga diretamente ao argumento estrutural de endpoints single-homed)
- Key, *Optimal control and trunk reservation in loss networks*, PEIS 1990 — DOI 10.1017/s0269964800001558 (a "preservação de admissibilidade" do agente é conceptualmente uma reserva de trunk aprendida — dizê-lo, é um ponto forte, não fraco)
- Marbach, Mihatsch & Tsitsiklis, *CAC and routing … neuro-dynamic programming*, IEEE JSAC 2000 — DOI 10.1109/49.824797 (AC+routing conjunto com RL, 26 anos antes)
- Awerbuch, Azar & Plotkin, *Throughput-competitive on-line routing*, FOCS 1993 — DOI 10.1109/sfcs.1993.366884 (AC+routing online com preços exponenciais; base do OPA de Sulaiman et al.)

### A4. Nova taxonomia para o Estado da Arte (substitui as subsecções atuais)
| Grupo | Onde está o routing | Exemplos | O que lhes falta face a nós |
|---|---|---|---|
| Clássico: loss networks / CAC+routing | Política analítica ou NDP | Kelly 1991; Gibbens–Kelly; Key 1990; Marbach 2000 | Slices multi-conexão; estado OSS; previsor a priori |
| Online / competitivo | Preços por ligação + caminho mais barato | AAP 1993; Sulaiman et al. 2025/2026 (OPA) | Não diz quando a escolha de caminho paga |
| AC aprendido, sem routing | Fora do espaço de ação | Tao et al. TCOM 2025; OACR2 TMC 2025; DePSAC 2025; CNSM 2025 | = o nosso AC-only; mostramos quando é preferível |
| 5G core, recursos agregados | Não existe | SARA/DSARA; FSAC; Sulaiman 2025 | Medimos o custo da abstração agregada (+14.6%) |
| Conjunto em RAN/MEC | Alocação rádio | Rezazadeh 2021; GAT multi-agente; SliceWise 2025 | Domínio diferente; sem topologia de core |
| Embedding (VNE/SFC/óptico) com caminho na ação | Ação conjunta | HRL-ACRA; FlagVNE; PPO-Full SDM-EON 2026; Alpha-SFC 2026 | Assumem que a ação conjunta paga; avaliam 1 substrato |

Nova coluna na tabela comparativa: **"Previsor a priori do benefício do routing"** — só nós temos ✓. Outra coluna: **"Nº de substratos / análise de condições"**.

### A5. Escada de afirmações (o que o paper vai dizer, por ordem)
1. **C1 — Critério ρ.** O ganho de dar escolha de caminho ao controlador escala com ρ; ρ≈0 ⇒ nenhum ganho (provável estruturalmente); ρ calcula-se em segundos, sem treino. *Atenção:* afirmar sobre "flexibilidade de routing", não sobre "routing aprendido" (o agente conjunto empata com o greedy na família controlada).
2. **C2 — O que vale a pena aprender é a admissão.** Ganho durável sobre o greedy sem memória vem da preservação de admissibilidade (≈ trunk reservation aprendida).
3. **C3 — A representação importa.** Estado por caminho vs. agregado: +6.7 pp só pela representação.
4. **C4 — Regra de implantação.** Provisioning ratio → ρ → arquitetura.

### A6. Decisões de nome e tom
- "Law" com 5 pontos convida ataque. Manter **"law"** só se a validação fora de amostra (WP2) passar; caso contrário **"criterion"**.
- "Routing headroom" já aparece noutros contextos (ML/LLM routing; Alpha-SFC). Ou definimos com muita clareza, ou mudamos para **"routing reach"**. Decidir na semana 6.

---

## Parte B — Trabalho necessário (pacotes, com gates)

### WP0 — Higiene (semana 1) · obrigatório · ~0 GPU
- Corrigir: "−1.7% em ρ=0" (tabela: −0.58% em ρ=0, −1.65% em ρ=0.70%); "rises monotonically" (falso nos dois primeiros níveis); diferença de greedy entre Tabela da família (0.4739–0.4838) e Tabela da lei (0.4708–0.4849).
- Regenerar todos os números a partir dos logs (hoje `make_figures.py` tem valores escritos à mão) → script único `results/ → tables + figures`.
- Escrever o **plano de análise pré-registado** (comparações, nº seeds, teste, correção para comparações múltiplas) num ficheiro no repositório, com data, antes das novas corridas.

### WP1 — Desenho fatorial (semanas 1–3) · obrigatório · **gate 1**
Grelha admissão × routing, no operador, Waxman e família ρ:

| | caminho mais curto | widest/bottleneck greedy | aprendido |
|---|---|---|---|
| **admissão greedy** | novo (barato) | Greedy (existe) | — |
| **admissão aprendida** | AC-only (existe) | **NOVO: AC-DQN + widest** | Joint / Factored (existem) |
| **trunk reservation clássica** | **NOVO** | **NOVO** | — |

- Decompõe o ganho total em (i) flexibilidade de routing e (ii) previsão na admissão. É o resultado que arruma a história toda.
- **Gate 1 (fim semana 3):** se AC-DQN + widest ≈ Joint em todo o lado → mensagem final: "aprende a admissão; usa ρ para decidir se precisas de flexibilidade de caminho (heurística chega)". Continua publicável e até mais limpo. Se o Joint ganhar com ρ alto → o routing aprendido tem valor próprio, também reportável.

### WP2 — Validação da lei / critério (semanas 2–6) · obrigatório · **gate 2**
1. **Família atual:** 8 níveis de ρ × 10 seeds; IC a 95% por bootstrap no declive e no ρ* (ponto de mudança de sinal).
2. **Duas famílias novas:** outro core (SNDlib/Topology Zoo), K∈{3,6}, e dois pontos de operação (greedy ≈0.35 e ≈0.60). Testar se o declive é invariante (normalizar o ganho por ρ).
3. **Previsão fora de amostra (a peça mais forte):** 8–10 topologias reais; calcular ρ, **escrever as previsões no repositório antes de treinar**, treinar, reportar erro absoluto médio. Isto transforma "ajuste" em "previsão".
4. **Sensibilidade de ρ** à política de referência usada para o estimar (greedy vs. AC-only) e ao horizonte de simulação; **custo**: segundos de estimação vs. horas de treino.
- **Gate 2 (fim semana 6):** se o erro fora de amostra for grande, rebaixar para "condição necessária" (ρ≈0 ⇒ sem ganho, que é demonstrável) e mudar "law" → "criterion".

### WP3 — Formalização leve (semanas 3–6) · fortemente recomendado
- **Lema 1 (estrutural):** com endpoints single-homed e ligações de acesso a ligar primeiro, ρ=0 — demonstração direta, já está em prosa.
- **Proposição 2:** limite superior da perda por passo de qualquer política fixada no caminho mais curto face a uma com escolha de caminho, em função de ρ, ao longo da mesma trajetória de estado. Explicar porque o declive ≈1 aparece (ganho relativo ≈ ρ ponderado pela probabilidade de admissão).
- Usar o skill de escrita no teu estilo para definições/provas.

### WP4 — Comparadores modernos (semanas 4–8) · obrigatório (pelo menos 1 e 2)
1. **Preços exponenciais com routing (AAP/OPA-style)**: preços por ligação, caminho mais barato, admitir se receita ≥ custo. Barato de implementar, é o estado da arte não-DRL e responde ao OPA (+32% sobre DRL). **Bónus:** verificar se ρ também prevê o ganho do routing para *este* controlador → mostra que ρ é propriedade do substrato, não do DQN.
2. **PPO com máscara** (conjunto e AC-only) num subconjunto: operador, Waxman, 3 níveis de ρ. Se o declive se mantiver, fecha a crítica "só DQN".
3. GNN: **não** para o TNSM; justificar (substrato fixo; transferência é trabalho futuro e complementar a Ahmadi et al.).

### WP5 — Realismo (semanas 6–9) · escolher 1 dos 2
- **(a) Multi-homing** (recomendado): redefinir ρ sobre conjuntos de ligações por caminho; uma família. Converte a principal limitação num resultado e liga a Gibbens–Kelly–Turner 1993.
- **(b) Procura não estacionária**: chegadas diurnas/MMPP; ρ por janela; o critério ainda prevê?
- Retirar qualquer menção a OVS que não seja usada; manter "flow-level" claramente assumido.

### WP6 — Escrita e submissão (semanas 8–12)
- Reescrever no teu estilo (skill research-paper-writer): Estado da Arte com a taxonomia A4, nova tabela comparativa, resultados reorganizados em torno de C1→C4.
- Libertar código, gerador de substratos e o ficheiro de previsões pré-registadas (Zenodo/GitHub) — conta muito no TNSM.
- Revisão interna "adversarial" dos números antes de submeter.

---

## Calendário

| Semana | Trabalho | Gate / entrega |
|---|---|---|
| 1 | WP0 + arranque WP1 | Plano de análise pré-registado |
| 2–3 | WP1; arranque WP2 (família atual, 10 seeds) | **Gate 1**: decomposição fatorial |
| 3–6 | WP2 (famílias novas, fora de amostra); WP3 | **Gate 2**: erro de previsão; decidir "law" e o nome de ρ |
| 4–8 | WP4 (preços exponenciais, PPO) | Comparadores fechados |
| 6–9 | WP5 (multi-homing) | **Congelar experiências fim semana 9** |
| 8–12 | WP6 escrita, figuras, código | Submissão TNSM |

## Orçamento de computação (estimativa — preciso do tempo por corrida)
Ordem de grandeza ~400 treinos novos: WP1 ~35–70 · WP2 ~340 (família 160, novas famílias 100, fora de amostra 80) · WP4 PPO ~50 · WP5 ~50. Se um treino demorar ~1 h e couberem ~4 em paralelo na GPU (rede pequena; o ambiente é provavelmente o gargalo de CPU), são ~4–5 dias de máquina por bloco. **Risco principal de calendário** — se o tempo por corrida for maior, cortar primeiro: K=6 nas famílias novas, depois PPO em ρ intermédio.

## Riscos e mitigação
| Risco | Mitigação |
|---|---|
| Revisor: "a lei é tautológica" | WP3 (formalizar o que é mecânico) + WP2.3 (o valor está na previsão sem treino e no ponto de mudança de sinal) |
| Revisor: "o DRL não bate o greedy" | Narrativa já escolhida (ρ como decisão); WP1 isola onde a aprendizagem paga (admissão) |
| Revisor: "comparadores fracos" | WP4.1 preços exponenciais + PPO; citar OPA/OACR2/Tao |
| Previsão fora de amostra falha | Gate 2: rebaixar para condição necessária + "criterion" |
| Atraso de computação | Cortes pré-definidos acima |

## Próximo passo imediato
1. Dizer-me o tempo médio de um treino e onde está o repositório NetworkSliceDRL (ligar a pasta), para confirmar o orçamento e escrever o plano de análise pré-registado.
2. Arrancar WP0 + WP1.
