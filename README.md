# Fleet Force — motor analítico unificado (determinístico + ML com gate)

Motor Python **único e modular** produzido a partir de `main.ipynb` e `fleet_force_inteligencia_ml_ufmg.ipynb` enviados pelo usuário, com as fontes UFMG 2025, Inmetro/PBEV 2009–2026 e ANP 2018–2026. **EPA não é usada.**

A camada determinística é a referência principal. O ML só assume um papel operacional quando há avaliação independente compatível com o uso pretendido e ganho demonstrado contra o baseline. Modelos reprovados continuam disponíveis para auditoria, mas **não geram alertas automáticos**.

## Estrutura

```text
fleet_force_unificado/
├── run.py                       # CLI de execução (um motor)
├── api.py                       # API REST de relatórios e uploads CSV
├── requirements.txt
├── requirements-api.txt
├── notebooks/fleet_force_unificado.ipynb
├── examples/fleet.service.ts    # Integração Angular
├── data/
│   ├── anp_precos_mensais_uf.csv
│   └── pbev_catalogo_historico_extraido_auditar.csv
├── fleet_force_engine/
│   ├── config.py                # Regras e limiares
│   ├── utils.py                 # Normalização e tipos
│   ├── ingest.py                # UFMG PDF ou CSV; dedup
│   ├── deterministic.py         # Regras US01/US02/US03, Pareto, custos
│   ├── references.py            # PBEV e ANP
│   ├── pbev_extract.py          # Extração histórica (não auditada)
│   ├── learning.py              # Modelos, backtests, gate
│   └── engine.py                # Orquestração, API de biblioteca, exports
└── tests/test_engine.py
```

## Instalação e uso

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

O pacote já inclui o PDF UFMG enviado (`data/despesa-veiculos-ufmg-2025.pdf`) e **agregações históricas** PBEV/ANP retiradas dos arquivos enviados, para execução rápida sem abrir 7 milhões de linhas em todas as análises.

```bash
python run.py --out outputs
```

Caso já tenha exportado `fleet_analysis_outputs/fleet_vehicles.csv` pelo `main.ipynb`, o CLI detecta e prioriza esse arquivo quando não é fornecido `--fleet-pdf`.

Para regenerar os datasets do ZIP original do usuário (lento e dependente dos layouts originais de PDF):

```bash
python run.py --fleet-pdf "despesa-veiculos-ufmg-2025.pdf" \
  --source-zip "fleet-force.zip" --rebuild-references --out outputs
```

**Não use `--rebuild-references` sem fornecer `--source-zip`.** A versão em cache do PBEV foi extraída por heurísticas de layout e requer auditoria antes de ser tratada como referência exata de versão/combustível.

Para utilizar outro UF (não deduzida da origem UFMG):

```bash
python run.py --fleet-pdf "despesa-veiculos-ufmg-2025.pdf" --uf SP
```

Para avaliar consumo com um período externo independente:

```bash
python run.py --fleet-pdf "despesa-veiculos-ufmg-2025.pdf" \
  --external-validation "frota_2026_validada.csv"
```

O CSV externo deve incluir `fleet_id`, `model`, `km`, `kmpl`, `reference_period` (por exemplo `2026`), além dos campos financeiros que existirem. Sem amostra externa independente e comparação satisfatória, o ML de consumo fica **desabilitado**, mesmo quando a validação cruzada indicar ganho. O CSV usado na validação externa **não** deve ser o mesmo conjunto de treino com nomes alterados. Recomendado observar combustível, versão, carga e trajeto no futuro.

## Lógica de referência

1. Calcula o custo/km, litros estimados, composição de gastos e integridade financeira.
2. Exclui do benchmark registros abaixo da distância mínima, consumo implausível, cobertura incompleta e duplicatas pendentes.
3. Compara cada veículo com a **mediana dos outros veículos do mesmo nome normalizado**, sem incluir o próprio veículo (leave-one-out). Grupo mínimo: 3 outros, identificação genérica não recebe confiança moderada.
4. Prioriza pelo desvio de consumo a partir de 10% **e** no mínimo 25 litros estimados, desde que a confiança seja moderada. Regra original mantida.
5. Analisa despesas e custo/km por categoria, com rótulo **exploratório**, pois categoria não garante operação equivalente.
6. Usa PBEV como **comparação exploratória adicional** apenas para leves com alguma associação de nome; não infere versão, ano ou combustível desconhecidos.
7. Faz três cenários independentes de preços para gasolina, etanol e diesel S10 usando a referência **histórica observada** da ANP (UF configurada; padrão MG). Os três valores não podem ser somados nem apresentados como economia realizada.
8. Treina regressão de consumo com nome do modelo, categoria e log(distância), sem usar custos/litros calculados da própria observação como features. Avalia por famílias de modelos completamente separadas entre treino/teste.
9. **Gate de consumo:** no mínimo 150 registros válidos, 25 famílias, redução mínima de MAE de 10%, nenhuma categoria com regressão superior a 10%, e **período externo independente com pelo menos 40 registros**. Como a UFMG traz apenas 2025, o modelo treinado **não** é promovido no dataset atual.
10. **Gate de preços:** previsão de um mês à frente em duas janelas cronológicas de 6 meses, contra `lag1` (último preço), redução mínima de MAE de 10% e proteção de segmentos por combustível. Se reprovado, usa mediana/média histórica observada, com mês explícito.
11. Detecta outliers com Isolation Forest em dados válidos, **sem usar esse rótulo como diagnóstico ou fonte de economia**.

## Dados e limitações

- UFMG: o relatório cobre 01/01/2025 a 31/12/2025. `liters_est` é derivado de `km / kmpl_reported`, não é uma medição de abastecimento independente. Há 232 linhas e 217 identificadores únicos; `GMF7102` aparece em duas linhas integralmente idênticas e exige confirmação documental, sendo contabilizado uma única vez para relatórios e excluído da comparação até auditoria.
- A soma de despesas depois de remover a duplicata propositalmente difere do total impresso no PDF. Esse resultado é esperado e não significa falha contábil. Consultar `fleet_auditoria_duplicatas.csv`.
- PBEV histórico: extração por layouts heurísticos, ano de homologação distinto do ano de operação; consulta por nome não substitui identificação de versão, combustível e configuração reais.
- ANP: dados históricos de postos, não preços negociados pela frota; série enviada possui lacunas semestrais. Série agregada incluída no pacote é uma cópia **derivada** do histórico enviado, não cotação em tempo real.
- ML treinado em UFMG anual: variações de carga, motorista, condições, combustível e rota não são observadas. Mesmo que tenha bom MAE de validação cruzada, não implica causalidade nem economia garantida.
- O mecanismo não faz previsões de manutenção preditiva, acidentes, multas ou pneus sem registros correspondentes.

## Artefatos de saída

- `fleet_vehicles.csv`: contrato completo por veículo, fontes e confiança.
- `fleet_priority_ranking.csv`: prioridades operacionais aprovadas, não outliers ML experimentais.
- `fleet_summary.csv` e `fleet_kpis.json`: KPIs financeiros e de cobertura.
- `fleet_comparison_groups.csv`: grupos equivalentes usados no benchmark.
- `fleet_public_comparisons.csv`: associações exploratórias PBEV.
- `fleet_data_coverage.csv`: campos efetivamente disponíveis.
- `fleet_resumo_categorias.csv`: análises por tipo.
- `fleet_pareto_custos.csv`: concentração financeira.
- `fleet_auditoria_duplicatas.csv`: auditoria documental.
- `fleet_scenarios.csv`: cenários de preços alternativos, NÃO aditivos.
- `fleet_ml_backtest_consumo.csv`: previsões *out-of-fold* de consumo, não promovidas.
- `anp_ml_backtest_temporal.csv`: preços reais, baseline e previsão do modelo.
- `fleet_model_metrics.json`: critérios objetivos e aprovação/rejeição.
- `fleet_manifest.json`: origem, hash, período e configuração.
- `models/*_candidate.joblib`: modelos candidatos; carregar **somente de origem confiável**, pois Joblib executa desserialização Python.

## API e Angular

A API serve relatórios já calculados por GET e analisa uploads CSV por POST, conforme descrito abaixo. Instale `requirements-api.txt` e execute:

```bash
uvicorn api:app --port 8000 --reload
```

Rotas: `GET /v1/fleet/summary`, `/v1/fleet/vehicles?offset=0&limit=50`, `/v1/fleet/vehicles/{fleet_id}`, `/v1/fleet/categories`, `/v1/fleet/priorities`, `/v1/fleet/reference-prices`, `/v1/fleet/models/quality`. Consulte `examples/fleet.service.ts` para o `HttpClient` Angular. A API fornecida é um exemplo sem autenticação; **implemente autenticação e autorização multi-tenant antes de publicação**.

## Testes

```bash
python -m unittest discover -s tests -v
```

O notebook `notebooks/fleet_force_unificado.ipynb` demonstra a execução, apresenta KPIs, rankings, validação e consulta de veículos, mas **não é um segundo motor**: ele chama `fleet_force_engine.run_engine()`.

Para desenvolvimento Angular usando a URL `/api/v1/fleet`, inclua `examples/proxy.conf.json` na configuração do `ng serve --proxy-config examples/proxy.conf.json` (copie o arquivo para a pasta do seu projeto Angular). O proxy remove o prefixo `/api` antes de chegar ao FastAPI. Em produção, configure o gateway de modo equivalente e aplique autenticação e segregação de dados por cliente.

## Análise direta por upload CSV

`POST /v1/fleet/analyze` recebe `file` (`UploadFile`, multipart) e `uf`
(opcional, sigla de UF brasileira, padrão `MG`). Retorna HTTP 200 com toda a
análise do arquivo enviado na mesma requisição. Não exige executar o CLI,
notebooks ou consultar outra rota. As rotas GET continuam lendo os relatórios
existentes em `outputs/`.

```bash
pip install -r requirements-api.txt
uvicorn api:app --reload --port 8000

curl -X POST \
  "http://localhost:8000/v1/fleet/analyze" \
  -F "file=@minha_frota.csv;type=text/csv" \
  -F "uf=MG"
```

O formulário de upload também está disponível em `/docs`.

### Formatos e validação

- Extensão `.csv`; UTF-8 com ou sem BOM; separador `;` ou `,`.
- Limite padrão: **10 MiB (10 × 1024 × 1024 bytes)** de conteúdo do arquivo.
  Configure `FLEET_MAX_UPLOAD_BYTES` no ambiente antes de iniciar a API.
- Campos obrigatórios: `fleet_id`, `model`, `km`. Identificador e modelo não
  podem estar vazios. Identificadores permanecem strings, com espaços externos
  removidos e letras em maiúsculas (zeros iniciais preservados).
- Campos opcionais: `kmpl` (ou `kmpl_reported`), `os_count`, `labor_internal`,
  `labor_external`, `parts`, `tires`, `fuel_expense`, `lubricant`, `wash`,
  `total_expense`, `cost_per_km`. Os campos de cobertura e litros de exportações
  normalizadas continuam aceitos pelo ingest existente.
- Colunas ausentes e células numéricas vazias (`NA`, `N/A`, `ND`, `N/D`, `null`,
  `NaN` ou `-`) representam indisponibilidade, nunca uma despesa zero.
  Distância zero e km/L numericamente implausível são preservados com avisos de
  qualidade; texto numérico inválido gera 422.
- Números aceitos: `1234.56`, `1234,56`, `1.234,56`, `1,234.56`, com prefixo
  opcional `R$`. Valores com ambos os separadores precisam ter agrupamento
  válido. Valores ambíguos como `1,234` e `1.234` são rejeitados: envie `1234`
  para mil duzentos e trinta e quatro ou `1.2340` para o decimal. Valores com
  vírgula precisam estar entre aspas em CSV separado por vírgulas.
- Linhas precisam ter exatamente a quantidade de campos do cabeçalho.
  Cabeçalhos repetidos, inclusive aliases do mesmo campo, são rejeitados.

O formato UFMG é mapeado explicitamente por cabeçalho, independentemente da
ordem; aceita variações de acentos, maiúsculas, pontuação e espaços:

| Cabeçalho UFMG | Campo interno |
| --- | --- |
| Frota | fleet_id |
| modelo | model |
| Qde. OS | os_count |
| Interna | labor_internal |
| Externa | labor_external |
| R$ Peças | parts |
| R$ Pneus | tires |
| R$ Abast. | fuel_expense |
| R$ Lubri. | lubricant |
| R$ Lavag. | wash |
| Km Percorrido | km |
| KM/L | kmpl_reported |
| Total Despesas | total_expense |
| $/km | cost_per_km |

Exemplo normalizado mínimo:

```csv
fleet_id,model,km,kmpl,total_expense
001,ONIX 1.0,2000,10,1000
```

Linhas brutas completas usam a consolidação existente: somas de valores
conhecidos e km/L ponderado por litros estimados. Duplicatas integralmente
idênticas são contadas uma vez, preservadas em `quality.duplicates` com status
`pending_source_confirmation`, e retiradas do benchmark até revisão. IDs
repetidos em CSV agregado incompleto são rejeitados, pois não há informação
suficiente para aplicar a regra bruta com segurança. Para consolidar, envie
todas as colunas do relatório bruto (podendo deixar despesas opcionais vazias).

### Resposta

Exemplo de **recorte** da resposta para o CSV mínimo acima:

```json
{
  "analysis_id": "request-scoped-uuid",
  "status": "completed",
  "metadata": {
    "source": "uploaded_csv",
    "input_format": "normalized",
    "price_reference_state": "MG",
    "currency": "BRL",
    "distance_unit": "km",
    "efficiency_unit": "km_per_liter"
  },
  "summary": {"vehicles": 1, "total_km": 2000, "total_expense_brl": 1000},
  "vehicles": [{
    "fleet_id": "001", "kmpl": 10, "liters_est": 200,
    "cost_per_km": 0.5, "quality_status": "eligible",
    "peer_status": "insufficient_other_peers", "priority_alert": false
  }],
  "priorities": [],
  "ml": {
    "enabled": false,
    "status": "deterministic_fallback",
    "reason": "No trusted independently validated model artifact is available",
    "used_for_operational_alerts": false
  }
}
```

A resposta completa inclui `categories`, `quality.issues`, `quality.duplicates`,
`quality.coverage`, `comparisons.internal`, `comparisons.public_references`,
`financial.expense_distribution`, `financial.pareto`, `financial.fuel_scenarios`,
`reference_prices` e `ml.models`, além das tabelas de backtest e previsão ML.
Todos os indicadores por veículo são retornados, incluindo fontes de referência,
qualidade, status de comparação e flags de prioridade. Nomes de colunas do motor
são preservados; os agrupamentos JSON são específicos da API. Despesas são BRL,
distâncias km e eficiência km/L. `quality.coverage` informa contagens disponíveis
por campo; distribuição financeira informa cobertura por rubrica. Não há
truncamento de veículos. Números indisponíveis, NaN e infinitos viram `null`.

Os CSVs processados em `data/anp_precos_mensais_uf.csv` e
`data/pbev_catalogo_historico_extraido_auditar.csv` são resolvidos em relação ao
aplicativo e mantidos em cache com invalidação por tamanho/data de modificação.
Não há downloads nem processamento de ZIPs no upload. Referências ausentes ou
ilegíveis desabilitam as comparações dependentes: consulte `metadata.availability`,
`metadata.price_source`, `metadata.pbev_source` e os status por veículo. As tabelas
indisponíveis são arrays vazios e os indicadores permanecem presentes como `null`.
Menos de três outros veículos elegíveis do mesmo modelo impede o benchmark.

O upload usa `run_engine(..., inference_only=True)` e **não treina nem promove
modelos**. Este repositório exporta candidatos, mas não possui registro confiável
de artefatos aprovados; por isso a API declara fallback determinístico, sem
carregar arquivos Joblib candidatos. Os gates e o treinamento padrão do CLI
permanecem ativos no fluxo CLI. A análise de anomalias que exige ajuste de
Isolation Forest também é explicitamente `not_evaluated_inference_only` no upload.
PBEV permanece exploratório; cenários ANP são alternativas não aditivas e não
representam economia garantida ou diagnóstico mecânico.

O processamento bloqueante roda no thread pool, com contexto em memória próprio
para cada requisição. Não escreve uploads, relatórios, referências ou modelos.
O arquivo multipart é fechado ao terminar, inclusive em erros, liberando qualquer
arquivo temporário criado pelo parser HTTP. O nome original nunca é usado como
caminho. Cada resposta recebe um UUID exclusivo.

### Erros

Todas as falhas do upload usam `{"error":{"code":"...","message":"...","details":{}}}`.

| HTTP | Código | Motivo |
| --- | --- | --- |
| 400 | INVALID_UF / INVALID_REQUEST | UF inválida ou multipart inválido |
| 413 | UPLOAD_TOO_LARGE | Limite de tamanho excedido |
| 415 | UNSUPPORTED_FORMAT | Extensão diferente de CSV |
| 422 | EMPTY_FILE | Arquivo vazio ou somente cabeçalho |
| 422 | INVALID_ENCODING | Conteúdo não UTF-8 |
| 422 | MALFORMED_CSV | Aspas, linhas ou estrutura inválidas |
| 422 | INVALID_CSV_SCHEMA | Colunas obrigatórias ausentes ou cabeçalhos repetidos |
| 422 | INVALID_NUMERIC_DATA | Número inválido, ambíguo ou infinito |
| 422 | INVALID_CSV_DATA / INVALID_REQUEST | Dados de veículo inválidos ou arquivo ausente |
| 500 | ANALYSIS_FAILED | Falha interna inesperada, sem detalhes internos públicos |

Exemplo:

```json
{"error":{"code":"INVALID_CSV_SCHEMA","message":"The uploaded CSV is missing required columns.","details":{"missing_columns":["fleet_id","km","model"],"accepted_formats":["ufmg","normalized"]}}}
```

Execute todos os testes existentes e de upload após instalar as dependências API:

```bash
python -m unittest discover -s tests -v
```
