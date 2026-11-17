# API multimodal de apoio à revisão

Protótipo acadêmico sem interface gráfica. Transcreve áudio e organiza o texto de
prescrições para **revisão humana**. Não diagnostica, valida doses nem recomenda
tratamentos. Utilize somente dados sintéticos ou devidamente desidentificados e
autorizados para os serviços Azure.

## Preparação

Requisitos: Python 3.11 ou superior, Redis e recursos Azure Speech, Azure AI
Language e Azure OpenAI com implantação compatível com a API Responses e saída
JSON. O Docker com Compose é necessário para iniciar o Redis local fornecido.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edite `.env` localmente, sem compartilhar credenciais:

| Variável | Uso |
|---|---|
| `AZURE_SPEECH_KEY`, `AZURE_SPEECH_REGION` | Transcrição de áudio. |
| `AZURE_LANGUAGE_ENDPOINT`, `AZURE_LANGUAGE_KEY` | Menções clínicas e sentimento textual. |
| `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT` | Análise de PDFs com texto. O endpoint deve terminar em `/openai/v1/`; o modelo é o nome da implantação. |
| `SENTIMENT_ENABLED` | `false` por padrão. Habilite somente após verificar o recurso e o suporte ao idioma. |
| `CRITICAL_TERMS` | Lista JSON de termos para busca literal, em português e/ou inglês. |
| `DATABASE_PATH`, `UPLOAD_DIR` | Caminhos locais do SQLite e dos arquivos temporários. |
| `REDIS_URL`, `RQ_QUEUE_NAME` | Conexão Redis e nome da fila compartilhada pela API e pelos workers. |
| `RQ_JOB_TIMEOUT_SECONDS` | Limite configurado de execução de cada job; padrão de 1.800 segundos. Veja as limitações no macOS abaixo. |
| `RQ_RESULT_TTL_SECONDS`, `RQ_FAILURE_TTL_SECONDS` | Retenção no Redis de jobs concluídos e falhos; padrões de 3.600 e 86.400 segundos. |
| `MAX_AUDIO_BYTES`, `MAX_AUDIO_DURATION_SECONDS`, `MAX_PDF_BYTES`, `MAX_PDF_PAGES`, `MAX_PDF_TEXT_CHARS` | Limites de entrada. |
| `MAX_CONCURRENT_JOBS` | Número de processos RQ iniciados por pool de workers; padrão de 2. |
| `PDF_MODEL_CHUNK_CHARS`, `AZURE_REQUEST_TIMEOUT_SECONDS`, `SPEECH_TIMEOUT_SECONDS` | Limites de processamento e tempo. |

O SDK aceita endpoints HTTPS oficiais de Azure Language e Azure OpenAI/Foundry;
não há provedor generativo alternativo. As chaves são necessárias para os fluxos
correspondentes; `/health` e PDFs sem texto funcionam sem elas. Sem Azure
Language, a etapa clínica é explicitamente indisponível e o áudio requer revisão.

Para iniciar um Redis local com persistência em volume Docker:

```bash
docker compose up -d
docker compose exec redis redis-cli ping
```

Se usou a configuração anterior, pare o contêiner com
`docker stop multimodal-redis` antes de iniciar o Compose. O volume existente
`multimodal-redis-data` é reutilizado. Para parar o Redis sem apagar os dados,
use `docker compose stop`; não use `docker compose down --volumes` se quiser
preservá-los. A configuração habilita AOF, mas uma interrupção abrupta ainda
pode causar perda dos dados mais recentes.

Na raiz do projeto, com o ambiente virtual ativado, inicie o pool de workers e
a API em terminais separados:

```bash
# Terminal 1
python -m src.jobs.worker

# Terminal 2
uvicorn src.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Ambos devem usar o mesmo `.env`, Redis e fila, além de acessar os mesmos
arquivos de upload e o SQLite. Cada pool inicia `MAX_CONCURRENT_JOBS` processos;
iniciar mais de um pool aumenta a concorrência total. A documentação interativa
está em `http://127.0.0.1:8000/docs`. Não inclua `decode_responses` na URL:
a conexão usa respostas binárias e o RQ serializa os jobs em JSON.

No macOS, o pool usa `spawn` e `SimpleWorker` para evitar `fork` na execução dos
jobs. Nesse modo, cada job roda no próprio processo do worker, sem isolamento por
job nem heartbeats periódicos durante o processamento. O timeout usa sinais e
pode não interromper prontamente chamadas nativas; uma interrupção abrupta pode
deixar o job em processamento até a limpeza do RQ.

Os jobs e seus estados ficam no Redis e sobrevivem ao reinício da API enquanto
os dados forem preservados. Jobs concluídos expiram após uma hora por padrão e
falhos após 24 horas; depois disso, a consulta ao job retorna `404`. As análises
e os alertas permanecem no SQLite. Guarde o `analysis_id` retornado pelo job.
Sem Redis, o envio e a consulta de jobs retornam `503`; `/health` verifica
apenas que o processo da API está ativo.

**Segurança:** não há autenticação, autorização nem TLS. Execute apenas
localmente, sem expor a API ou o Redis à rede pública. A imagem Redis local não
exige senha e aceita conexões da rede interna do Docker; não a utilize em redes
não confiáveis. Quem puder escrever no Redis pode executar funções com os
privilégios do worker, mesmo com serialização JSON. Proteja o volume do Redis,
que contém identificadores, caminhos e metadados dos jobs. O SQLite contém
resultados e transcrições em texto claro; proteja seu diretório e faça sua própria
gestão de retenção.

O upload permanece em disco enquanto aguarda um worker e é removido ao final do
processamento. Se o worker não estiver ativo ou sofrer interrupção abrupta, os
arquivos podem permanecer em `UPLOAD_DIR`: confira os jobs antes de removê-los.
Uma falha de rede durante o envio pode deixar o enfileiramento indeterminado;
reenvios não são deduplicados. Não habilite logs detalhados dos SDKs com dados
clínicos.

## Amostras sintéticas

```bash
# PDFs fictícios, inclusive uma página sem texto; em macOS, --audio gera WAVs.
python examples/gerar_amostras.py --audio
```

Em outros sistemas, omita `--audio` e gere WAVs sintéticos usando um sintetizador
de fala autorizado. No macOS é possível alterar as vozes com `--voice-pt` e
`--voice-en`. O script não sobrescreve amostras existentes; use `--output` para
outro diretório. Os arquivos ficam em `examples/generated/`, ignorado pelo Git.

Em `examples/receitas-ficticias/` há seis PDFs com texto selecionável, baseados
nos modelos de receituário simples e de controle especial. As amostras incluem
um ou dois medicamentos, campos omitidos e frequências divergentes. Pacientes e
profissionais são fictícios; os nomes, apresentações e exemplos de posologia
para adultos foram consultados nas bulas de
[Cimegripe](https://consultaremedios.com.br/cimegripe/bula),
[paracetamol](https://consultaremedios.com.br/paracetamol/bula),
[loratadina](https://consultaremedios.com.br/loratadina/bula),
[dipirona](https://consultaremedios.com.br/dipirona-monoidratada/bula),
[azitromicina](https://consultaremedios.com.br/azitromicina/bula) e
[amoxicilina](https://consultaremedios.com.br/amoxicilina/bula), em 14/11/2026.

As frequências divergentes são um conflito proposital para teste. Nenhum PDF é
uma prescrição válida ou orientação de tratamento, e o modelo visual não indica
o tipo de receituário legalmente exigido para cada medicamento. Antibióticos não
tratam gripe, que é causada por vírus.

## Uso da API

Os áudios aceitos são **WAV PCM mono, 16 bits, 8 ou 16 kHz**, não vazios, de
até 300 segundos e 20 MB por padrão. O PDF deve ser válido, não criptografado,
com até 50 páginas e 10 MB por padrão. Os limites podem ser ajustados em `.env`.

```bash
# Os exemplos abaixo são exclusivamente fictícios.
curl -s -X POST http://127.0.0.1:8000/api/v1/cases/demo-01/audio \
  -F file=@examples/generated/consulta-pt.wav -F language=pt-BR

curl -s -X POST http://127.0.0.1:8000/api/v1/cases/demo-01/audio \
  -F file=@examples/generated/consultation-en.wav -F language=en-US

curl -s -X POST http://127.0.0.1:8000/api/v1/cases/demo-01/prescriptions \
  -F file=@examples/generated/prescricao-pt.pdf
```

Os uploads retornam `202` com `job_id`, `case_id`, `status: queued` e
`status_url`. O idioma do áudio é `pt-BR` ou `en-US`; o padrão é `en-US`.

```bash
curl -s http://127.0.0.1:8000/api/v1/jobs/JOB_ID
curl -s http://127.0.0.1:8000/api/v1/analyses/ANALYSIS_ID
curl -s 'http://127.0.0.1:8000/api/v1/cases/demo-01/alerts?limit=20&offset=0'
curl -s http://127.0.0.1:8000/api/v1/health
```

Os estados da API são `queued`, `processing`, `completed`, `needs_review` e
`failed`; são derivados do estado do RQ e do resultado da análise.
Em `completed` ou `needs_review`, o resultado inclui `analysis_id`,
`analysis_url` e a análise persistida. Em `failed`, `error` contém somente código
e mensagem seguros. `completed` significa processamento concluído, **não**
validação clínica. Alertas são menções para revisão, inclusive menções negadas;
veja a citação, o timestamp e as afirmações de negação/incerteza.

Para PDF, cada campo de medicamento contém `value` e `evidence` com página e
citação literal. O resumo é extrativo. Campos desconhecidos permanecem `null` e
são listados em `uncertainties`. Uma página sem texto selecionável, uma resposta
inválida ou um documento dividido em blocos resulta em `needs_review`; não há
OCR nem preenchimento por plausibilidade. O modelo pode omitir ou interpretar
informações incorretamente, apesar da validação de citações.

O sentimento, quando habilitado, refere-se somente ao **texto**; não mede emoção
pela voz nem constitui diagnóstico ou alerta clínico. Não são calculados
indicadores acústicos. A análise por segmentos pode perder relações entre trechos.
Tabelas e colunas do PDF podem ser extraídas fora da ordem visual; confira o
documento original.
Consulte [RELATORIO_TECNICO.md](RELATORIO_TECNICO.md) para limitações e roteiro.

## Testes

```bash
python -m pytest -q tests/unit

# Opcional: use um Redis de teste; os jobs usam uma fila exclusiva.
TEST_REDIS_URL=redis://127.0.0.1:6379/15 python -m pytest -q tests/integration
```

Os testes usam dados fictícios e não chamam Azure. O teste de integração inicia
um pool de workers real, exercita a API e o Redis e não executa `FLUSHDB`.
Sem `TEST_REDIS_URL`, ele é ignorado. A disponibilidade real dos recursos, dos
idiomas e do modelo deve ser confirmada na conta e região utilizadas.

## Licenças

As dependências diretas estão fixadas em `requirements.txt`. Em particular,
PyMuPDF é distribuído sob AGPL-3.0 ou licença comercial Artifex; verifique a
adequação ao seu uso. O SDK Azure Speech e os serviços Azure têm termos próprios.
Confira também os termos da imagem oficial Redis.
