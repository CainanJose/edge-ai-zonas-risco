# Edge AI — Monitoramento de Risco em Zonas de Máquinas Pesadas

Protótipo de **visão computacional embarcada (Edge AI)** para detectar pessoas e sinalizar sua presença em regiões virtuais de risco de um ambiente industrial. Desenvolvido como parte do desafio técnico do **FIT Sorocaba — Analista de Capacitação Tecnológica II / Professor de Inteligência Artificial**.

A aplicação utiliza **YOLO26n**, **Python**, **OpenCV** e **FastAPI**. A inferência ocorre **localmente em CPU**, sem solicitar inferência a um serviço em nuvem. A API é distribuída em Docker, com build multi-arquitetura **AMD64/ARM64** para possibilitar adaptação à **Raspberry Pi 5**.

> **Escopo e segurança:** este é um protótipo acadêmico de **detecção de presença em zonas fixas**, não um sistema certificado de segurança industrial. Ele não detecta automaticamente movimentos de máquinas, gestos perigosos ou incidentes ao longo do tempo. “SEGURO” significa apenas que a pessoa detectada está fora dos polígonos configurados.

## 1. Demonstração visual

| Imagem de entrada | Resultado da inferência |
|:---:|:---:|
| ![Imagem original com empilhadeira e trabalhadores](examples/imagens/test.jpg) | ![Pessoas identificadas com zonas virtuais e alerta](docs/images/deposito-alerta.png) |

O cenário `deposito` contém zonas vermelha e amarela definidas para a imagem de exemplo. As caixas indicam pessoas detectadas, e o alerta é determinado por sua posição nas zonas.

## 2. Como funciona

```mermaid
flowchart LR
    A[Imagem / quadro de vídeo] --> B[YOLO26n: detecção de pessoas]
    B --> C[Caixas e ponto inferior central]
    C --> D{Posição nos polígonos}
    D -->|Zona vermelha| E[PERIGO]
    D -->|Zona amarela| F[ATENCAO]
    D -->|Fora das zonas| G[SEGURO]
    E --> H[Alerta global]
    F --> H
    G --> H
    H --> I[JSON / PNG / monitoramento]
```

1. A imagem é recebida pela API ou capturada da webcam/vídeo.
2. O YOLO26n identifica objetos da classe COCO `person` (`classes=[0]`, confiança mínima de `0.35`, `device="cpu"`). A biblioteca Ultralytics realiza o pré-processamento e o pós-processamento próprios do detector.
3. Para cada pessoa, o sistema calcula o **centro inferior da bounding box**, como aproximação da posição dos pés no piso.
4. O ponto é comparado a polígonos de risco definidos em coordenadas normalizadas; a zona vermelha tem prioridade sobre a amarela.
5. O alerta global segue a maior criticidade observada: `CRITICO` > `ATENCAO` > `NORMAL`.
6. A saída pode ser um JSON, uma imagem PNG anotada ou uma janela de monitoramento local.

**Risco por pessoa:** `SEGURO`, `ATENCAO` ou `PERIGO`. **Nível do alerta global:** `NORMAL`, `ATENCAO` ou `CRITICO`. Os alertas são **visuais e informativos**; não existe acionamento físico de sirene ou relé nesta versão.

### Cenários disponíveis

| Cenário | Aplicação | API HTTP | Monitoramento ao vivo |
|---|---|:---:|:---:|
| `padrao` | Exemplo genérico | Sim | Sim |
| `deposito` | Região próxima a empilhadeira | Sim | Sim |
| `corredor` | Faixa de circulação industrial | Sim | Sim |
| `webcam` | Demonstração com terço esquerdo vermelho, central amarelo e direito fora das zonas | Não | Sim |

As zonas são parametrizadas para uma **posição de câmera fixa**. Se a câmera mudar de lugar ou enquadramento, é necessário recalibrar os polígonos em `app/zones.py`.

## 3. Organização do repositório

```text
app/
  main.py                    # API FastAPI
  vision.py                  # Inferência e desenho das detecções
  zones.py                   # Polígonos e classificação geométrica
  alertas.py                 # Alerta global por criticidade
  monitorar_ao_vivo.py       # Webcam ou vídeo com janela OpenCV
  monitorar_video.py         # Vídeo anotado e relatório de desempenho
  criar_video_teste.py       # Geração de vídeo estático para teste
  testar_*.py                # Scripts auxiliares de demonstração
tests/                       # Testes automatizados
examples/imagens/test.jpg    # Imagem de entrada para a API
docs/images/deposito-alerta.png # Exemplo de resultado anotado
.github/workflows/docker-multiarch.yml
Dockerfile
docker-compose.yml
requirements.txt
yolo26n.pt                  # Pesos locais do modelo
```

## 4. Executar com Docker (forma recomendada para avaliação)

**Pré-requisitos:** Git e Docker Desktop (Windows) ou Docker Engine com Compose (Linux). Não é necessária uma Raspberry Pi física.

```powershell
git clone https://github.com/CainanJose/edge-ai-zonas-risco.git
cd edge-ai-zonas-risco
docker compose up --build -d
```

A primeira construção instala dependências e pode demorar. Depois, verifique o serviço:

```powershell
docker compose ps
curl.exe http://localhost:8000/health
```

Resposta esperada:

```json
{"status":"ok"}
```

Documentação interativa: **http://localhost:8000/docs**. A API utiliza a porta **8000**. Para acompanhar os logs ou encerrar:

```powershell
docker compose logs -f
docker compose down
```

O `docker-compose.yml` define `restart: always`. O container disponibiliza **a API HTTP em modo headless**; a janela gráfica de webcam descrita abaixo é uma execução local separada.

## 5. Testar os endpoints HTTP

Os comandos abaixo estão adaptados ao **PowerShell**. Execute na raiz do repositório, com o container em execução. No Windows, use `curl.exe` para chamar o executável curl em vez do alias do PowerShell. **No Linux/macOS**, use `curl` com os mesmos argumentos (e `mkdir -p outputs` no lugar de `New-Item`).

O arquivo é enviado no campo multipart `file`. O parâmetro opcional de query `cenario` aceita `padrao` (padrão), `deposito` ou `corredor`.

### 5.1. JSON — `POST /infer/json`

```powershell
curl.exe -X POST "http://localhost:8000/infer/json?cenario=deposito" -F "file=@examples/imagens/test.jpg"
```

A resposta contém cenário, modelo, dimensões, número de pessoas, contagens, tempo de inferência, alerta global e uma lista de detecções (classe, confiança, caixa, ponto de referência e risco).

**Exemplo ilustrativo do formato da resposta** (valores fictícios, apenas para mostrar a estrutura; não é a saída de `test.jpg`):

```json
{
  "cenario": "deposito",
  "modelo": "yolo26n.pt",
  "largura": 640,
  "altura": 480,
  "total_pessoas": 1,
  "contagem": {"SEGURO": 0, "ATENCAO": 1, "PERIGO": 0},
  "tempo_inferencia_ms": 198.7,
  "alerta": {
    "ativo": true,
    "nivel": "ATENCAO",
    "mensagem": "1 pessoa(s) na zona amarela",
    "resposta": "AVISO_VISUAL"
  },
  "deteccoes": [
    {
      "classe": "person",
      "confianca": 0.84,
      "bbox": [415, 120, 485, 350],
      "ponto_referencia": [450, 350],
      "risco": "ATENCAO"
    }
  ]
}
```

### 5.2. PNG anotado — `POST /infer/image`

```powershell
New-Item -ItemType Directory -Force outputs | Out-Null
curl.exe -X POST "http://localhost:8000/infer/image?cenario=deposito" -F "file=@examples/imagens/test.jpg" --output "outputs/resultado.png"
```

A resposta tem `Content-Type: image/png`. Abra `outputs/resultado.png` para visualizar os polígonos, as caixas delimitadoras, o rótulo de risco com a confiança de cada pessoa e a faixa de alerta.

**Validação de entradas:** arquivo vazio ou não decodificável retorna HTTP `400`; um cenário desconhecido retorna `422`.

## 6. Demonstração técnica

A demonstração exigida no desafio apresenta os **dois endpoints obrigatórios com a aplicação em Docker**, ao vivo ou em vídeo de até cinco minutos. O roteiro previsto é:

1. Subir o serviço com `docker compose up --build -d` e mostrar `docker compose ps` (política `restart: always`).
2. Consultar `GET /health`.
3. Enviar `examples/imagens/test.jpg` para `POST /infer/json?cenario=deposito` e comentar o JSON (pessoas, confiança, risco, alerta e tempo de inferência).
4. Enviar a mesma imagem para `POST /infer/image?cenario=deposito` e abrir o PNG anotado.
5. Opcionalmente, mostrar a documentação interativa em `/docs`.

**Vídeo da demonstração:** _link a ser adicionado._

O monitoramento por webcam (seção 7) é uma demonstração complementar e não substitui a apresentação dos endpoints em Docker.

## 7. Executar localmente e usar webcam

Para a demonstração com janela gráfica do OpenCV, execute **fora do container**. Os comandos a seguir utilizam Windows PowerShell; rode-os na raiz do repositório:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Os pesos `yolo26n.pt` estão no repositório; por isso, a execução deve partir da raiz do projeto.

Para iniciar a API localmente, caso prefira não usar Docker:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Para monitorar a câmera integrada ou USB (os índices podem variar conforme o dispositivo):

```powershell
# Normalmente, camera integrada
python -m app.monitorar_ao_vivo --fonte 0 --cenario webcam --largura 640

# Normalmente, webcam USB externa
python -m app.monitorar_ao_vivo --fonte 1 --cenario webcam --largura 640
```

Pressione **Q** ou **Esc** na janela para encerrar. No Windows, a aplicação tenta os backends Media Foundation (`CAP_MSMF`), DirectShow (`CAP_DSHOW`) e automático (`CAP_ANY`), nessa ordem, e só aceita a câmera se conseguir ler um quadro. O rodapé da janela mostra o FPS efetivo, o tempo de processamento do quadro e o número de pessoas.

O mesmo script aceita um arquivo de vídeo em `--fonte` (com `--repetir` para reprodução em laço). Nesse modo, a exibição é sincronizada com o FPS original do vídeo; por isso, para medir desempenho, utilize o `monitorar_video.py`, descrito a seguir.

### Processar um vídeo de teste

O script abaixo cria um MP4 de **30 quadros repetidos da imagem de exemplo**. Ele serve para exercitar o fluxo e medir processamento, **não para avaliar movimentação real entre zonas**.

```powershell
python -m app.criar_video_teste
python -m app.monitorar_video --fonte examples/videos/teste_estatico.mp4 --cenario deposito --saida outputs/monitoramento_teste.mp4 --largura 640 --intervalo 1 --max-quadros 30
```

Os resultados são gravados em `outputs/monitoramento_teste.mp4` e `outputs/monitoramento_teste.json`. A pasta `outputs/` contém artefatos gerados localmente e não é versionada.

## 8. Escolha do modelo

Foi selecionado o **Ultralytics YOLO26n**, variante *nano* da família YOLO26, pré-treinada no conjunto COCO. O COCO já inclui a classe `person`, que é a única necessária para o problema; por isso, o modelo é usado **sem treinamento adicional**, filtrando apenas essa classe (`classes=[0]`).

**Por que a variante nano:** o problema exige localizar pessoas em uma cena, não distinguir categorias finas de objetos. Nesse contexto, o fator limitante em um dispositivo de borda é o custo computacional por quadro. O YOLO26n é a variante mais compacta da família: tem cerca de 4 vezes menos parâmetros que o YOLO26s e 8 vezes menos que o YOLO26m, com a menor latência de referência em CPU. Em troca, apresenta o menor mAP da família. Para um protótipo que roda em CPU, sem GPU dedicada, e que visa adaptação à Raspberry Pi 5, esse é o compromisso mais adequado entre custo computacional e capacidade de detecção.

**Referência oficial da Ultralytics** (detecção no COCO val2017, entrada de 640 px):

| Modelo | Parâmetros (M) | FLOPs (B) | mAP50-95 | Latência CPU ONNX (ms) |
|---|---:|---:|---:|---:|
| **YOLO26n** | **2,4** | **5,5** | **40,9** | **38,9 ± 0,7** |
| YOLO26s | 9,5 | 20,9 | 48,6 | 87,2 ± 0,9 |
| YOLO26m | 20,4 | 68,4 | 53,1 | 220,0 ± 1,4 |

Fonte: [Ultralytics — YOLO26](https://docs.ultralytics.com/models/yolo26/), consultada em 06/11/2026. Segundo a documentação, as latências foram medidas em formato ONNX, em uma CPU Intel Xeon de 2,00 GHz, com a cabeça sem NMS (`nms=False`).

Esses valores são **referências do fabricante**, obtidas em outro hardware e em formato ONNX. **Não são medições deste projeto**, que executa os pesos PyTorch (`.pt`) diretamente (ver seção 9). O mAP no COCO também não comprova a precisão do modelo em cenas industriais.

**Inferência sem NMS:** a arquitetura YOLO26 oferece um modo de inferência ponta a ponta, sem a etapa de supressão de não máximos (NMS), o que simplifica o pós-processamento em dispositivos de borda. Este projeto usa a chamada padrão `predict` da biblioteca Ultralytics e **não mediu** separadamente o ganho desse modo.

**Limiar de confiança:** o valor `0.35` é um ponto de partida do protótipo e **não foi calibrado experimentalmente**. Não foram medidos mAP, precisão ou recall em um conjunto industrial rotulado. A confiança retornada para cada detecção não equivale à acurácia global do modelo.

## 9. Desempenho e viabilidade em CPU

### Resultado medido

Medição local com o `monitorar_video.py`, usando o vídeo de teste de 30 quadros derivado de `test.jpg` (comandos na seção 7). Valores registrados no relatório JSON gerado pelo script:

| Métrica | Valor |
|---|---:|
| Quadros lidos / processados | 30 / 30 |
| Largura processada | 640 px |
| FPS nominal do vídeo de entrada | 10 FPS |
| **FPS efetivo de processamento** | **4,48 FPS** |
| **Tempo médio de processamento por quadro** | **219,14 ms** |
| Tempo total | 6,69 s |
| Quadros com alerta `ATENCAO` | 30 de 30 |

**Ambiente:** notebook Windows, inferência em CPU com PyTorch, sem GPU. O modelo do processador e a quantidade de RAM não foram registrados junto a essa medição. Para registrá-los no Windows:

```powershell
Get-CimInstance Win32_Processor | Select-Object Name
Get-CimInstance Win32_ComputerSystem | Select-Object TotalPhysicalMemory
```

**Estes números são do notebook. Não são medições na Raspberry Pi 5.**

### O que cada métrica representa

| Métrica | Onde aparece | O que mede |
|---|---|---|
| Tempo de inferência do YOLO | `tempo_inferencia_ms` na resposta da API | Apenas a chamada `predict` do YOLO26n (pré-processamento, rede neural e pós-processamento internos da Ultralytics) |
| Tempo de processamento por quadro | `latencia_media_ms` no relatório do vídeo | A função `analisar_imagem` completa: inferência, classificação nas zonas e desenho da imagem anotada |
| FPS efetivo | `fps_processamento` no relatório do vídeo | Quadros processados divididos pelo tempo total, incluindo também leitura do vídeo e gravação do MP4 |

Nenhuma dessas métricas mede a latência completa entre a captura da câmera e a exibição do alerta. O `monitorar_video.py` processa os quadros sem espera artificial, por isso é o script indicado para benchmark.

Por ser um ensaio curto, com um único quadro repetido, a média inclui o aquecimento inicial do modelo. A contagem de alertas é **por quadro**, não por incidentes distintos.

### Viabilidade

O resultado mostra que o pipeline completo (detecção, classificação nas zonas, alerta e anotação) funciona com **processamento local em CPU**, sem GPU e sem serviço em nuvem, a cerca de 4,5 quadros por segundo em 640 px. Essa taxa é adequada para um **monitoramento demonstrativo** de presença em zonas.

O resultado **não permite garantir** um tempo de resposta adequado para aplicações industriais de segurança. Isso exigiria requisitos de tempo de reação definidos para o processo, medições no hardware final e validação em campo. Latência e taxa de quadros variam conforme o processador, a resolução, o modelo, o formato de execução (PyTorch, ONNX, NCNN) e a carga do sistema.

Para repetir o experimento, use os comandos da seção 7, registre CPU, RAM, sistema operacional e versões das bibliotecas, e faça várias execuções após o aquecimento.

## 10. Docker multi-arquitetura e Raspberry Pi 5

O workflow [Docker Multi-Arquitetura](.github/workflows/docker-multiarch.yml) utiliza QEMU e Docker Buildx para construir e publicar a imagem em `linux/amd64` e `linux/arm64` no GitHub Container Registry (GHCR):

```text
ghcr.io/cainanjose/edge-ai-zonas-risco:latest
```

O Dockerfile escolhe o PyTorch conforme a arquitetura: no AMD64, instala a versão somente CPU (índice `download.pytorch.org/whl/cpu`, sem as bibliotecas CUDA); no ARM64, usa as wheels padrão do PyPI. Os pesos `yolo26n.pt` são copiados para a imagem, então o container não depende de download do modelo em tempo de execução.

**Evidência do build:** [GitHub Actions — execução #2 do workflow "Docker Multi-Arquitetura"](https://github.com/CainanJose/edge-ai-zonas-risco/actions/runs/42208927148), concluída com sucesso em 06/11/2026 (commit `84c027b`). A última etapa do workflow executa `docker buildx imagetools inspect` para listar as plataformas publicadas. Para consultar o manifesto (requer acesso ao pacote no GHCR):

```bash
docker buildx imagetools inspect ghcr.io/cainanjose/edge-ai-zonas-risco:latest
```

### Estado da validação

| Etapa | Situação |
|---|---|
| Build e publicação da imagem `linux/amd64` e `linux/arm64` | Realizado no GitHub Actions |
| Execução do container e dos endpoints em AMD64 | Realizada localmente com Docker |
| Execução do container ARM64 | Não realizada |
| Inferência e medição de desempenho na Raspberry Pi 5 | Não realizada |

O build ARM64 bem-sucedido mostra que a imagem pode ser construída para a arquitetura da Raspberry Pi 5. Ele não substitui um teste de execução nem uma medição de FPS na placa.

### Adaptações para Raspberry Pi 5

- **Sistema e imagem:** usar Raspberry Pi OS 64 bits (ARM64) com Docker. O Docker seleciona automaticamente a variante `linux/arm64` da imagem.
- **Inferência em CPU:** a Raspberry Pi 5 não possui GPU compatível com a aceleração deste pipeline PyTorch. A inferência ocorre nos núcleos ARM64 da CPU, e por isso o desempenho esperado é inferior ao de um notebook.
- **Ajustes sem alterar a arquitetura:** reduzir a resolução de entrada (`--largura`) e processar um a cada N quadros (`--intervalo`) nos scripts de monitoramento.
- **Otimizações futuras:** exportar o modelo para **NCNN** (formato recomendado no guia da Ultralytics para Raspberry Pi) ou **ONNX** e reduzir o `imgsz` do modelo. Essas otimizações **não foram implementadas nem medidas** neste projeto.
- **Aceleradores externos:** módulos de NPU podem ser avaliados em uma evolução futura. Eles exigem hardware e runtime específicos e não são necessários para o funcionamento atual.
- **Operação:** usar fonte de alimentação adequada, dissipação de calor e câmera CSI ou USB compatível; monitorar temperatura, uso de CPU e memória.

**Procedimento proposto para validação na placa:** em Raspberry Pi OS 64 bits, clonar o repositório e executar `docker compose up --build -d` (ou baixar a imagem do GHCR). Em seguida, verificar `/health`, testar `/infer/json` e `/infer/image` com `test.jpg` e medir latência, FPS e uso de recursos (`docker stats`) em várias execuções após o aquecimento.

Referência externa: [Ultralytics — guia para Raspberry Pi](https://docs.ultralytics.com/guides/raspberry-pi/). Os benchmarks publicados ali usam outras configurações e não substituem medições deste pipeline.

## 11. Testes

A suíte cobre os endpoints da API (incluindo inferência real com o modelo), a classificação geométrica nas zonas e as regras de alerta. Em execução local em 06/11/2026, **os 12 testes passaram**.

```powershell
python -m pip install pytest httpx
python -m pytest -q
```

Os testes validam contratos e regras principais, mas não são uma avaliação estatística de precisão do modelo em ambiente industrial.

## 12. Limitações e próximos passos

- A geometria das zonas depende da posição e perspectiva da câmera; mudar o enquadramento exige recalibração.
- O ponto inferior da caixa aproxima a posição dos pés e pode ser incorreto em cenas com oclusão, corte da pessoa ou perspectiva desfavorável.
- Pessoas distantes, mal iluminadas ou parcialmente ocultas podem não ser detectadas.
- Não existe rastreamento de identidades ou contagem de ocorrências distintas; os alertas são recalculados a cada quadro.
- Não há acionamento físico de mecanismos de segurança, validação em uma fábrica real ou certificação para uso industrial.
- A execução e a performance na Raspberry Pi 5 permanecem como proposta de validação futura.

**Possíveis evoluções (não necessárias ao desafio atual):** ampliar o conjunto de cenas industriais rotuladas, medir precisão e recall, avaliar otimização NCNN/ONNX e integrar mecanismos de alerta externos sob requisitos de segurança apropriados.

## 13. Referências e licenciamento

- [Documentação do YOLO26 — Ultralytics](https://docs.ultralytics.com/models/yolo26/)
- [Guia Ultralytics para Raspberry Pi](https://docs.ultralytics.com/guides/raspberry-pi/)
- [Histórico dos builds multi-arquitetura](https://github.com/CainanJose/edge-ai-zonas-risco/actions)

**Licenças de terceiros:** a biblioteca Ultralytics e os pesos pré-treinados do YOLO26 são distribuídos pela Ultralytics sob a licença AGPL-3.0, com opção de licença comercial. O uso fora do contexto acadêmico e demonstrativo deve observar esses termos.

**Imagens de exemplo:** `examples/imagens/test.jpg` e o resultado derivado `docs/images/deposito-alerta.png` são usados apenas para demonstrar o funcionamento do protótipo. A procedência e a licença de redistribuição da imagem original não estão documentadas neste repositório.
