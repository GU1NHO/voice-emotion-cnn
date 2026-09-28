# Voice Emotion Recognition - CNN

Aplicação demonstrável para classificar emoção em fala: o navegador grava ou envia áudio, a API cria e exibe um espectrograma log-Mel e uma CNN devolve a classe e as probabilidades.

## Decisão de produto

O **site local** é a melhor entrega principal, pois demonstra ponta a ponta: captura de voz -> pré-processamento -> CNN -> predição. Um documento colaborativo deve acompanhar o repositório somente para relatório, decisões experimentais, resultados e link do vídeo de demonstração. Ele não substitui a aplicação, pois não grava nem executa a CNN de forma reprodutível.

## Estrutura

```text
app/          API, extração log-Mel e duas CNNs PyTorch
web/          página estática com MediaRecorder
data/manifest.csv  path,label,speaker; permite divisão por locutor
artifacts/    pesos e rótulos gerados no treinamento (não versionados)
train.py      treino scratch ou transfer, avaliação speaker-independent
```

## Arquiteturas requeridas

- **`scratch`:** quatro blocos `Conv2d 3x3 -> BatchNorm -> ReLU -> Conv2d 3x3 -> BatchNorm -> ReLU -> MaxPool2d 2x2`, com 32, 64, 128 e 256 canais. Global average pooling, dropout e uma camada final classificam as emoções. Durante o treino há máscaras de frequência/tempo (SpecAugment) e deslocamento temporal aleatório.
- **`transfer`:** ResNet-18 pré-treinada em ImageNet. Cada espectrograma log-Mel é repetido nos três canais RGB; todos os pesos do backbone são congelados e somente a camada `fc` final é otimizada.

## Execução

Pré-requisitos: Python 3.10+ e `ffmpeg` no PATH (necessário para gravações WebM do navegador).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python prepare_emodb.py
# O script lê data/emodb/wav e cria manifest.csv com path,label,speaker.
python train.py --architecture scratch --epochs 25
python train.py --architecture transfer --epochs 25
Copy-Item artifacts/model-transfer.pt artifacts/model.pt
uvicorn app.main:app --reload
```

Execute a validação estrutural antes do primeiro treino:

```powershell
pytest -q
```

Abra `http://127.0.0.1:8000`. Use uma base pública, por exemplo EmoDB, RAVDESS ou CREMA-D. Preencha `data/manifest.csv` com `path,label,speaker`; o `speaker` é obrigatório para impedir que a voz de uma mesma pessoa apareça no treino e no teste. Não versione o conjunto de dados nem pesos grandes: disponibilize-os por link com licença e instruções de reprodução.

## Plano de desenvolvimento e critérios de entrega

1. **Dados:** escolher uma única base pública (recomendação: RAVDESS, pois disponibiliza IDs de ator), normalizar suas emoções e gerar `manifest.csv`.
2. **CNN do zero:** executar `--architecture scratch`; registrar accuracy e macro-F1 no conjunto de locutores nunca vistos.
3. **Transfer learning:** executar `--architecture transfer`, que usa ResNet-18 pré-treinada em ImageNet. O espectrograma é repetido nos três canais RGB e o backbone permanece congelado; somente a camada final é treinada. Comparar as duas CNNs honestamente no mesmo split.
4. **App e demo:** selecionar o melhor modelo, copiar seu peso para `artifacts/model.pt` e testar o fluxo gravar -> espectrograma -> previsão.
5. **Deploy e pacote:** publicar a API, testar o microfone na URL pública, acrescentar resultados, quem fez o quê e limitações ao README.

## Limites responsáveis

A emoção inferida é uma estimativa de padrões acústicos do conjunto de treino, não um diagnóstico psicológico nem um atributo confiável de uma pessoa. Use apenas áudios autorizados e deixe essa limitação explícita na apresentação.

## Equipe e contribuições

Preencher antes da entrega, com nomes e contribuições verificáveis:

| Pessoa | Contribuição |
|---|---|
| Nome 1 | Dados e avaliação speaker-independent |
| Nome 2 | CNN do zero e transfer learning |
| Nome 3 | App, deploy e documentação |

## Checklist antes de enviar

- [ ] As duas CNNs foram treinadas: uma do zero e uma com transfer learning
- [ ] A CNN do zero usa quatro blocos duplos 3x3, BatchNorm, ReLU, max-pooling e SpecAugment
- [ ] A CNN de transferência usa ResNet-18 congelada e espectrograma repetido em RGB
- [ ] Ambiente limpo reproduz instalação, treino e execução
- [ ] `results-scratch.json` e `results-transfer.json` comprovam divisão por locutor e macro-F1
- [ ] Dados, pesos e credenciais não foram enviados ao Git
- [ ] Demonstração testada em navegador com microfone permitido
- [ ] URL do repositório e URL pública do app prontas para envio em 29/09/2026
