# Identificacao de Dispositivos IoT por Fingerprint de Trafego Wi-Fi

## 1. Introducao

O crescimento do numero de dispositivos IoT em redes domesticas e corporativas amplia a superficie de ataque e dificulta o gerenciamento dos ativos conectados. Nesse contexto, tecnicas de fingerprinting baseadas em trafego de rede permitem identificar dispositivos a partir de padroes comportamentais observados nos pacotes transmitidos, sem depender exclusivamente de inventario manual ou de mecanismos tradicionais de descoberta.

Este projeto investiga a identificacao de dispositivos IoT por meio da analise de trafego Wi-Fi tratado, com foco na extracao de caracteristicas comportamentais e no treinamento de um classificador supervisionado. A proposta combina etapas de tratamento de dados, analise exploratoria, engenharia de atributos e validacao temporal do modelo.

## 2. Objetivo

O objetivo geral deste trabalho e identificar dispositivos IoT com base em fingerprints extraidas do trafego Wi-Fi.

Como objetivos especificos, o projeto busca:

1. Organizar os dados em etapas de bruto, tratado e features.
2. Extrair atributos relevantes de comportamento de rede, como tamanho de frame, intervalo entre pacotes e padrao de destinos.
3. Treinar um classificador Random Forest para distinguir dispositivos a partir dessas caracteristicas.
4. Avaliar o desempenho com validacao temporal, reduzindo risco de vazamento entre treino e teste.

## 3. Base de Dados

O conjunto principal utilizado no modelo esta em data/02 - Tratados/processed_training2.csv. A base contem 640889 linhas validas apos tratamento, com 15 dispositivos de origem observados e 6399 janelas agregadas usadas na modelagem. A organizacao do projeto foi mantida da seguinte forma:

1. data/01 - Nao-Tratados: capturas e arquivos brutos.
2. data/02 - Tratados: datasets preparados para analise e treino.
3. data/03 - Features: reservado para features derivadas.

As colunas centrais usadas no processamento foram:

1. frame.time_delta_displayed: intervalo entre quadros.
2. wlan.sa: endereco MAC de origem.
3. wlan.da: endereco MAC de destino.
4. wlan.fc.pwrmgt: indicador de power management.
5. frame.len: tamanho do frame.

## 4. Metodologia

### 4.1 Tratamento dos Dados

Os dados tratados foram carregados, convertidos para tipos numericos adequados e filtrados para remocao de registros com valores ausentes. A variavel de power management foi normalizada para representacao binaria.

### 4.2 Engenharia de Features

O trafego foi agregado em janelas fixas de 100 pacotes por dispositivo de origem. Para cada janela foram calculadas estatisticas descritivas que formam a fingerprint do dispositivo:

1. packet_count
2. frame_len_mean, frame_len_std, frame_len_min, frame_len_max
3. iat_mean, iat_std, iat_min, iat_max
4. pwrmgt_ratio
5. unique_destinations

Essa estrategia resume o comportamento do dispositivo ao longo do tempo e reduz ruido de pacotes individuais.

### 4.3 Modelo de Classificacao

Foi utilizado o algoritmo Random Forest com 300 arvores, random_state fixo em 42, balanceamento de classes e execucao paralela. O alvo de classificacao foi o endereco MAC de origem, representando o dispositivo emissor.

### 4.4 Estrategia de Validacao

Para reduzir o risco de overfitting e evitar avaliacao otimista, foram adotadas duas estrategias complementares:

1. Holdout temporal por dispositivo: as janelas de cada dispositivo foram ordenadas cronologicamente, usando os blocos iniciais para treino e os finais para teste.
2. Validacao cruzada temporal por blocos: as janelas de cada dispositivo foram divididas em 5 blocos contiguos, aplicando avaliacao walk-forward entre folds.

Essa abordagem e mais adequada do que um embaralhamento aleatorio, pois evita misturar amostras temporalmente vizinhas entre treino e teste.

## 5. Analise Exploratoria dos Dados

A analise exploratoria mostrou forte desbalanceamento entre dispositivos, com concentracao significativa de frames em poucos MACs de origem. Ainda assim, foram observados sinais discriminativos relevantes entre os dispositivos, principalmente em:

1. tamanho medio e maximo dos frames
2. intervalo medio entre pacotes
3. proporcao de power management
4. quantidade de destinos distintos por janela

As figuras exploratorias geradas pelo projeto foram:

1. reports/figures/top_sources.png
2. reports/figures/power_management_distribution.png
3. reports/figures/frame_length_by_device.png
4. reports/figures/window_feature_scatter.png
5. reports/figures/feature_correlation_heatmap.png

Essas visualizacoes indicam que diferentes dispositivos apresentam assinaturas de trafego suficientemente distintas para sustentar o processo de classificacao.

## 6. Resultados do Modelo

### 6.1 Holdout Temporal por Dispositivo

No holdout temporal, o modelo alcancou desempenho elevado, com acuracia global de 0.98. A media macro de F1 foi 0.96 e a media ponderada de F1 foi 0.98. Esses valores indicam boa capacidade de separacao entre os dispositivos com quantidade suficiente de janelas.

Os casos de maior dificuldade ficaram concentrados em algumas classes com menos amostras ou com padroes parcialmente proximos, como ac:41:6a:66:e6:f3 e 44:17:93:14:08:37.

### 6.2 Validacao Cruzada Temporal por Blocos

Na validacao cruzada temporal por blocos, os resultados por fold foram:

| Fold | Janelas de treino | Janelas de teste | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 1283 | 1279 | 0.9765 | 0.9715 | 0.9762 |
| 2 | 2562 | 1281 | 0.9813 | 0.9627 | 0.9814 |
| 3 | 3843 | 1279 | 0.9593 | 0.8903 | 0.9561 |
| 4 | 5122 | 1277 | 0.9765 | 0.9587 | 0.9760 |

As medias finais foram:

1. Accuracy media: 0.9734
2. Macro F1 medio: 0.9458
3. Weighted F1 medio: 0.9724

Mesmo com um criterio de avaliacao mais rigoroso, o modelo manteve desempenho consistente, o que reforca a presenca de padroes temporais e estruturais relevantes no trafego analisado.

## 7. Interpretacao das Features

As features mais importantes para o Random Forest foram:

1. frame_len_max: 0.1688
2. pwrmgt_ratio: 0.1390
3. frame_len_mean: 0.1291
4. unique_destinations: 0.1204
5. iat_mean: 0.1080

Esses resultados indicam que o modelo depende fortemente de caracteristicas de tamanho dos quadros, comportamento temporal e padrao de comunicacao com destinos. Em conjunto, essas variaveis formam uma assinatura robusta do dispositivo.

As figuras de suporte do modelo foram:

1. reports/figures/rf_confusion_matrix.png
2. reports/figures/rf_feature_importance.png
3. reports/figures/rf_cv_metrics_by_fold.png

## 8. Discussao

Os resultados indicam que a proposta e viavel para identificacao de dispositivos IoT a partir de trafego Wi-Fi tratado. A combinacao entre agregacao por janelas, atributos simples e Random Forest se mostrou suficiente para produzir desempenho elevado.

Ainda assim, existem limitacoes importantes:

1. O alvo atual do modelo corresponde ao MAC de origem, o que aproxima o problema de identificacao de emissores especificos, e nao necessariamente de categorias funcionais de dispositivos.
2. O conjunto e desbalanceado, com forte predominancia de alguns dispositivos.
3. O estudo foi conduzido sobre uma base especifica, o que limita a generalizacao para ambientes completamente distintos.

Como extensoes futuras, recomenda-se:

1. ampliar o numero de dispositivos e cenarios de captura
2. testar modelos adicionais e comparacoes sistematicas
3. incluir classificacao por tipo de dispositivo, fabricante ou familia
4. incorporar deteccao de dispositivos desconhecidos ou anomalias

## 9. Conclusao

O projeto demonstrou que e possivel identificar dispositivos IoT por fingerprint de trafego Wi-Fi com desempenho elevado, desde que os dados sejam devidamente tratados e agregados. A avaliacao temporal reforcou a confiabilidade do resultado e mostrou que o modelo nao depende apenas de um particionamento favoravel.

Em sintese, a abordagem adotada se mostrou tecnicamente consistente, reproduzivel e adequada como base para uma entrega academica sobre tratamento, analise e identificacao de dispositivos IoT em redes sem fio.
