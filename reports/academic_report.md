# Identificacao de Dispositivos IoT por Fingerprint de Trafego Wi-Fi

## 1. Introducao

O crescimento do numero de dispositivos IoT em ambientes domesticos e corporativos amplia a superficie de ataque e torna mais complexo o gerenciamento dos ativos conectados. Nesse cenario, tecnicas de fingerprinting baseadas em trafego de rede surgem como alternativa promissora para identificar dispositivos a partir de padroes comportamentais observados nos pacotes transmitidos, reduzindo a dependencia de inventario manual e de mecanismos tradicionais de descoberta.

Este trabalho investiga a identificacao de dispositivos IoT por meio da analise de trafego Wi-Fi previamente tratado, com enfase na extracao de caracteristicas comportamentais e no treinamento de classificadores supervisionados. A proposta foi estruturada em etapas de tratamento de dados, analise exploratoria, engenharia de atributos, comparacao entre algoritmos e validacao rigorosa do desempenho obtido.

## 2. Objetivo

O objetivo geral deste trabalho e identificar dispositivos IoT com base em fingerprints extraidas do trafego Wi-Fi.

De forma mais especifica, o projeto busca:

1. Organizar os dados em etapas de bruto, tratado e features.
2. Extrair atributos relevantes de comportamento de rede, como tamanho de frame, intervalo entre pacotes e padrao de destinos.
3. Treinar e comparar classificadores supervisionados distintos para distinguir dispositivos a partir dessas caracteristicas.
4. Avaliar o desempenho com particionamento temporal de holdout e validacao cruzada estratificada, reduzindo vies por desbalanceamento entre classes.

## 3. Base de Dados

O conjunto principal utilizado na modelagem esta em data/02 - Tratados/processed_training2.csv. A base contem 640889 linhas validas apos o tratamento, com 15 dispositivos de origem observados e 6399 janelas agregadas efetivamente utilizadas no processo de modelagem. A organizacao adotada no projeto foi a seguinte:

1. data/01 - Nao-Tratados: capturas e arquivos brutos.
2. data/02 - Tratados: datasets preparados para analise e treino.
3. data/03 - Features: reservado para features derivadas.

As colunas centrais utilizadas no processamento foram:

1. frame.time_delta_displayed: intervalo entre quadros.
2. wlan.sa: endereco MAC de origem.
3. wlan.da: endereco MAC de destino.
4. wlan.fc.pwrmgt: indicador de power management.
5. frame.len: tamanho do frame.

## 4. Metodologia

### 4.1 Tratamento dos Dados

Os dados tratados foram carregados, convertidos para tipos numericos adequados e filtrados para remocao de registros com valores ausentes. Alem disso, a variavel de power management foi normalizada para uma representacao binaria, de modo a facilitar sua utilizacao na etapa de modelagem.

### 4.2 Engenharia de Features

O trafego foi agregado em janelas fixas de 100 pacotes por dispositivo de origem. Para cada janela foram calculadas estatisticas descritivas que formam a fingerprint do dispositivo:

1. packet_count
2. frame_len_mean, frame_len_std, frame_len_min, frame_len_max
3. iat_mean, iat_std, iat_min, iat_max
4. pwrmgt_ratio
5. unique_destinations

Essa estrategia resume o comportamento do dispositivo ao longo do tempo, reduz o ruido inerente a pacotes individuais e torna mais estavel a representacao usada pelos classificadores.

### 4.3 Modelos de Classificacao

Foram avaliados tres classificadores supervisionados sobre o mesmo conjunto de features e a mesma variavel alvo, o endereco MAC de origem:

1. Random Forest: escolhido por sua robustez, capacidade de lidar bem com relacoes nao lineares e interpretabilidade por importancia das features.
2. K-Nearest Neighbors: utilizado como baseline baseado em proximidade no espaco de atributos, apos padronizacao das variaveis.
3. Support Vector Machine com kernel RBF: utilizado para verificar se um modelo de margem maxima com fronteiras nao lineares melhoraria a separacao entre dispositivos.

Cada algoritmo foi definido com uma configuracao inicial simples e, em seguida, submetido a uma busca automatica de hiperparametros com RandomizedSearchCV sobre os dados de treino. O criterio de selecao adotado foi o Macro F1 medio, por ser mais apropriado em um contexto com desbalanceamento entre classes. Essa etapa reduziu a arbitrariedade na escolha dos hiperparametros e tornou a comparacao entre modelos metodologicamente mais consistente.

Os espacos de busca considerados foram:

1. Random Forest: numero de arvores, profundidade maxima, minimo de amostras por divisao, minimo de amostras por folha e estrategia de selecao de atributos.
2. KNN: numero de vizinhos, tipo de ponderacao e metrica Minkowski com p = 1 ou p = 2.
3. SVM: parametro C, estrategia de gamma e tipo de kernel.

Os melhores hiperparametros encontrados foram:

1. Random Forest: n_estimators = 200, max_depth = 10, min_samples_split = 2, min_samples_leaf = 1, max_features = log2.
2. KNN: n_neighbors = 7, weights = distance, p = 1.
3. SVM: C = 10.0, gamma = scale, kernel = rbf.

### 4.4 Estrategia de Validacao

Para reduzir o risco de overfitting e evitar uma avaliacao excessivamente otimista, foram adotadas duas estrategias complementares:

1. Holdout temporal por dispositivo: as janelas de cada dispositivo foram ordenadas cronologicamente, usando os blocos iniciais para treino e os finais para teste.
2. Validacao cruzada estratificada: as janelas agregadas foram divididas em 5 folds com preservacao da proporcao de classes, garantindo distribuicao mais equilibrada entre treino e teste.

Essa combinacao permite avaliar a capacidade de generalizacao temporal no holdout principal e, ao mesmo tempo, controlar melhor o efeito do desbalanceamento na comparacao entre folds.

## 5. Analise Exploratoria dos Dados

A analise exploratoria evidenciou forte desbalanceamento entre dispositivos, com concentracao significativa de frames em poucos MACs de origem. Ainda assim, foram observados sinais discriminativos relevantes entre os dispositivos, principalmente em:

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

Essas visualizacoes indicam que diferentes dispositivos apresentam assinaturas de trafego suficientemente distintas para sustentar o processo de classificacao supervisionada.

## 6. Resultados do Modelo

### 6.1 Holdout Temporal por Dispositivo

No holdout temporal, o Random Forest tunado alcancou desempenho elevado, com accuracy global de 0.9743, Macro F1 de 0.9527 e Weighted F1 de 0.9739. Esses valores indicam boa capacidade de separacao entre os dispositivos com quantidade suficiente de janelas, mesmo quando a ordem temporal dentro de cada equipamento e preservada.

Os casos de maior dificuldade ficaram concentrados em algumas classes com menos amostras ou com padroes parcialmente proximos, como ac:41:6a:66:e6:f3 e 44:17:93:14:08:37.

### 6.2 Validacao Cruzada Estratificada

Na validacao cruzada estratificada, os resultados por fold foram:

| Fold | Janelas de treino | Janelas de teste | Accuracy | Macro F1 | Weighted F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 5119 | 1280 | 0.9875 | 0.9818 | 0.9876 |
| 2 | 5119 | 1280 | 0.9828 | 0.9810 | 0.9827 |
| 3 | 5119 | 1280 | 0.9773 | 0.9751 | 0.9777 |
| 4 | 5119 | 1280 | 0.9797 | 0.9625 | 0.9795 |
| 5 | 5120 | 1279 | 0.9844 | 0.9814 | 0.9844 |

As medias finais foram:

1. Accuracy media: 0.9823
2. Macro F1 medio: 0.9764
3. Weighted F1 medio: 0.9824

Mesmo sob um criterio mais rigoroso em termos de balanceamento entre classes, o modelo manteve desempenho consistente, o que reforca a existencia de padroes discriminativos relevantes no trafego analisado.

### 6.3 Comparacao entre Algoritmos

A comparacao entre os tres algoritmos avaliados mostrou vantagem clara para o Random Forest tanto no holdout temporal quanto na validacao cruzada estratificada.

| Modelo | Holdout Accuracy | Holdout Macro F1 | CV Accuracy Media | CV Macro F1 Medio |
| --- | ---: | ---: | ---: | ---: |
| Random Forest | 0.9743 | 0.9527 | 0.9823 | 0.9764 |
| SVM | 0.9665 | 0.9334 | 0.9612 | 0.9525 |
| KNN | 0.9680 | 0.9444 | 0.9687 | 0.9446 |

O Random Forest apresentou o melhor desempenho medio e a melhor estabilidade global. O SVM ficou em segundo lugar, com resultado competitivo, mas ainda inferior ao Random Forest. O KNN apresentou o menor desempenho entre os tres, o que sugere maior sensibilidade a variacoes locais e a sobreposicoes entre classes nesse espaco de atributos.

Essa comparacao fortalece a justificativa metodologica do trabalho, pois demonstra que o algoritmo selecionado para o resultado final nao foi escolhido de forma arbitraria, mas sim com base em desempenho empirico sob o mesmo protocolo de validacao.

Adicionalmente, a etapa de tuning automatico apresentou os seguintes Macro F1 medios durante a busca de hiperparametros:

1. Random Forest: 0.9784
2. SVM: 0.9550
3. KNN: 0.9450

Esses valores reforcam que o Random Forest permaneceu superior mesmo apos a otimizacao sistematica dos tres modelos sob um criterio uniforme de selecao.

## 7. Interpretacao das Features

As features mais importantes para o Random Forest foram:

1. frame_len_max: 0.1623
2. pwrmgt_ratio: 0.1476
3. frame_len_mean: 0.1275
4. unique_destinations: 0.1222
5. iat_mean: 0.1147

Esses resultados indicam que o modelo depende fortemente de caracteristicas ligadas ao tamanho dos quadros, ao comportamento temporal e ao padrao de comunicacao com destinos. Em conjunto, essas variaveis formam uma assinatura robusta do dispositivo.

As figuras de suporte do modelo foram:

1. reports/figures/rf_confusion_matrix.png
2. reports/figures/rf_feature_importance.png
3. reports/figures/rf_cv_metrics_by_fold.png

## 8. Discussao

Os resultados indicam que a proposta e viavel para a identificacao de dispositivos IoT a partir de trafego Wi-Fi tratado. A combinacao entre agregacao por janelas, atributos simples e algoritmos supervisionados mostrou-se eficaz, com destaque para o Random Forest, que superou KNN e SVM na comparacao experimental.

Apesar dos resultados positivos, existem limitacoes importantes:

1. O alvo atual do modelo corresponde ao MAC de origem, o que aproxima o problema de identificacao de emissores especificos, e nao necessariamente de categorias funcionais de dispositivos.
2. O conjunto e desbalanceado, com forte predominancia de alguns dispositivos.
3. O estudo foi conduzido sobre uma base especifica, o que limita a generalizacao para ambientes completamente distintos.

Como extensoes futuras, recomenda-se:

1. ampliar o numero de dispositivos e cenarios de captura
2. testar modelos adicionais e comparacoes sistematicas
3. incluir classificacao por tipo de dispositivo, fabricante ou familia
4. incorporar deteccao de dispositivos desconhecidos ou anomalias

## 9. Conclusao

O projeto demonstrou que e possivel identificar dispositivos IoT por fingerprint de trafego Wi-Fi com desempenho elevado, desde que os dados sejam devidamente tratados e agregados. A combinacao entre holdout temporal e validacao cruzada estratificada reforcou a confiabilidade dos resultados e mostrou que o desempenho obtido nao depende apenas de um particionamento favoravel.

Em sintese, a abordagem adotada se mostrou tecnicamente consistente, reproduzivel e adequada para uma entrega academica centrada em tratamento, analise e identificacao de dispositivos IoT em redes sem fio. Alem disso, a comparacao entre algoritmos e a selecao automatica de hiperparametros fortaleceram a justificativa metodologica do modelo final apresentado.
