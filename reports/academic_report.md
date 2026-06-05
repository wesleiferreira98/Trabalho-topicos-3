# Identificacao de Dispositivos IoT por Fingerprint de Trafego Wi-Fi

## 1. Introducao

O crescimento do numero de dispositivos IoT em ambientes domesticos e corporativos amplia a superficie de ataque das redes e dificulta o inventario manual de ativos conectados. Nesse contexto, tecnicas de fingerprinting baseadas em trafego surgem como alternativa promissora para identificar dispositivos a partir de padroes de comunicacao observados na rede, mesmo sem depender exclusivamente de mecanismos tradicionais de descoberta.

Este trabalho investiga a identificacao de dispositivos IoT por meio da analise de trafego Wi-Fi previamente tratado, com enfase na extracao de caracteristicas comportamentais, comparacao entre classificadores supervisionados e avaliacao sob diferentes protocolos de validacao. A proposta foi estruturada em etapas de tratamento dos dados, analise exploratoria, engenharia de atributos, tuning automatico de hiperparametros, comparacao entre algoritmos e validacao com preservacao temporal.

## 2. Objetivo

O objetivo geral deste trabalho e identificar dispositivos IoT com base em fingerprints extraidas do trafego Wi-Fi.

De forma mais especifica, o projeto busca:

1. Organizar os dados em etapas de bruto, tratado e features.
2. Extrair atributos relevantes de comportamento de rede, como tamanho de frame, intervalo entre pacotes e padrao de destinos.
3. Treinar e comparar classificadores supervisionados distintos para distinguir dispositivos a partir dessas caracteristicas.
4. Avaliar o desempenho por meio de holdout temporal, validacao cruzada estratificada, validacao temporal expanding e validacao temporal sliding.

## 3. Base de Dados

O conjunto principal utilizado na modelagem esta em data/02 - Tratados/processed_training2.csv. A base contem 640889 linhas validas apos o tratamento, com 15 dispositivos de origem observados e 6399 janelas agregadas efetivamente utilizadas no processo de modelagem.

A organizacao adotada no projeto foi a seguinte:

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

Os dados tratados foram carregados, convertidos para tipos numericos adequados e filtrados para remocao de registros com valores ausentes. Alem disso, a variavel de power management foi normalizada para uma representacao binaria, de modo a facilitar seu uso na etapa de modelagem.

### 4.2 Engenharia de Features

O trafego foi agregado em janelas fixas de 100 pacotes por dispositivo de origem. Para cada janela foram calculadas estatisticas descritivas que formam a fingerprint do dispositivo:

1. packet_count
2. frame_len_mean, frame_len_std, frame_len_min, frame_len_max
3. iat_mean, iat_std, iat_min, iat_max
4. pwrmgt_ratio
5. unique_destinations

Essa estrategia resume o comportamento do dispositivo ao longo do tempo, reduz o ruido inerente a pacotes individuais e produz uma representacao mais estavel para os classificadores. Ao mesmo tempo, trata-se de uma decisao de projeto: os resultados obtidos dependem dessa granularidade de 100 pacotes e nao implicam que essa seja a unica segmentacao valida para o problema.

### 4.3 Modelos de Classificacao

Foram avaliados tres classificadores supervisionados sobre o mesmo conjunto de features e a mesma variavel alvo, o endereco MAC de origem:

1. Random Forest: escolhido por sua robustez, capacidade de modelar relacoes nao lineares e possibilidade de interpretacao por importancia das features.
2. K-Nearest Neighbors: utilizado como baseline baseado em proximidade no espaco de atributos, apos padronizacao das variaveis.
3. Support Vector Machine com kernel RBF: utilizado para verificar se um modelo de margem maxima com fronteiras nao lineares melhoraria a separacao entre dispositivos.

Cada algoritmo foi submetido a uma busca automatica de hiperparametros com RandomizedSearchCV sobre os dados de treino. O criterio de selecao adotado foi o Macro F1 medio, por ser mais adequado em um contexto com desbalanceamento entre classes. Essa etapa reduz a arbitrariedade na escolha dos hiperparametros e fortalece a consistencia metodologica da comparacao entre modelos.

Os melhores hiperparametros encontrados foram:

1. Random Forest: n_estimators = 200, max_depth = 10, min_samples_split = 2, min_samples_leaf = 1, max_features = log2.
2. KNN: n_neighbors = 7, weights = distance, p = 1.
3. SVM: C = 10.0, gamma = scale, kernel = rbf.

### 4.4 Estrategia de Validacao

Para reduzir o risco de overfitting e evitar uma avaliacao excessivamente otimista, foram adotadas quatro estrategias complementares:

1. Holdout temporal por dispositivo: as janelas de cada dispositivo foram ordenadas cronologicamente, usando os blocos iniciais para treino e os finais para teste.
2. Validacao cruzada estratificada: as janelas agregadas foram divididas em 5 folds com preservacao da proporcao de classes.
3. Validacao temporal expanding: cada fold usa um historico acumulado de janelas para treino e avalia o bloco futuro seguinte.
4. Validacao temporal sliding: cada fold usa apenas uma janela temporal recente fixa para treino e avalia o bloco seguinte.

Essa combinacao permite observar o comportamento do modelo em cenarios com diferentes graus de rigor. A validacao estratificada funciona como referencia comparativa sem preservacao temporal estrita, enquanto as validacoes temporal expanding e temporal sliding aproximam melhor o uso em sequencias futuras. Em especial, a sliding e a mais exigente por restringir o historico disponivel ao modelo.

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

## 6. Resultados

### 6.1 Holdout Temporal por Dispositivo

No holdout temporal, o Random Forest tunado alcancou accuracy de 0.9743, Macro F1 de 0.9527 e Weighted F1 de 0.9739. Esses resultados indicam boa capacidade de separacao entre os dispositivos mesmo quando a ordem temporal dentro de cada equipamento e preservada.

Os casos de maior dificuldade ficaram concentrados em algumas classes com menos amostras ou com padroes parcialmente proximos, o que e coerente com o desbalanceamento observado na base.

### 6.2 Validacao Cruzada Estratificada

Na validacao cruzada estratificada, o Random Forest alcancou medias de:

1. Accuracy: 0.9823
2. Macro F1: 0.9764
3. Weighted F1: 0.9824

Esse e o cenario mais favoravel entre os protocolos avaliados, pois os folds preservam a distribuicao das classes, mas nao impõem a mesma restricao temporal dos protocolos expanding e sliding.

### 6.3 Validacao Temporal Expanding

Na validacao temporal expanding, o Random Forest alcancou medias de:

1. Accuracy: 0.9743
2. Macro F1: 0.9548
3. Weighted F1: 0.9740

Esse resultado se manteve muito proximo do holdout temporal, o que sugere estabilidade quando o modelo pode acumular historico anterior para prever blocos futuros.

### 6.4 Validacao Temporal Sliding

Na validacao temporal sliding, o Random Forest alcancou medias de:

1. Accuracy: 0.9599
2. Macro F1: 0.9147
3. Weighted F1: 0.9570

Embora ainda elevadas, essas metricas sao inferiores as obtidas na validacao estratificada e na temporal expanding. Esse comportamento era esperado, pois a sliding restringe o treinamento a uma janela temporal recente fixa, tornando a tarefa mais desafiadora e mais proxima de um cenario de adaptacao continua com memoria limitada.

### 6.5 Comparacao entre Algoritmos

Os resultados agregados dos tres modelos avaliados foram os seguintes:

| Modelo | Holdout Macro F1 | CV Estratificada Macro F1 | CV Temporal Macro F1 | CV Sliding Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| Random Forest | 0.9527 | 0.9764 | 0.9548 | 0.9147 |
| SVM | 0.9334 | 0.9525 | 0.9174 | 0.8638 |
| KNN | 0.9444 | 0.9446 | 0.8927 | 0.8688 |

O Random Forest apresentou o melhor desempenho em todos os protocolos de avaliacao. O SVM ficou em segundo lugar na maior parte dos cenarios, enquanto o KNN mostrou comportamento competitivo, mas com maior degradacao sob restricoes temporais.

Essa comparacao fortalece a justificativa metodologica do trabalho, pois demonstra que o modelo final nao foi escolhido de forma arbitraria. O algoritmo adotado foi selecionado apos comparacao empirica sob os mesmos dados, o mesmo processo de tuning e os mesmos protocolos de avaliacao.

Adicionalmente, a etapa de tuning automatico apresentou os seguintes Macro F1 medios durante a busca de hiperparametros:

1. Random Forest: 0.9784
2. SVM: 0.9550
3. KNN: 0.9450

## 7. Interpretacao dos Resultados

Os resultados indicam que as fingerprints construidas a partir de tamanho de frame, dinamica temporal e padrao de destinos capturam caracteristicas suficientemente discriminativas para separar dispositivos observados na base. A vantagem do Random Forest sugere que ha relacoes nao lineares entre essas variaveis e a identidade do emissor, o que e coerente com a heterogeneidade do trafego IoT.

Tambem e importante notar que a queda de desempenho entre a validacao estratificada e a validacao temporal sliding nao enfraquece o trabalho; pelo contrario, torna a avaliacao mais honesta. Em termos de defesa academica, esse comportamento demonstra que o projeto nao depende apenas de um protocolo conveniente, mas explicita como o desempenho muda quando o problema se torna metodologicamente mais rigoroso.

## 8. Limitacoes e Ameacas a Validade

Apesar dos resultados positivos, existem limitacoes importantes que precisam ser reconhecidas:

1. O alvo do modelo corresponde ao endereco MAC de origem. Assim, o estudo resolve a identificacao de dispositivos observados no conjunto analisado, e nao a classificacao universal de qualquer dispositivo IoT desconhecido em ambiente aberto.
2. O conjunto e desbalanceado, com predominancia de alguns dispositivos. O uso de Macro F1 e multiplos protocolos de validacao reduz esse problema, mas nao o elimina.
3. A engenharia de atributos depende de janelas fixas de 100 pacotes. Outra granularidade pode alterar o equilibrio entre estabilidade estatistica e sensibilidade a mudancas de comportamento.
4. O estudo foi conduzido sobre uma base especifica. Portanto, a generalizacao para outros ambientes, dias de captura, topologias e populacoes de dispositivos deve ser tratada com cautela.
5. A validacao temporal melhora a credibilidade do experimento, mas ainda nao equivale a uma avaliacao com dispositivos totalmente novos, cenarios cruzados ou mudancas fortes de contexto operacional.

Essas limitacoes nao invalidam o trabalho, mas delimitam corretamente o escopo da contribuicao: o projeto demonstra que fingerprints baseadas em trafego Wi-Fi podem identificar de forma eficaz dispositivos presentes na base analisada, sob diferentes niveis de rigor experimental.

## 9. Conclusao

O projeto demonstrou que e possivel identificar dispositivos IoT por fingerprint de trafego Wi-Fi com desempenho elevado, desde que os dados sejam devidamente tratados e agregados. A comparacao entre algoritmos mostrou superioridade consistente do Random Forest, enquanto a selecao automatica de hiperparametros tornou a escolha do modelo final mais justificavel.

Do ponto de vista metodologico, o principal fortalecimento do trabalho esta no uso combinado de holdout temporal, validacao cruzada estratificada, validacao temporal expanding e validacao temporal sliding. Esse conjunto de protocolos permite defender o resultado final com mais rigor, pois mostra tanto o desempenho em um cenario favoravel quanto a degradacao esperada em contextos temporais mais restritivos.

Em sintese, a abordagem adotada se mostrou tecnicamente consistente, reproduzivel e adequada para uma entrega academica centrada em tratamento, analise e identificacao de dispositivos IoT a partir de trafego Wi-Fi. A principal contribuicao do trabalho nao e afirmar uma solucao universal para qualquer cenario, mas apresentar uma pipeline completa, comparativa e metodologicamente defensavel para o problema investigado.