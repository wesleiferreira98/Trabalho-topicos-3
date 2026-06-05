# Relatorio Consolidado - SVM

## Resumo Executivo
- Linhas tratadas: 640889
- Dispositivos de origem: 15
- Janelas usadas no modelo: 6399
- Acuracia media da CV estratificada: 0.9612
- Macro F1 medio da CV estratificada: 0.9525
- Acuracia media da CV temporal: 0.9502
- Macro F1 medio da CV temporal: 0.9174
- Acuracia media da CV temporal sliding: 0.9152
- Macro F1 medio da CV temporal sliding: 0.8638

## EDA
```text
Linhas tratadas: 640889
Dispositivos de origem: 15
Dispositivos com janelas suficientes: 11
Janelas agregadas: 6399
Frame len medio: 72.797
IAT medio: 0.015586
Top 10 MACs de origem:
- 4c:a9:19:e9:7e:af: 308228
- e0:01:c7:56:19:0c: 113411
- 44:17:93:14:08:37: 52340
- d8:c8:0c:1a:72:05: 33615
- fe:51:35:71:8e:66: 31412
- 38:a5:c9:ac:2e:89: 30963
- fc:3c:d7:c7:37:8e: 22476
- 04:7e:4a:a1:b1:38: 14927
- d8:c8:0c:fb:c6:e3: 12061
- bc:bd:84:f0:4a:33: 11680
```

## Treino e Validacao
### Holdout temporal por dispositivo
```text
precision    recall  f1-score   support

04:7e:4a:a1:b1:38       0.97      0.97      0.97        30
38:a5:c9:ac:2e:89       0.97      0.94      0.95        62
44:17:93:14:08:37       0.93      0.93      0.93       105
4c:a9:19:e9:7e:af       0.99      0.98      0.98       617
ac:41:6a:66:e6:f3       0.94      0.88      0.91        17
bc:bd:84:f0:4a:33       0.64      0.88      0.74        24
d8:c8:0c:1a:72:05       1.00      0.97      0.99        68
d8:c8:0c:fb:c6:e3       1.00      0.92      0.96        25
e0:01:c7:56:19:0c       0.99      0.99      0.99       227
fc:3c:d7:c7:37:8e       0.88      0.93      0.90        45
fe:51:35:71:8e:66       0.94      0.97      0.95        63

         accuracy                           0.97      1283
        macro avg       0.93      0.94      0.93      1283
     weighted avg       0.97      0.97      0.97      1283
```

### Validacao cruzada estratificada
| fold | train_windows | test_windows | accuracy | macro_f1 | weighted_f1 |
| --- | --- | --- | --- | --- | --- |
| 1 | 5119 | 1280 | 0.9648 | 0.9575 | 0.9656 |
| 2 | 5119 | 1280 | 0.9578 | 0.9554 | 0.9590 |
| 3 | 5119 | 1280 | 0.9586 | 0.9535 | 0.9598 |
| 4 | 5119 | 1280 | 0.9641 | 0.9441 | 0.9647 |
| 5 | 5120 | 1279 | 0.9609 | 0.9519 | 0.9617 |

### Validacao cruzada temporal
| fold | train_windows | test_windows | accuracy | macro_f1 | weighted_f1 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1062 | 1068 | 0.9213 | 0.8808 | 0.9222 |
| 2 | 2130 | 1066 | 0.9615 | 0.9588 | 0.9625 |
| 3 | 3196 | 1066 | 0.9522 | 0.9085 | 0.9541 |
| 4 | 4262 | 1065 | 0.9512 | 0.9097 | 0.9491 |
| 5 | 5327 | 1072 | 0.9646 | 0.9292 | 0.9655 |

### Validacao cruzada temporal sliding
| fold | train_windows | test_windows | accuracy | macro_f1 | weighted_f1 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1062 | 1068 | 0.9213 | 0.8808 | 0.9222 |
| 2 | 1068 | 1066 | 0.9371 | 0.9379 | 0.9401 |
| 3 | 1066 | 1066 | 0.9259 | 0.8629 | 0.9282 |
| 4 | 1066 | 1065 | 0.9211 | 0.8522 | 0.9194 |
| 5 | 1065 | 1072 | 0.8703 | 0.7854 | 0.8750 |

### Hiperparametros selecionados automaticamente
- Melhor macro F1 medio na busca: 0.9550
- model__kernel: rbf
- model__gamma: scale
- model__C: 10.0

## Artefatos do Modelo
- reports/figures/svm_confusion_matrix.png
- reports/figures/svm_cv_metrics_by_fold.png
- reports/figures/svm_temporal_cv_metrics_by_fold.png
- reports/figures/svm_sliding_temporal_cv_metrics_by_fold.png
- reports/svm_cv_results.csv
- reports/svm_temporal_cv_results.csv
- reports/svm_sliding_temporal_cv_results.csv
- models/svm_device_fingerprint.joblib
- reports/svm_best_params.json
