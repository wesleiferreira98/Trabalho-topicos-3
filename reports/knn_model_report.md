# Relatorio Consolidado - KNN

## Resumo Executivo
- Linhas tratadas: 640889
- Dispositivos de origem: 15
- Janelas usadas no modelo: 6399
- Acuracia media da CV estratificada: 0.9687
- Macro F1 medio da CV estratificada: 0.9446
- Acuracia media da CV temporal: 0.9537
- Macro F1 medio da CV temporal: 0.8927
- Acuracia media da CV temporal sliding: 0.9393
- Macro F1 medio da CV temporal sliding: 0.8688

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

04:7e:4a:a1:b1:38       0.97      1.00      0.98        30
38:a5:c9:ac:2e:89       0.92      0.95      0.94        62
44:17:93:14:08:37       0.99      0.86      0.92       105
4c:a9:19:e9:7e:af       0.97      1.00      0.98       617
ac:41:6a:66:e6:f3       0.79      0.88      0.83        17
bc:bd:84:f0:4a:33       1.00      0.92      0.96        24
d8:c8:0c:1a:72:05       1.00      1.00      1.00        68
d8:c8:0c:fb:c6:e3       1.00      0.92      0.96        25
e0:01:c7:56:19:0c       0.99      0.99      0.99       227
fc:3c:d7:c7:37:8e       0.93      0.91      0.92        45
fe:51:35:71:8e:66       0.95      0.87      0.91        63

         accuracy                           0.97      1283
        macro avg       0.95      0.94      0.94      1283
     weighted avg       0.97      0.97      0.97      1283
```

### Validacao cruzada estratificada
| fold | train_windows | test_windows | accuracy | macro_f1 | weighted_f1 |
| --- | --- | --- | --- | --- | --- |
| 1 | 5119 | 1280 | 0.9656 | 0.9400 | 0.9656 |
| 2 | 5119 | 1280 | 0.9641 | 0.9446 | 0.9638 |
| 3 | 5119 | 1280 | 0.9695 | 0.9366 | 0.9693 |
| 4 | 5119 | 1280 | 0.9742 | 0.9481 | 0.9740 |
| 5 | 5120 | 1279 | 0.9703 | 0.9538 | 0.9699 |

### Validacao cruzada temporal
| fold | train_windows | test_windows | accuracy | macro_f1 | weighted_f1 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1062 | 1068 | 0.9391 | 0.8713 | 0.9368 |
| 2 | 2130 | 1066 | 0.9644 | 0.9397 | 0.9642 |
| 3 | 3196 | 1066 | 0.9606 | 0.9079 | 0.9604 |
| 4 | 4262 | 1065 | 0.9408 | 0.8066 | 0.9352 |
| 5 | 5327 | 1072 | 0.9636 | 0.9378 | 0.9629 |

### Validacao cruzada temporal sliding
| fold | train_windows | test_windows | accuracy | macro_f1 | weighted_f1 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1062 | 1068 | 0.9391 | 0.8713 | 0.9368 |
| 2 | 1068 | 1066 | 0.9568 | 0.9163 | 0.9568 |
| 3 | 1066 | 1066 | 0.9428 | 0.8626 | 0.9406 |
| 4 | 1066 | 1065 | 0.9371 | 0.8597 | 0.9332 |
| 5 | 1065 | 1072 | 0.9207 | 0.8340 | 0.9199 |

### Hiperparametros selecionados automaticamente
- Melhor macro F1 medio na busca: 0.9450
- model__weights: distance
- model__p: 1
- model__n_neighbors: 7

## Artefatos do Modelo
- reports/figures/knn_confusion_matrix.png
- reports/figures/knn_cv_metrics_by_fold.png
- reports/figures/knn_temporal_cv_metrics_by_fold.png
- reports/figures/knn_sliding_temporal_cv_metrics_by_fold.png
- reports/knn_cv_results.csv
- reports/knn_temporal_cv_results.csv
- reports/knn_sliding_temporal_cv_results.csv
- models/knn_device_fingerprint.joblib
- reports/knn_best_params.json
