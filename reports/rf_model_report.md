# Relatorio Consolidado - Random Forest

## Resumo Executivo
- Linhas tratadas: 640889
- Dispositivos de origem: 15
- Janelas usadas no modelo: 6399
- Acuracia media da CV estratificada: 0.9823
- Macro F1 medio da CV estratificada: 0.9764
- Acuracia media da CV temporal: 0.9743
- Macro F1 medio da CV temporal: 0.9548
- Acuracia media da CV temporal sliding: 0.9599
- Macro F1 medio da CV temporal sliding: 0.9147

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

04:7e:4a:a1:b1:38       0.94      0.97      0.95        30
38:a5:c9:ac:2e:89       1.00      0.95      0.98        62
44:17:93:14:08:37       0.95      0.88      0.91       105
4c:a9:19:e9:7e:af       0.98      0.99      0.99       617
ac:41:6a:66:e6:f3       0.87      0.76      0.81        17
bc:bd:84:f0:4a:33       1.00      0.92      0.96        24
d8:c8:0c:1a:72:05       0.99      1.00      0.99        68
d8:c8:0c:fb:c6:e3       1.00      1.00      1.00        25
e0:01:c7:56:19:0c       0.98      0.99      0.98       227
fc:3c:d7:c7:37:8e       0.94      0.98      0.96        45
fe:51:35:71:8e:66       0.95      0.95      0.95        63

         accuracy                           0.97      1283
        macro avg       0.96      0.94      0.95      1283
     weighted avg       0.97      0.97      0.97      1283
```

### Validacao cruzada estratificada
| fold | train_windows | test_windows | accuracy | macro_f1 | weighted_f1 |
| --- | --- | --- | --- | --- | --- |
| 1 | 5119 | 1280 | 0.9875 | 0.9818 | 0.9876 |
| 2 | 5119 | 1280 | 0.9828 | 0.9810 | 0.9827 |
| 3 | 5119 | 1280 | 0.9773 | 0.9751 | 0.9777 |
| 4 | 5119 | 1280 | 0.9797 | 0.9625 | 0.9795 |
| 5 | 5120 | 1279 | 0.9844 | 0.9814 | 0.9844 |

### Validacao cruzada temporal
| fold | train_windows | test_windows | accuracy | macro_f1 | weighted_f1 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1062 | 1068 | 0.9625 | 0.9295 | 0.9611 |
| 2 | 2130 | 1066 | 0.9822 | 0.9793 | 0.9821 |
| 3 | 3196 | 1066 | 0.9747 | 0.9476 | 0.9748 |
| 4 | 4262 | 1065 | 0.9765 | 0.9655 | 0.9766 |
| 5 | 5327 | 1072 | 0.9757 | 0.9522 | 0.9752 |

### Validacao cruzada temporal sliding
| fold | train_windows | test_windows | accuracy | macro_f1 | weighted_f1 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1062 | 1068 | 0.9625 | 0.9295 | 0.9611 |
| 2 | 1068 | 1066 | 0.9737 | 0.9709 | 0.9735 |
| 3 | 1066 | 1066 | 0.9700 | 0.9457 | 0.9697 |
| 4 | 1066 | 1065 | 0.9408 | 0.8415 | 0.9314 |
| 5 | 1065 | 1072 | 0.9524 | 0.8858 | 0.9494 |

### Hiperparametros selecionados automaticamente
- Melhor macro F1 medio na busca: 0.9784
- n_estimators: 200
- min_samples_split: 2
- min_samples_leaf: 1
- max_features: log2
- max_depth: 10

### Comparacao entre algoritmos
| model | holdout_accuracy | holdout_macro_f1 | holdout_weighted_f1 | cv_accuracy_mean | cv_macro_f1_mean | cv_weighted_f1_mean | temporal_cv_accuracy_mean | temporal_cv_macro_f1_mean | temporal_cv_weighted_f1_mean | sliding_temporal_cv_accuracy_mean | sliding_temporal_cv_macro_f1_mean | sliding_temporal_cv_weighted_f1_mean | tuning_macro_f1 | best_params |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Random Forest | 0.9743 | 0.9527 | 0.9739 | 0.9823 | 0.9764 | 0.9824 | 0.9743 | 0.9548 | 0.9740 | 0.9599 | 0.9147 | 0.9570 | 0.9784 | {"max_depth": 10, "max_features": "log2", "min_samples_leaf": 1, "min_samples_split": 2, "n_estimators": 200} |
| SVM | 0.9665 | 0.9334 | 0.9674 | 0.9612 | 0.9525 | 0.9621 | 0.9502 | 0.9174 | 0.9507 | 0.9152 | 0.8638 | 0.9170 | 0.9550 | {"model__C": 10.0, "model__gamma": "scale", "model__kernel": "rbf"} |
| KNN | 0.9680 | 0.9444 | 0.9677 | 0.9687 | 0.9446 | 0.9685 | 0.9537 | 0.8927 | 0.9519 | 0.9393 | 0.8688 | 0.9375 | 0.9450 | {"model__n_neighbors": 7, "model__p": 1, "model__weights": "distance"} |

### Features mais importantes
- frame_len_max: 0.1623
- pwrmgt_ratio: 0.1476
- frame_len_mean: 0.1275
- unique_destinations: 0.1222
- iat_mean: 0.1147
- iat_std: 0.0884
- iat_min: 0.0831
- frame_len_std: 0.0726
- iat_max: 0.0442
- frame_len_min: 0.0373
- packet_count: 0.0000

## Artefatos do Modelo
- reports/figures/rf_confusion_matrix.png
- reports/figures/rf_cv_metrics_by_fold.png
- reports/figures/rf_temporal_cv_metrics_by_fold.png
- reports/figures/rf_sliding_temporal_cv_metrics_by_fold.png
- reports/rf_cv_results.csv
- reports/rf_temporal_cv_results.csv
- reports/rf_sliding_temporal_cv_results.csv
- models/rf_device_fingerprint.joblib
- reports/figures/rf_feature_importance.png
- reports/model_benchmark.csv
- reports/figures/model_comparison_cv.png
- reports/rf_best_params.json
