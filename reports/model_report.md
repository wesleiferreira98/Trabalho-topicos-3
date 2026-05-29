# Relatorio Consolidado - IoT Fingerprint

## Resumo Executivo
- Linhas tratadas: 640889
- Dispositivos de origem: 15
- Janelas usadas no modelo: 6399
- Acuracia media da CV temporal: 0.9734
- Macro F1 medio da CV temporal: 0.9458

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

Figuras geradas:
- reports/figures/top_sources.png
- reports/figures/power_management_distribution.png
- reports/figures/frame_length_by_device.png
- reports/figures/window_feature_scatter.png
- reports/figures/feature_correlation_heatmap.png

## Treino e Validacao
### Holdout temporal por dispositivo
```text
precision    recall  f1-score   support

04:7e:4a:a1:b1:38       0.94      0.97      0.95        30
38:a5:c9:ac:2e:89       1.00      0.97      0.98        62
44:17:93:14:08:37       0.97      0.88      0.92       105
4c:a9:19:e9:7e:af       0.98      1.00      0.99       617
ac:41:6a:66:e6:f3       0.93      0.76      0.84        17
bc:bd:84:f0:4a:33       1.00      0.92      0.96        24
d8:c8:0c:1a:72:05       0.99      1.00      0.99        68
d8:c8:0c:fb:c6:e3       1.00      1.00      1.00        25
e0:01:c7:56:19:0c       0.98      0.99      0.98       227
fc:3c:d7:c7:37:8e       0.96      0.98      0.97        45
fe:51:35:71:8e:66       0.97      0.95      0.96        63

         accuracy                           0.98      1283
        macro avg       0.97      0.95      0.96      1283
     weighted avg       0.98      0.98      0.98      1283
```

### Validacao cruzada temporal por blocos
| fold | train_windows | test_windows | accuracy | macro_f1 | weighted_f1 |
| --- | --- | --- | --- | --- | --- |
| 1 | 1283 | 1279 | 0.9765 | 0.9715 | 0.9762 |
| 2 | 2562 | 1281 | 0.9813 | 0.9627 | 0.9814 |
| 3 | 3843 | 1279 | 0.9593 | 0.8903 | 0.9561 |
| 4 | 5122 | 1277 | 0.9765 | 0.9587 | 0.9760 |

### Features mais importantes
- frame_len_max: 0.1688
- pwrmgt_ratio: 0.1390
- frame_len_mean: 0.1291
- unique_destinations: 0.1204
- iat_mean: 0.1080
- iat_std: 0.0935
- iat_min: 0.0805
- frame_len_std: 0.0755
- iat_max: 0.0495
- frame_len_min: 0.0356
- packet_count: 0.0000

## Artefatos do Modelo
- reports/figures/rf_confusion_matrix.png
- reports/figures/rf_feature_importance.png
- reports/figures/rf_cv_metrics_by_fold.png
- reports/cv_results.csv
- models/random_forest_device_fingerprint.joblib
