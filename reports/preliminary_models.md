# Preliminary forecasting model comparison

Development-only comparison on Panama 2018, 2019. Each year is predicted using training targets strictly before that year. The final test period is excluded before feature generation. Source quality auditing and hashing cover the original CSV but its test observations do not enter modelling or selection.

All four models use the same 13 base input features, feature availability rules, common observation mask and target timestamps. One StandardScaler is fitted on each training fold and its identical outputs are supplied to every model, including tree models. The shared mask follows the main feature builder, including required correction-input availability; those extra inputs are not supplied to these forecasting models.

Each family has one fixed parameter setting from configs/experiment.json, the only configuration file. There is no hyperparameter search, feature search or correction/XAI experiment here.

| model | n | MAE_2018 | MAE_2019 | MAE | RMSE | bias |
| --- | --- | --- | --- | --- | --- | --- |
| HistGBR | 17520 | 13.8817 | 19.3317 | 16.6067 | 25.6586 | -1.4890 |
| RandomForest | 17520 | 15.0092 | 22.2582 | 18.6337 | 29.6843 | -1.7600 |
| GBR | 17520 | 17.9250 | 24.5936 | 21.2593 | 30.8325 | -2.4997 |
| Ridge | 17520 | 29.2048 | 30.8666 | 30.0357 | 39.8541 | 3.2776 |

Development MAE favors **HistGBR** among these fixed candidates. Overall metrics pool individual hourly errors across development years. Bias is prediction minus actual load; all errors use the source load units.

This preliminary result describes the tested settings and is not an unbiased test estimate or a universal ranking of model families. It does not automatically replace the main GBR or change any residual-correction result. A main-model change would require a separately fixed experiment; the existing thesis comparisons retain their original backbone.

The HistGBR here predicts load from base features directly. It is distinct from the main experiment's direct baseline, which also receives base predictions, extra load features and known errors.

Reproduce: `.\.venv\Scripts\python.exe -B -m src.compare_forecasting_models`. Parameters are in [experiment.json](../configs/experiment.json). Fold boundaries, preprocessing statistics, configuration/source hashes and package versions are in [preliminary_models.json](preliminary_models.json); [CSV](preliminary_models.csv).
