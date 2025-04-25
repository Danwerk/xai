from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

def evaluate_model(y_true, y_pred, time_period, name="Model Name", show_plot=False):
    """Evaluate model by calculating different metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred)

    print(f"{name}")
    print(f"MAE  = {mae:.2f}")
    print(f"MSE  = {mse:.2f}")
    print(f"RMSE = {rmse:.2f}")
    print(f"MAPE = {mape:.2f}")
    print(f"R2   = {r2:.2f}")

    if show_plot:
        plt.figure(figsize=(15, 5))
        if time_period is not None:
            plt.plot(time_period, y_true, label='Actual', color='black')
            plt.plot(time_period, y_pred, label='Forecast', linestyle='--', color='orange')
        else:
            plt.plot(y_true, label='Actual', color='black')
            plt.plot(y_pred, label='Forecast', linestyle='--', color='orange')

        plt.title(name)
        plt.xlabel('Aeg')
        plt.ylabel('Koormus (MW)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    return {'Model': name, 'MAE': round(mae, 2), 'MAPE': round(mape, 2), 'RMSE': round(rmse, 2), 'R2': round(r2, 2)}
    #return {'MAE': mae, 'MAPE': mape, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

def evaluate_all_cases(y_test, y_pred, time_index, model_name="Model"):
    """Testandmestiku hindamine 3s erinevas etapis"""
    start = "2019-01-17 14:00"
    end = "2019-01-23 20:00"
    mask = (time_index >= start) & (time_index <= end)

    res_1 = evaluate_model(
        y_true=y_test.loc[mask],
        y_pred=y_pred[mask],
        time_period=time_index.loc[mask],
        name=f"{model_name} (17.01–23.01)"
    )


    res_2 = evaluate_model(
        y_true=y_test.iloc[:200],
        y_pred=y_pred[:200],
        time_period=time_index.iloc[:200],
        name=f"{model_name} (first 200 hours)"
    )

    res_3 = evaluate_model(
        y_true=y_test,
        y_pred=y_pred,
        time_period=time_index,
        name=f"{model_name} (all dataset)"
    )

    return res_1, res_2, res_3