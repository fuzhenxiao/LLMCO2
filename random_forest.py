import numpy as np
import joblib
from util import Metric

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


if __name__ == "__main__":
    data_all = joblib.load("data_all.joblib")
    # [X_train, X_val, X_test, y_train, y_val, y_test]

    rf = RandomForestRegressor(n_estimators=50, random_state=5)
    label = []
    for i in range(data_all[3].shape[0]):
        temp = data_all[3][i][0] + data_all[3][i][1]
        label.append(temp)
    label_array = np.array(label, dtype=np.float32)

    rf.fit(data_all[0], label_array)

    metric1 = Metric()
    predictions = rf.predict(data_all[2])
    ps = predictions.tolist()

    label = []
    for i in range(data_all[5].shape[0]):
        temp = data_all[5][i][0] + data_all[5][i][1]
        label.append(temp)
    label_array = np.array(label, dtype=np.float32)

    gs = label_array.ravel().tolist()
    metric1.update(ps, gs)
    acc, err, err1, err2, cnt = metric1.get()
    print(
        "MAPE:{:.5f} ErrBnd(0.3):{:.5f} ErrBnd(0.1):{:.5f} ErrBnd(0.05):{:.5f}".format(
            acc, err, err1, err2
        )
    )
