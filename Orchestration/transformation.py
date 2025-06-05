from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def train_model(df, **kwargs):
    features = ['PULocationID', 'DOLocationID']
    target = 'duration'

    train_dicts = df[features].to_dict(orient='records')

    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    y_train = df[target].values

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred)
    rmse = mse ** 0.5


    return {
        'intercept': lr.intercept_,
        'rmse': rmse
    }


@test
def test_output(output) -> None:
    assert 'rmse' in output, 'Missing RMSE'
