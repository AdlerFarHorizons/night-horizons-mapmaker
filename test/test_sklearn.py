import pytest
from sklearn.utils.estimator_checks import check_estimator

from night_horizons import preprocess


@pytest.mark.parametrize(
    "estimator",
    [preprocess.NITELitePreprocesser(),]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
