import pytest
from sklearn.utils.estimator_checks import check_estimator

from night_horizons_mapmaker import reference

@pytest.mark.parametrize(
    "estimator",
    [reference.Georeferencer(),]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
