import pytest
from sklearn.utils.estimator_checks import check_estimator

from night_horizons import preprocessers


# TODO: Get this up-and-running. Currently doesn't work
#   because the NITELite Preprocesser can't take in random arrays.
# @pytest.mark.parametrize(
#     "estimator",
#     [preprocess.NITELitePreprocesser(),]
# )
# def test_all_estimators(estimator):
#     return check_estimator(estimator)
