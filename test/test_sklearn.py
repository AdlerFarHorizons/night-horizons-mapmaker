'''TODO: Delete'''
import pytest
from sklearn.utils.estimator_checks import check_estimator

from night_horizons.transformers import preprocessors


#   because the NITELite Preprocessor can't take in random arrays.
# @pytest.mark.parametrize(
#     "estimator",
#     [preprocess.NITELitePreprocessor(),]
# )
# def test_all_estimators(estimator):
#     return check_estimator(estimator)
