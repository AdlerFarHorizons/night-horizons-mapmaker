import unittest

from night_horizons.container import DIContainer


class TestContainer(unittest.TestCase):

    def test_basic_get_service(self):
        '''Can we register and get a service?
        '''

        container = DIContainer(
            config_filepath='./test/config.yaml',
        )

        class ParamReturner:
            def get(self, x):
                return x
        container.register_service('return_param', ParamReturner)

        self.assertEqual(container.get_service('return_param').get(5), 5)

    def test_get_generic_service(self):
        '''Can we register a service where the actual service used is
        determined by the config file?
        '''

        container = DIContainer(
            config_filepath='./test/config.yaml',
            local_options={
                'return_param': {
                    'version': 'return_param_regular',
                }
            }
        )

        def return_param(x):
            return x
        self.container.register_service(
            'return_param_regular',
            return_param,
        )

        def return_param_doubled(x):
            return 2 * x
        self.container.register_service(
            'return_param_doubled',
            return_param_doubled,
        )

        self.assertEqual(self.container.get_service('return_param')(5), 5)
