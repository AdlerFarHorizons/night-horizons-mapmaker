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
        container.register_service('param_getter', ParamReturner)

        self.assertEqual(container.get_service('param_getter').get(5), 5)

    def test_get_generic_service(self):
        '''Can we register a service where the actual service used is
        determined by the config file?
        '''

        container = DIContainer(
            config_filepath='./test/config.yaml',
            local_options={
                'param_getter': {
                    'version': 'param_getter_double',
                }
            }
        )

        class ParamReturner:
            def get(self, x):
                return x
        container.register_service('param_getter_regular', ParamReturner)

        class ParamReturner2:
            def get(self, x):
                return 2 * x
        container.register_service('param_getter_double', ParamReturner2)

        container.register_service('param_getter', None)

        self.assertEqual(container.get_service('param_getter').get(5), 10)
