import os
import unittest
import psycopg2

from night_horizons.mapmake import create_mapmaker
import pandas as pd
from sqlalchemy import create_engine, sql


class TestDatabaseConnection(unittest.TestCase):
    '''We do not connect to a SQL database after all, so this test is not
    relevant
    '''

#     def test_sql_connection(self):
#         url = os.getenv('DATABASE_URL')
#         try:
#             conn = psycopg2.connect(url)
#             self.assertTrue(conn)
#         except psycopg2.OperationalError:
#             self.fail("Failed to establish SQL connection")


class TestMetadataProcessing(unittest.TestCase):

#     '''We do not connect to a SQL database after all, so this test is not
#     relevant. Currently this test does not fail, because table FH135 was
#     created but not properly deleted.
#     '''
#     def test_metadata_processing_sql_output(self):
# 
#         local_options = {'mapmaker': {'map_type': 'metadata_processor'}}
#         metadata_processor = create_mapmaker(
#             './test/config.yml',
#             local_options=local_options
#         )
#         # metadata_processor.run()
# 
#         io_manager = metadata_processor.container.get_service('io_manager')
#         engine = io_manager.get_connection()
#         with engine.connect() as conn:
#             conn.execute(sql.text("DROP TABLE IF EXISTS FH135"))
# 
#         # Create a connection to the Postgres instance
#         url = os.getenv('DATABASE_URL')
#         engine = create_engine(url)
# 
#         # Retrieve the table 'FH135' with pandas
#         df = pd.read_sql_table('FH135', con=engine)
# 
#         # Assert that the dataframe is not empty
#         self.assertFalse(df.empty)
