import os
import unittest
import psycopg2

from night_horizons.mapmake import create_mapmaker
import pandas as pd
from sqlalchemy import create_engine


class TestDatabaseConnection(unittest.TestCase):

    def test_sql_connection(self):
        url = os.getenv('DATABASE_URL')
        try:
            conn = psycopg2.connect(url)
            self.assertTrue(conn)
        except psycopg2.OperationalError:
            self.fail("Failed to establish SQL connection")


class TestMetadataProcessing(unittest.TestCase):

    def test_metadata_processing(self):

        local_options = {'mapmaker': {'map_type': 'metadata_processor'}}
        metadata_processor = create_mapmaker(
            './test/config.yml',
            local_options=local_options
        )
        metadata_processor.run()

        # Create a connection to the Postgres instance
        url = os.getenv('DATABASE_URL')
        engine = create_engine(url)

        # Retrieve the table 'FH135' with pandas
        df = pd.read_sql_table('FH135', con=engine)

        # Assert that the dataframe is not empty
        self.assertFalse(df.empty)
