import os
import unittest
import psycopg2


class TestDatabaseConnection(unittest.TestCase):
    def test_sql_connection(self):
        url = os.getenv('DATABASE_URL')
        try:
            conn = psycopg2.connect(url)
            self.assertTrue(conn)
        except psycopg2.OperationalError:
            self.fail("Failed to establish SQL connection")
