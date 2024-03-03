import unittest
import psycopg2


class TestDatabaseConnection(unittest.TestCase):
    def test_sql_connection(self):
        try:
            conn = psycopg2.connect(
                host="database",
                port=5432,
                user="nitelite",
                password="nitelite",
                database="nitelite"
            )
            self.assertTrue(conn)
        except psycopg2.OperationalError:
            self.fail("Failed to establish SQL connection")
