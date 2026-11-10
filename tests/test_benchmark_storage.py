"""Safety and workload checks for the local storage latency probe."""

import itertools
import unittest
from unittest import mock

from tools import benchmark_storage as benchmark


class BenchmarkTests(unittest.TestCase):
    def test_measurement_excludes_warmup_and_runs_finish_outside_timed_calls(self):
        session, journal, memory = mock.Mock(), mock.Mock(spec=benchmark.postgres.PostgresJournal), mock.Mock()
        memory.search.return_value = [object()] * 5
        journal.begin.return_value = "action"
        with mock.patch.multiple(benchmark, WARMUP=1, SAMPLES=2, RECORDS=10), \
                mock.patch.object(benchmark.time, "perf_counter", side_effect=(i / 1000 for i in itertools.count())):
            results = benchmark.measure(session, journal, memory)
        self.assertEqual(results, {name: {"median_ms": 1.0, "p95_ms": 1.0}
                                   for name in ("save", "begin", "search")})
        self.assertEqual(session.save.call_count, 4)  # Initial save, warm-up, two samples.
        self.assertEqual(memory.add.call_count, 10)
        self.assertEqual(journal.begin.call_count, 3)
        self.assertEqual(journal.finish.call_count, 3)
        journal.close_run.assert_called_once_with("run")

    def test_existing_mongo_database_is_not_modified_or_dropped(self):
        client = mock.MagicMock()
        client.list_database_names.return_value = ["lh_bench_existing"]
        with mock.patch.dict(benchmark.os.environ, {benchmark.MONGO_ENV: "fixture"}), \
                mock.patch.object(benchmark.mongodb, "MongoClient", return_value=client, create=True):
            with self.assertRaises(RuntimeError):
                benchmark.mongo_probe("lh_bench_existing")
        client.drop_database.assert_not_called()
        client.__getitem__.assert_not_called()
        client.close.assert_called_once()

    def test_failed_postgres_schema_creation_is_not_dropped(self):
        driver, sql = mock.MagicMock(), mock.MagicMock()
        connection = driver.connect.return_value.__enter__.return_value
        connection.execute.side_effect = RuntimeError("already exists")
        with mock.patch.dict(benchmark.os.environ, {benchmark.POSTGRES_ENV: "fixture"}), \
                mock.patch.object(benchmark.postgres, "psycopg", driver), \
                mock.patch.object(benchmark.postgres, "sql", sql, create=True):
            with self.assertRaises(RuntimeError):
                benchmark.postgres_probe("lh_bench_existing")
        self.assertEqual(connection.execute.call_count, 1)
        driver.connect.assert_called_once()


if __name__ == "__main__":
    unittest.main()