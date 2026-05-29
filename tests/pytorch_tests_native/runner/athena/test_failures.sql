-- Athena DDL for the upstream pytorch test_failures table.
--
-- This file is the canonical source for the table definition. If you
-- change the parquet schema in ../write_parquet.py, update this file in
-- the same PR and re-run the DDL on the Athena workgroup. Nothing in CI
-- applies this automatically.
--
-- Run order:
--   1. DROP TABLE IF EXISTS upstream_pytorch.test_failures;
--   2. The CREATE EXTERNAL TABLE statement below.
--
-- Replace `<bucket>` and `<prefix>` with the dashboard-owned S3 prefix
-- before running. Keep `'projection.branch.values'` and
-- `'projection.dt.range'` consistent with run_metrics.sql so both tables
-- accept the same partitions.

CREATE EXTERNAL TABLE upstream_pytorch.test_failures (
    commit_sha       string,
    run_id           string,
    run_timestamp    timestamp,
    component        string,
    test_id          string,
    status           string,
    failure_reason   string,
    duration_sec     double
)
PARTITIONED BY (
    branch  string,
    dt      date
)
STORED AS PARQUET
LOCATION 's3://<bucket>/<prefix>/upstream_pytorch/test_failures/'
TBLPROPERTIES (
    'parquet.compression' = 'SNAPPY',

    'projection.enabled' = 'true',

    'projection.branch.type' = 'enum',
    'projection.branch.values' = 'main,nightly,release/2.11',

    'projection.dt.type' = 'date',
    'projection.dt.range' = '2026-01-01,NOW',
    'projection.dt.format' = 'yyyy-MM-dd',
    'projection.dt.interval' = '1',
    'projection.dt.interval.unit' = 'DAYS',

    'storage.location.template' =
        's3://<bucket>/<prefix>/upstream_pytorch/test_failures/branch=${branch}/dt=${dt}/'
);
