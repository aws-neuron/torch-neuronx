-- Athena DDL for the upstream pytorch run_metrics table.
--
-- This file is the canonical source for the table definition. If you
-- change the parquet schema in ../write_parquet.py, update this file in
-- the same PR and re-run the DDL on the Athena workgroup. Nothing in CI
-- applies this automatically.
--
-- Run order for first-time setup or after a schema change:
--   1. (one-time) CREATE DATABASE IF NOT EXISTS upstream_pytorch
--                 LOCATION 's3://<bucket>/<prefix>/upstream_pytorch/';
--   2. DROP TABLE IF EXISTS upstream_pytorch.run_metrics;
--   3. The CREATE EXTERNAL TABLE statement below.
--
-- Replace `<bucket>` and `<prefix>` with the dashboard-owned S3 prefix
-- before running. Replace `'main,nightly,release/2.11'` with the
-- enumerated list of branches the dashboard tracks; widen as needed.

CREATE EXTERNAL TABLE upstream_pytorch.run_metrics (
    commit_sha          string,
    run_id              string,
    run_timestamp       timestamp,
    component           string,
    passed              bigint,
    passed_on_retry     bigint,
    failed              bigint,
    error               bigint,
    skipped             bigint,
    skipped_on_retry    bigint,
    xfailed             bigint,
    xfailed_on_retry    bigint,
    xpassed             bigint,
    xpassed_on_retry    bigint,
    subtests_passed     bigint,
    total               bigint,
    pass_rate_pct       double,
    duration_sec        double,
    flaky               bigint,
    failures            bigint
)
PARTITIONED BY (
    branch  string,
    dt      date
)
STORED AS PARQUET
LOCATION 's3://<bucket>/<prefix>/upstream_pytorch/run_metrics/'
TBLPROPERTIES (
    'parquet.compression' = 'SNAPPY',

    -- Partition projection — Athena infers partitions from the path
    -- pattern below at query time. No Glue Crawler, no MSCK REPAIR.
    'projection.enabled' = 'true',

    'projection.branch.type' = 'enum',
    'projection.branch.values' = 'main,nightly,release/2.11',

    'projection.dt.type' = 'date',
    'projection.dt.range' = '2026-01-01,NOW',
    'projection.dt.format' = 'yyyy-MM-dd',
    'projection.dt.interval' = '1',
    'projection.dt.interval.unit' = 'DAYS',

    'storage.location.template' =
        's3://<bucket>/<prefix>/upstream_pytorch/run_metrics/branch=${branch}/dt=${dt}/'
);
