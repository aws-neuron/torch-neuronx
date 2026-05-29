# Athena DDL for the upstream pytorch dashboard

These `.sql` files are the **canonical Athena table definitions** for the parquet output of `../write_parquet.py`. They are documentation, not deployment — nothing in CI applies them.

## Files

| File | Table |
|---|---|
| `run_metrics.sql` | One row per `(run, component)`, including the `component='overall'` roll-up |
| `test_failures.sql` | One row per `failed`/`error` test |

## Why these live in the repo

1. **Drift protection.** The pyarrow schema in `../write_parquet.py` and the Athena DDL must match column-for-column. Keeping the DDL next to the writer means PR reviewers see schema changes on both sides.
2. **Reproducible setup.** Anyone provisioning a new Athena workgroup (test env, recovery, new region) can paste the SQL directly.

## Running the DDL (one-time setup)

```bash
# 1. Create the database (one-time per workgroup)
aws athena start-query-execution \
  --query-string "CREATE DATABASE IF NOT EXISTS upstream_pytorch LOCATION 's3://<bucket>/<prefix>/upstream_pytorch/';" \
  --work-group <workgroup>

# 2. Create each table (drop-and-recreate after a schema change)
aws athena start-query-execution \
  --query-string "$(cat run_metrics.sql)" \
  --work-group <workgroup>

aws athena start-query-execution \
  --query-string "$(cat test_failures.sql)" \
  --work-group <workgroup>
```

Or paste each file's contents into the Athena console and click Run.

## Schema-change checklist

When you change `write_parquet.py` in a way that affects the on-disk schema:

- [ ] Update the corresponding `.sql` file in this directory in the **same PR**
- [ ] Bump the file's header comment if the change is breaking
- [ ] After merge, manually re-run the DDL on the prod Athena workgroup (`DROP TABLE` + `CREATE EXTERNAL TABLE`)
- [ ] If renaming a column, you'll need a follow-up CTAS / reload of historical data; Athena does not support `RENAME COLUMN` on external tables

## Placeholders

The `.sql` files contain `<bucket>` and `<prefix>` placeholders that the operator fills in once. The branch enum (`projection.branch.values`) and date range (`projection.dt.range`) may also need widening as new branches and runs accumulate.
