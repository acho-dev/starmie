#!/usr/bin/env bash
#
# export_schema_to_csv.sh
#
# Exports every table in a given PostgreSQL schema into separate CSV files
# in a specified directory. Uses psql’s \copy command (client‐side COPY).
#
# USAGE:
#   ./export_schema_to_csv.sh
#
# VARIABLES TO ADJUST:
#   DB_NAME       – name of your PostgreSQL database
#   SCHEMA        – schema to export (e.g. "public" or "my_schema")
#   OUTPUT_DIR    – directory where CSV files will be written
#   PGHOST, PGPASSWORD, PGUSER, PGPORT – (optional) connection settings
#
# NOTE:
#   You can also set connection parameters via environment or .pgpass
#   If you prefer prompting for a password, omit PGPASSWORD here.

####### BEGIN CONFIGURATION #######
DB_NAME="postgres"
SCHEMA="direct_3345_1baa9"
OUTPUT_DIR="./data/tcb/tables"

# (Optional) If you want to define host/port/user/password here, uncomment and fill these:
export PGHOST="34.123.74.5"
export PGPORT="5432"
export PGUSER="direct_3345_1baa9"
export PGPASSWORD="LUQwhuv8OUsp122P"
####### END CONFIGURATION #######

# 1. Create (or verify) the output directory
mkdir -p "${OUTPUT_DIR}"
if [[ ! -d "${OUTPUT_DIR}" ]]; then
  echo "ERROR: Could not create or access folder: ${OUTPUT_DIR}"
  exit 1
fi

# 2. Fetch list of tables in the given schema
TABLE_LIST=( $(psql \
  -d "${DB_NAME}" \
  -qAtc "SELECT tablename
         FROM pg_tables
         WHERE schemaname = '${SCHEMA}'
         ORDER BY tablename;" ) )

# If no tables found, exit
if [[ ${#TABLE_LIST[@]} -eq 0 ]]; then
  echo "No tables found in schema '${SCHEMA}'. Exiting."
  exit 0
fi

TOTAL=${#TABLE_LIST[@]}
echo "Found ${TOTAL} tables in schema '${SCHEMA}'. Beginning export..."

# 3. Loop through each table with a counter
count=0
for tbl in "${TABLE_LIST[@]}"; do
  ((count++))
  OUTFILE="${OUTPUT_DIR}/${tbl}.csv"
  echo "[${count}/${TOTAL}] Exporting ${SCHEMA}.${tbl} → ${OUTFILE}"
  
  psql \
    -d "${DB_NAME}" \
    -c "\copy ${SCHEMA}.\"${tbl}\" TO '${OUTFILE}' WITH CSV HEADER" \
    &> /dev/null
  
  if [[ $? -ne 0 ]]; then
    echo "    ✗ Failed to export ${tbl}"
  else
    echo "    ✔ ${tbl}.csv"
  fi
done

echo "All done: ${TOTAL} tables exported to '${OUTPUT_DIR}'."