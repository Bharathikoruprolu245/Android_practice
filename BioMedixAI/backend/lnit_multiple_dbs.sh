#!/bin/bash
# Creates the separate test database alongside the main POSTGRES_DB
# (biomedix_db) that the postgres image creates automatically on first boot.
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    CREATE DATABASE biomedix_test_db;
EOSQL