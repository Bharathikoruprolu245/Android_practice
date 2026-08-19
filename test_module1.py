"""
Quick standalone test of Module 1 — run this directly, no Streamlit needed.

Requires a running PostgreSQL instance. Either:
  - set DATABASE_URL, e.g.:
      export DATABASE_URL="postgresql://biomedix_user:password@localhost:5432/biomedix_db"
  - or rely on the default in db.py (same connection string) if you set up
    Postgres via the provided docker-compose.yml with default credentials.
"""

from data_ingestor import DataIngestor, MockGeneProvider
from network_analyzer import NetworkAnalyzer
from target_discovery import TargetDiscoveryPipeline
from db import Database

# Wire up dependencies (this is what PipelineRunner will do later, at a bigger scale)
provider = MockGeneProvider()
ingestor = DataIngestor(gene_provider=provider)
analyzer = NetworkAnalyzer()
db = Database()  # reads DATABASE_URL env var; creates tables on first run
pipeline = TargetDiscoveryPipeline(ingestor=ingestor, analyzer=analyzer, db=db)

print("=" * 60)
print("TEST 1: Rice cadmium toxicity (your CRISPR project's disease)")
print("=" * 60)
# NOTE: species=4530 is Oryza sativa (rice) in STRING DB, not human (9606)
result = pipeline.run("rice cadmium toxicity", species="rice")
print()
print(result)

print()
print("=" * 60)
print("TEST 2: Alzheimer's disease (human, sanity check on a well-connected PPI network)")
print("=" * 60)
result2 = pipeline.run("alzheimer's disease", species="human")
print()
print(result2)

print()
print("=" * 60)
print("DATABASE CONTENTS (verifying persistence worked)")
print("=" * 60)
for table in ["DISEASE", "GENE", "DISEASE_GENE_ASSOC", "PPI_INTERACTION", "NETWORK_ANALYSIS_RUN"]:
    rows = db.fetch_all(table)
    print(f"\n{table} ({len(rows)} rows):")
    for row in rows[:5]:  # print first 5 rows only, keep it readable
        print(f"  {row}")
    if len(rows) > 5:
        print(f"  ... and {len(rows) - 5} more")

db.close()