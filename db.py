"""
Persistence layer — PostgreSQL implementation of the ER diagram tables that
Module 1 is responsible for populating:

    DISEASE
    GENE
    DISEASE_GENE_ASSOC
    PPI_INTERACTION
    NETWORK_ANALYSIS_RUN

Migrated from SQLite to PostgreSQL per project mentor guidance (the data is
relational with real FK integrity requirements — see SRS Section 7 and SDD
Section 7.3 for the rationale). The public Database class interface is
UNCHANGED from the SQLite version, so target_discovery.py, test_module1.py,
and test_pipeline.py did not need their calling code rewritten — only the
connection string and a few SQL dialect details changed underneath.

Modules 2 and 3 will extend this same database with their own tables
(PROTEIN_STRUCTURE, DRUGGABILITY_RESULT, GRNA_SUBMISSION,
CRISPR_SAFETY_RESULT, OFF_TARGET_SITE, INTEGRATED_REPORT) later — this file
still only owns the Module 1 subset. The full schema (all 11 tables) is
already modeled as SQLAlchemy ORM classes in common/models.py for whoever
picks up Module 4's CacheManager; this file intentionally stays a thin,
dependency-light psycopg2 wrapper since Module 1's needs are simple upserts.

Connection is configured via the DATABASE_URL environment variable, e.g.:
    postgresql://biomedix_user:password@localhost:5432/biomedix_db
Falls back to a sensible local-dev default if DATABASE_URL is not set
(see docker-compose.yml for spinning up a matching local Postgres).
"""

import os
import psycopg2
import psycopg2.extras
from typing import Optional, List, Dict


DEFAULT_DATABASE_URL = "postgresql://biomedix_user:password@localhost:5432/biomedix_db"

# Table names are written in uppercase here purely for readability/consistency
# with the ER diagram in the SRS. PostgreSQL folds unquoted identifiers to
# lowercase automatically and consistently, so CREATE TABLE DISEASE and
# SELECT * FROM DISEASE both resolve to the same underlying "disease" table —
# no behavior change from the SQLite version.
SCHEMA = """
CREATE TABLE IF NOT EXISTS DISEASE (
    disease_id   SERIAL PRIMARY KEY,
    disease_name VARCHAR(255) NOT NULL UNIQUE,
    umls_cui     VARCHAR(20),
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS GENE (
    gene_id           SERIAL PRIMARY KEY,
    gene_symbol       VARCHAR(50) NOT NULL UNIQUE,
    ncbi_gene_id      VARCHAR(20),
    uniprot_accession VARCHAR(20)
);

CREATE TABLE IF NOT EXISTS DISEASE_GENE_ASSOC (
    assoc_id          SERIAL PRIMARY KEY,
    disease_id        INTEGER NOT NULL REFERENCES DISEASE(disease_id),
    gene_id           INTEGER NOT NULL REFERENCES GENE(gene_id),
    association_score FLOAT,
    source            VARCHAR(30),
    UNIQUE(disease_id, gene_id)
);

CREATE TABLE IF NOT EXISTS PPI_INTERACTION (
    interaction_id SERIAL PRIMARY KEY,
    gene_a_id      INTEGER NOT NULL REFERENCES GENE(gene_id),
    gene_b_id      INTEGER NOT NULL REFERENCES GENE(gene_id),
    combined_score INTEGER
);

CREATE TABLE IF NOT EXISTS NETWORK_ANALYSIS_RUN (
    run_id            SERIAL PRIMARY KEY,
    disease_id        INTEGER NOT NULL REFERENCES DISEASE(disease_id),
    hub_gene_id       INTEGER NOT NULL REFERENCES GENE(gene_id),
    centrality_method VARCHAR(20),
    centrality_score  FLOAT,
    run_timestamp     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Module 2 tables (Structural ML — PocketDetector / FeatureExtractor /
-- DruggabilityEngine). Owned by this same Database class per the SDD note
-- that Modules 2/3 extend this database rather than standing up their own.

CREATE TABLE IF NOT EXISTS PROTEIN_STRUCTURE (
    structure_id SERIAL PRIMARY KEY,
    gene_id      INTEGER NOT NULL REFERENCES GENE(gene_id),
    pdb_id       VARCHAR(10) NOT NULL,
    resolution   FLOAT,
    method       VARCHAR(50),
    fetched_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(gene_id, pdb_id)
);

CREATE TABLE IF NOT EXISTS DRUGGABILITY_RESULT (
    result_id          SERIAL PRIMARY KEY,
    gene_id             INTEGER NOT NULL REFERENCES GENE(gene_id),
    structure_id        INTEGER REFERENCES PROTEIN_STRUCTURE(structure_id),
    pocket_rank         INTEGER,
    druggability_score  FLOAT NOT NULL,
    pocket_volume       FLOAT,
    pocket_features     JSONB,
    used_fallback       BOOLEAN NOT NULL DEFAULT FALSE,
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Module 3 tables (Genomic DL — SequenceFetcher / OffTargetScanner /
-- CrisprSafetyEngine). Owned by this same Database class per the SDD note
-- that Modules 2/3 extend this database rather than standing up their own.

CREATE TABLE IF NOT EXISTS GRNA_SUBMISSION (
    submission_id  SERIAL PRIMARY KEY,
    gene_id        INTEGER NOT NULL REFERENCES GENE(gene_id),
    guide_sequence VARCHAR(40) NOT NULL,
    pam_pattern    VARCHAR(10) NOT NULL DEFAULT 'NGG',
    submitted_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS CRISPR_SAFETY_RESULT (
    result_id            SERIAL PRIMARY KEY,
    gene_id              INTEGER NOT NULL REFERENCES GENE(gene_id),
    submission_id        INTEGER REFERENCES GRNA_SUBMISSION(submission_id),
    safety_score         FLOAT NOT NULL,
    num_candidate_sites  INTEGER NOT NULL DEFAULT 0,
    num_flagged_sites    INTEGER NOT NULL DEFAULT 0,
    used_fallback        BOOLEAN NOT NULL DEFAULT FALSE,
    created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS OFF_TARGET_SITE (
    site_id        SERIAL PRIMARY KEY,
    result_id      INTEGER NOT NULL REFERENCES CRISPR_SAFETY_RESULT(result_id),
    position       INTEGER,
    site_sequence  VARCHAR(40) NOT NULL,
    mismatches     INTEGER NOT NULL,
    pam_ok         BOOLEAN NOT NULL,
    risk_score     FLOAT NOT NULL
);

-- Module 4 table (Orchestration & Dashboard — PipelineRunner). Ties one
-- NETWORK_ANALYSIS_RUN together with the DRUGGABILITY_RESULT and (optional,
-- since a gRNA is optional per UC-8) CRISPR_SAFETY_RESULT it was paired
-- with, plus the generated verdict text.

CREATE TABLE IF NOT EXISTS INTEGRATED_REPORT (
    report_id             SERIAL PRIMARY KEY,
    run_id                INTEGER NOT NULL REFERENCES NETWORK_ANALYSIS_RUN(run_id),
    druggability_result_id INTEGER REFERENCES DRUGGABILITY_RESULT(result_id),
    crispr_result_id       INTEGER REFERENCES CRISPR_SAFETY_RESULT(result_id),
    verdict_text          TEXT NOT NULL,
    generated_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# Tables owned by Module 1, in FK-safe order for TRUNCATE (children first).
# Used by reset() for test isolation — see the updated test fixtures.
_MODULE1_TABLES_CHILD_FIRST = [
    "NETWORK_ANALYSIS_RUN",
    "PPI_INTERACTION",
    "DISEASE_GENE_ASSOC",
    "GENE",
    "DISEASE",
]

# Tables owned by Module 2, in FK-safe order for TRUNCATE (children first).
# GENE is intentionally excluded here — it's shared with Module 1, and
# truncating it from a Module-2-only reset would cascade into Module 1's
# rows too. Use reset() (which already includes GENE) for a full wipe.
_MODULE2_TABLES_CHILD_FIRST = [
    "DRUGGABILITY_RESULT",
    "PROTEIN_STRUCTURE",
]

# Tables owned by Module 3, in FK-safe order for TRUNCATE (children first).
# Same GENE-exclusion rationale as _MODULE2_TABLES_CHILD_FIRST above.
_MODULE3_TABLES_CHILD_FIRST = [
    "OFF_TARGET_SITE",
    "CRISPR_SAFETY_RESULT",
    "GRNA_SUBMISSION",
]

# Table owned by Module 4. Must be truncated before Modules 1-3's tables
# it references (NETWORK_ANALYSIS_RUN, DRUGGABILITY_RESULT,
# CRISPR_SAFETY_RESULT), hence it's listed first in reset()'s TRUNCATE.
_MODULE4_TABLES_CHILD_FIRST = [
    "INTEGRATED_REPORT",
]


class Database:
    """Thin wrapper around psycopg2 — one instance per connection string."""

    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or os.environ.get(
            "DATABASE_URL", DEFAULT_DATABASE_URL
        )
        self.conn = psycopg2.connect(self.database_url)
        self.conn.autocommit = False
        with self.conn.cursor() as cur:
            cur.execute(SCHEMA)
        self.conn.commit()

    # -- DISEASE ------------------------------------------------------
    def upsert_disease(self, disease_name: str, umls_cui: Optional[str] = None) -> int:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO DISEASE (disease_name, umls_cui) VALUES (%s, %s)
                ON CONFLICT (disease_name) DO UPDATE
                    SET umls_cui = COALESCE(EXCLUDED.umls_cui, DISEASE.umls_cui)
                RETURNING disease_id
                """,
                (disease_name, umls_cui),
            )
            disease_id = cur.fetchone()[0]
        self.conn.commit()
        return disease_id

    # -- GENE -----------------------------------------------------------
    def upsert_gene(
        self,
        gene_symbol: str,
        ncbi_gene_id: Optional[str] = None,
        uniprot_accession: Optional[str] = None,
    ) -> int:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO GENE (gene_symbol, ncbi_gene_id, uniprot_accession)
                VALUES (%s, %s, %s)
                ON CONFLICT (gene_symbol) DO UPDATE SET
                    ncbi_gene_id = COALESCE(EXCLUDED.ncbi_gene_id, GENE.ncbi_gene_id),
                    uniprot_accession = COALESCE(EXCLUDED.uniprot_accession, GENE.uniprot_accession)
                RETURNING gene_id
                """,
                (gene_symbol, ncbi_gene_id, uniprot_accession),
            )
            gene_id = cur.fetchone()[0]
        self.conn.commit()
        return gene_id

    # -- DISEASE_GENE_ASSOC ----------------------------------------------
    def insert_disease_gene_assoc(
        self,
        disease_id: int,
        gene_id: int,
        association_score: Optional[float] = None,
        source: str = "mock",
    ) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO DISEASE_GENE_ASSOC
                    (disease_id, gene_id, association_score, source)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (disease_id, gene_id) DO NOTHING
                """,
                (disease_id, gene_id, association_score, source),
            )
        self.conn.commit()

    # -- PPI_INTERACTION --------------------------------------------------
    def insert_ppi_interaction(
        self, gene_a_id: int, gene_b_id: int, combined_score: int
    ) -> None:
        with self.conn.cursor() as cur:
            cur.execute(
                "INSERT INTO PPI_INTERACTION (gene_a_id, gene_b_id, combined_score) "
                "VALUES (%s, %s, %s)",
                (gene_a_id, gene_b_id, combined_score),
            )
        self.conn.commit()

    # -- NETWORK_ANALYSIS_RUN ----------------------------------------------
    def insert_network_analysis_run(
        self,
        disease_id: int,
        hub_gene_id: int,
        centrality_method: str,
        centrality_score: float,
    ) -> int:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO NETWORK_ANALYSIS_RUN
                    (disease_id, hub_gene_id, centrality_method, centrality_score)
                VALUES (%s, %s, %s, %s) RETURNING run_id
                """,
                (disease_id, hub_gene_id, centrality_method, centrality_score),
            )
            run_id = cur.fetchone()[0]
        self.conn.commit()
        return run_id

    # -- PROTEIN_STRUCTURE ------------------------------------------------
    def upsert_protein_structure(
        self,
        gene_id: int,
        pdb_id: str,
        resolution: Optional[float] = None,
        method: Optional[str] = None,
    ) -> int:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO PROTEIN_STRUCTURE (gene_id, pdb_id, resolution, method)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (gene_id, pdb_id) DO UPDATE SET
                    resolution = COALESCE(EXCLUDED.resolution, PROTEIN_STRUCTURE.resolution),
                    method = COALESCE(EXCLUDED.method, PROTEIN_STRUCTURE.method)
                RETURNING structure_id
                """,
                (gene_id, pdb_id, resolution, method),
            )
            structure_id = cur.fetchone()[0]
        self.conn.commit()
        return structure_id

    # -- DRUGGABILITY_RESULT ------------------------------------------------
    def insert_druggability_result(
        self,
        gene_id: int,
        druggability_score: float,
        structure_id: Optional[int] = None,
        pocket_rank: Optional[int] = None,
        pocket_volume: Optional[float] = None,
        pocket_features: Optional[Dict] = None,
        used_fallback: bool = False,
    ) -> int:
        """
        One row per DruggabilityEngine.predict() call. Not an upsert —
        unlike NETWORK_ANALYSIS_RUN, re-running a gene deliberately keeps
        every historical result rather than overwriting, since scores can
        shift as the model is retrained or a newer PDB structure is used.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO DRUGGABILITY_RESULT
                    (gene_id, structure_id, pocket_rank, druggability_score,
                     pocket_volume, pocket_features, used_fallback)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING result_id
                """,
                (
                    gene_id,
                    structure_id,
                    pocket_rank,
                    druggability_score,
                    pocket_volume,
                    psycopg2.extras.Json(pocket_features) if pocket_features else None,
                    used_fallback,
                ),
            )
            result_id = cur.fetchone()[0]
        self.conn.commit()
        return result_id

    # -- GRNA_SUBMISSION ---------------------------------------------------
    def insert_grna_submission(
        self, gene_id: int, guide_sequence: str, pam_pattern: str = "NGG"
    ) -> int:
        """
        One row per guide RNA a user submits for safety evaluation. Not an
        upsert — the same guide can legitimately be resubmitted (e.g. after
        the model is retrained), and GRNA_SUBMISSION is the audit trail of
        what was actually asked, same rationale as DRUGGABILITY_RESULT's
        "keep every historical result" choice.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO GRNA_SUBMISSION (gene_id, guide_sequence, pam_pattern)
                VALUES (%s, %s, %s) RETURNING submission_id
                """,
                (gene_id, guide_sequence, pam_pattern),
            )
            submission_id = cur.fetchone()[0]
        self.conn.commit()
        return submission_id

    # -- CRISPR_SAFETY_RESULT / OFF_TARGET_SITE ------------------------------
    def insert_crispr_safety_result(
        self,
        gene_id: int,
        safety_score: float,
        submission_id: Optional[int] = None,
        num_candidate_sites: int = 0,
        num_flagged_sites: int = 0,
        used_fallback: bool = False,
        off_target_sites: Optional[List[Dict]] = None,
    ) -> int:
        """
        One row per CrisprSafetyEngine evaluation, plus one OFF_TARGET_SITE
        row per flagged candidate site (off_target_sites: list of dicts with
        keys position, site_sequence, mismatches, pam_ok, risk_score).
        Same "insert, don't upsert" rationale as insert_druggability_result.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO CRISPR_SAFETY_RESULT
                    (gene_id, submission_id, safety_score, num_candidate_sites,
                     num_flagged_sites, used_fallback)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING result_id
                """,
                (
                    gene_id,
                    submission_id,
                    safety_score,
                    num_candidate_sites,
                    num_flagged_sites,
                    used_fallback,
                ),
            )
            result_id = cur.fetchone()[0]

            for site in off_target_sites or []:
                cur.execute(
                    """
                    INSERT INTO OFF_TARGET_SITE
                        (result_id, position, site_sequence, mismatches, pam_ok, risk_score)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        result_id,
                        site.get("position"),
                        site["site_sequence"],
                        site["mismatches"],
                        site["pam_ok"],
                        site["risk_score"],
                    ),
                )
        self.conn.commit()
        return result_id

    # -- INTEGRATED_REPORT --------------------------------------------------
    def insert_integrated_report(
        self,
        run_id: int,
        verdict_text: str,
        druggability_result_id: Optional[int] = None,
        crispr_result_id: Optional[int] = None,
    ) -> int:
        """
        One row per PipelineRunner.run_full_pipeline() call. Not an upsert —
        same "keep every historical result" rationale as
        insert_druggability_result / insert_crispr_safety_result, since a
        disease can legitimately be re-run (new gRNA, retrained models,
        updated PPI data) and each full report is worth keeping for
        historical comparison (SDD Section 7.3 / SRS UC-11).
        """
        with self.conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO INTEGRATED_REPORT
                    (run_id, druggability_result_id, crispr_result_id, verdict_text)
                VALUES (%s, %s, %s, %s)
                RETURNING report_id
                """,
                (run_id, druggability_result_id, crispr_result_id, verdict_text),
            )
            report_id = cur.fetchone()[0]
        self.conn.commit()
        return report_id

    def get_integrated_report(self, report_id: int) -> Optional[Dict]:
        """Joins INTEGRATED_REPORT back to its run/disease/hub gene/scores —
        handy for CacheManager.get_cached_report()-style lookups and for the
        dashboard's history view."""
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT ir.report_id, ir.verdict_text, ir.generated_at,
                       nar.run_id, nar.centrality_method, nar.centrality_score,
                       d.disease_name, g.gene_symbol AS hub_gene_symbol,
                       dr.druggability_score, dr.used_fallback AS drug_used_fallback,
                       csr.safety_score, csr.used_fallback AS crispr_used_fallback
                FROM INTEGRATED_REPORT ir
                JOIN NETWORK_ANALYSIS_RUN nar ON nar.run_id = ir.run_id
                JOIN DISEASE d ON d.disease_id = nar.disease_id
                JOIN GENE g ON g.gene_id = nar.hub_gene_id
                LEFT JOIN DRUGGABILITY_RESULT dr ON dr.result_id = ir.druggability_result_id
                LEFT JOIN CRISPR_SAFETY_RESULT csr ON csr.result_id = ir.crispr_result_id
                WHERE ir.report_id = %s
                """,
                (report_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    # -- convenience read helpers (handy for debugging / the UI later) ----
    def fetch_all(self, table: str) -> List[Dict]:
        with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"SELECT * FROM {table}")
            return [dict(row) for row in cur.fetchall()]

    def list_recent_reports(self, limit: int = 10) -> List[Dict]:
        """
        Fetches the latest integrated reports to populate the dashboard history panel.
        Returns lightweight summary metadata including the gene symbol, verdict summary, and timestamp.
        """
        query = """
            SELECT ir.report_id, g.gene_symbol, ir.verdict_text, ir.generated_at
            FROM INTEGRATED_REPORT ir
            JOIN NETWORK_ANALYSIS_RUN nar ON ir.run_id = nar.run_id
            JOIN GENE g ON nar.hub_gene_id = g.gene_id
            ORDER BY ir.generated_at DESC
            LIMIT %s;
        """
        try:
            with self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(query, (limit,))
                rows = cur.fetchall()

                # Format fields cleanly for the Streamlit selectbox/history list
                formatted_reports = []
                for r in rows:
                    verdict_preview = r["verdict_text"][:30] + "..." if len(r["verdict_text"]) > 30 else r["verdict_text"]
                    formatted_reports.append({
                        "report_id": r["report_id"],
                        "gene_symbol": r["gene_symbol"],
                        "verdict": verdict_preview,
                        "timestamp": r["generated_at"]
                    })
                return formatted_reports
        except Exception as e:
            print(f"[Database] Failed to list recent reports: {e}")
            return []

    # -- test isolation helper ---------------------------------------------
    def reset(self) -> None:
        """
        Wipes all Module 1 tables and resets identity sequences. Used by the
        pytest fixtures for a clean slate per test, replacing the SQLite
        version's "fresh tempfile per test" strategy (Postgres doesn't have
        a throwaway-file equivalent, so we truncate instead).
        """
        with self.conn.cursor() as cur:
            tables = ", ".join(
                _MODULE4_TABLES_CHILD_FIRST
                + _MODULE1_TABLES_CHILD_FIRST
                + _MODULE2_TABLES_CHILD_FIRST
                + _MODULE3_TABLES_CHILD_FIRST
            )
            cur.execute(f"TRUNCATE {tables} RESTART IDENTITY CASCADE")
        self.conn.commit()

    def reset_module2(self) -> None:
        """
        Wipes only DRUGGABILITY_RESULT and PROTEIN_STRUCTURE, leaving
        Module 1's DISEASE/GENE/etc. rows untouched. Handy for Module 2's
        own test suite so it doesn't need to re-seed genes every test.
        """
        with self.conn.cursor() as cur:
            tables = ", ".join(_MODULE2_TABLES_CHILD_FIRST)
            cur.execute(f"TRUNCATE {tables} RESTART IDENTITY CASCADE")
        self.conn.commit()

    def reset_module3(self) -> None:
        """
        Wipes only OFF_TARGET_SITE, CRISPR_SAFETY_RESULT, and
        GRNA_SUBMISSION, leaving Module 1/2's rows untouched. Handy for
        Module 3's own test suite so it doesn't need to re-seed genes
        every test.
        """
        with self.conn.cursor() as cur:
            tables = ", ".join(_MODULE3_TABLES_CHILD_FIRST)
            cur.execute(f"TRUNCATE {tables} RESTART IDENTITY CASCADE")
        self.conn.commit()

    def reset_module4(self) -> None:
        """
        Wipes only INTEGRATED_REPORT, leaving Modules 1-3's rows untouched.
        Handy for Module 4's own test suite so it doesn't need to re-run
        the full pipeline every test.
        """
        with self.conn.cursor() as cur:
            tables = ", ".join(_MODULE4_TABLES_CHILD_FIRST)
            cur.execute(f"TRUNCATE {tables} RESTART IDENTITY CASCADE")
        self.conn.commit()

    def close(self):    
        self.conn.close()