from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ============================================================
# BioMedix-AI existing modules
# ============================================================

from data_ingestor import (
    DataIngestor,
    OpenTargetsProvider,
    MockGeneProvider,
    FallbackGeneProvider,
)

from network_analyzer import NetworkAnalyzer
from target_discovery import TargetDiscoveryPipeline

from structural_ml import (
    PocketDetector,
    FeatureExtractor,
    DruggabilityEngine,
)

from crispr_safety import (
    SequenceFetcher,
    OffTargetScanner,
    CrisprSafetyEngine,
)

from pipeline_runner import (
    PipelineRunner,
    fetch_best_structure,
)


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="BioMedix-AI API",
    description="AI-Driven Drug Discovery and CRISPR-Cas Therapeutic Design API",
    version="1.0.0",
)


# ============================================================
# CORS
# Allows Android / React frontend to call the API
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Initialize Module 1
# ============================================================

real_provider = OpenTargetsProvider()

mock_provider = MockGeneProvider()

gene_provider = FallbackGeneProvider(
    primary=real_provider,
    fallback=mock_provider,
)

data_ingestor = DataIngestor(
    gene_provider=gene_provider
)

network_analyzer = NetworkAnalyzer()

target_pipeline = TargetDiscoveryPipeline(
    ingestor=data_ingestor,
    analyzer=network_analyzer,
    db=None,
)


# ============================================================
# Initialize Module 2
# ============================================================

pocket_detector = PocketDetector(
    fpocket_binary="fpocket"
)

feature_extractor = FeatureExtractor()

druggability_model_path = "druggability_model.joblib"

druggability_engine = DruggabilityEngine(
    model_path=druggability_model_path
)

druggability_pipeline = __import__(
    "structural_ml"
).DruggabilityPipeline(
    detector=pocket_detector,
    extractor=feature_extractor,
    engine=druggability_engine,
    db=None,
)


# ============================================================
# Initialize Module 3
# ============================================================

sequence_fetcher = SequenceFetcher()

off_target_scanner = OffTargetScanner(
    pam_pattern="NGG"
)

crispr_engine = CrisprSafetyEngine()

crispr_pipeline = __import__(
    "crispr_safety"
).CrisprSafetyPipeline(
    fetcher=sequence_fetcher,
    scanner=off_target_scanner,
    engine=crispr_engine,
    db=None,
)


# ============================================================
# Complete Pipeline
# ============================================================

pipeline_runner = PipelineRunner(
    target_pipeline=target_pipeline,
    druggability_pipeline=druggability_pipeline,
    crispr_pipeline=crispr_pipeline,
    db=None,
    organism="Homo sapiens",
)


# ============================================================
# Request Models
# ============================================================

class AnalyzeRequest(BaseModel):
    disease_name: str = Field(
        ...,
        description="Disease or biological condition"
    )

    species: str = Field(
        default="human",
        description="Species such as human, mouse or rice"
    )

    guide_rna: Optional[str] = Field(
        default=None,
        description="Optional CRISPR guide RNA"
    )

    gene_limit: int = Field(
        default=10,
        ge=1,
        le=50
    )

    centrality_method: str = Field(
        default="degree"
    )

    max_mismatches: int = Field(
        default=6,
        ge=0,
        le=10
    )


class DruggabilityRequest(BaseModel):
    gene_symbol: str


class CrisprRequest(BaseModel):
    gene_symbol: str
    guide_rna: str

    species: str = "human"

    max_mismatches: int = Field(
        default=6,
        ge=0,
        le=10
    )


# ============================================================
# Helper functions
# ============================================================

def result_to_dict(obj):
    """
    Convert dataclasses / normal Python objects
    into JSON-friendly dictionaries.
    """

    if obj is None:
        return None

    if hasattr(obj, "__dataclass_fields__"):
        result = {}

        for field_name in obj.__dataclass_fields__:
            value = getattr(obj, field_name)

            if isinstance(value, list):
                result[field_name] = [
                    result_to_dict(item)
                    for item in value
                ]

            elif hasattr(value, "__dataclass_fields__"):
                result[field_name] = result_to_dict(value)

            elif isinstance(value, dict):
                result[field_name] = {
                    str(k): result_to_dict(v)
                    for k, v in value.items()
                }

            else:
                result[field_name] = value

        return result

    if isinstance(obj, list):
        return [result_to_dict(x) for x in obj]

    if isinstance(obj, dict):
        return {
            str(k): result_to_dict(v)
            for k, v in obj.items()
        }

    return obj


def species_to_organism(species: str) -> str:
    mapping = {
        "human": "Homo sapiens",
        "mouse": "Mus musculus",
        "rice": "Oryza sativa",
        "arabidopsis": "Arabidopsis thaliana",
        "zebrafish": "Danio rerio",
    }

    return mapping.get(
        species.lower(),
        "Homo sapiens"
    )


# ============================================================
# Health Check
# ============================================================

@app.get("/")
def root():
    return {
        "project": "BioMedix-AI",
        "status": "running",
        "message": "BioMedix-AI API is working"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "BioMedix-AI API"
    }


# ============================================================
# MODULE 1
# Data Ingestion + Network Analysis
# ============================================================

@app.post("/api/targets")
def discover_targets(request: AnalyzeRequest):

    try:

        result = target_pipeline.run(
            disease_name=request.disease_name,
            gene_limit=request.gene_limit,
            species=request.species,
            centrality_method=request.centrality_method,
        )

        return {
            "success": True,
            "module": "Data Ingestion + Target Discovery",
            "disease": request.disease_name,
            "result": result_to_dict(result),
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# MODULE 2
# Druggability
# ============================================================

@app.post("/api/druggability")
def analyze_druggability(
    request: DruggabilityRequest
):

    try:

        gene = request.gene_symbol

        # Find structure using existing RCSB helper
        structure_info = fetch_best_structure(
            gene_symbol=gene,
            organism="Homo sapiens"
        )

        if structure_info is not None:

            structure_path = structure_info["structure_path"]
            pdb_id = structure_info["pdb_id"]

        else:

            structure_path = None
            pdb_id = None

        result = druggability_pipeline.run(
            gene_symbol=gene,
            structure_path=structure_path,
            pdb_id=pdb_id,
        )

        return {
            "success": True,
            "module": "Druggability Analysis",
            "result": result_to_dict(result),
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# MODULE 3
# CRISPR Safety
# ============================================================

@app.post("/api/crispr")
def analyze_crispr(
    request: CrisprRequest
):

    try:

        result = crispr_pipeline.run(
            gene_symbol=request.gene_symbol,
            guide_rna=request.guide_rna,
            species=request.species,
            max_mismatches=request.max_mismatches,
        )

        return {
            "success": True,
            "module": "CRISPR Safety",
            "result": result_to_dict(result),
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# COMPLETE BIO-MEDIX ANALYSIS
# Module 1 → Module 2 + Module 3 → Verdict
# ============================================================

@app.post("/api/analyze")
def complete_analysis(
    request: AnalyzeRequest
):

    try:

        result = pipeline_runner.run_full_pipeline(
            disease_name=request.disease_name,
            guide_rna=request.guide_rna,
            species=request.species,
            gene_limit=request.gene_limit,
            centrality_method=request.centrality_method,
            max_mismatches=request.max_mismatches,
        )

        return {
            "success": True,
            "project": "BioMedix-AI",
            "result": result_to_dict(result),
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# Run server directly
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
