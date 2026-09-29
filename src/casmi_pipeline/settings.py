import os


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


DEFAULT_INPUT = ["./data/CASMI2016_Cat2and3_Challenge_1-neg.instrument_fields_smiles.json",
                 "./data/CASMI2016_Cat2and3_Challenge_1-pos.instrument_fields_smiles.json",
                 "./data/CASMI2016_Cat2and3_Challenge_1-all.instrument_fields_smiles.json",
                 './data/casmi2022_all_in_one_structured.instrument_fields_smiles.json',
                 './data/GNPS-NIH-NATURALPRODUCTSLIBRARY_ROUND2_POSITIVE_new_with_retrieval_top50.json_with_ooo.json',
                 './data/GNPS-NIH-NATURALPRODUCTSLIBRARY_ROUND2_POSITIVE_new_C15H20O3_11_pepmass_MH_top930_ppm5_guaranteed_candidate_id_spectrumid.json_1.json',
                 './data/GNPS-NIH-NATURALPRODUCTSLIBRARY_ROUND2_POSITIVE_new_with_adduct_50_pepmass_MH_calibrated_guaranteed_candidate_id_spectrumid.json_17.json',
                 './data/GNPS-NIH-NATURALPRODUCTSLIBRARY_ROUND2_POSITIVE_new_C15H22O3_10_200_new.json',
                 './data/siyou_data_20_formula_pubchem_candidates_new_200.json',
                 './data/siyou_data_20_candidates_new_76.json'][-1]

DEFAULT_OUTPUT_DIR = ["./outputs/casmi_retrieval_signature_research_gnps_17",
                      "./outputs/casmi_retrieval_signature_research_gnps_C15H22O3_10_200",
                      "./outputs/casmi_retrieval_signature_siyou_data_20",
                      "./outputs/casmi_retrieval_signature_siyou_data_20_zijian_76"][-1]
DEFAULT_IMAGE_DIR = "./data/image"
PROMPT_VERSION = "retrieval_signature_v1"

CHAT_API_URL = ["https://api.commonstack.ai/v1/chat/completions",
                "https://openrouter.ai/api/v1/chat/completions",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
                'https://api.deepseek.com/chat/completions',
                "http://localhost:8000/"][-1]
PUBCHEM_CID_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/cids/JSON"
PUBCHEM_PNG_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/PNG"
PUBCHEM_PROPERTY_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/"
    "MolecularFormula,ExactMass,HBondDonorCount,HBondAcceptorCount,XLogP/JSON"
)

PROTON_MASS = 1.007276466812
ELECTRON_MASS = 0.000548579909065
WATER_MASS = 18.0105646837
SODIUM_MASS = 22.989218
POTASSIUM_MASS = 38.963158
AMMONIUM_MASS = 18.033823
FORMATE_MASS = 44.998201
ACETATE_MASS = 59.013851
ACETONITRILE_MASS = 41.026549

ADDUCT_21D_MAPPING = {
    "[M+H]+": 0,
    "[M+Na]+": 1,
    "[M+NH4]+": 2,
    "[M-H]-": 3,
    "[M+K]+": 4,
    "[M+HCOOH-H]-": 5,
    "[M]+": 6,
    "[M+CH3COOH-H]-": 7,
    "[M+H-H2O]+": 8,
    "[M+Cl]-": 9,
    "[M-H2O+H]+": 10,
    "[M+2H]2+": 11,
    "[2M+Na]+": 12,
    "[2M+H]+": 13,
    "[M+H-2H2O]+": 14,
    "[M-2H2O+H]+": 15,
    "[M+ACN+H]+": 16,
    "[M-e]+": 17,
    "[2M+NH4]+": 18,
    "[M+3H]3+": 19,
    "[M-H+2Na]+": 20,
}

ADDUCT_SPECS = {
    "[M+H]+": {"multiplier": 1, "delta": PROTON_MASS, "charge": 1, "mode": "positive"},
    "[M+Na]+": {"multiplier": 1, "delta": SODIUM_MASS, "charge": 1, "mode": "positive"},
    "[M+NH4]+": {"multiplier": 1, "delta": AMMONIUM_MASS, "charge": 1, "mode": "positive"},
    "[M-H]-": {"multiplier": 1, "delta": -PROTON_MASS, "charge": -1, "mode": "negative"},
    "[M+K]+": {"multiplier": 1, "delta": POTASSIUM_MASS, "charge": 1, "mode": "positive"},
    "[M+HCOOH-H]-": {"multiplier": 1, "delta": FORMATE_MASS, "charge": -1, "mode": "negative"},
    "[M]+": {"multiplier": 1, "delta": -ELECTRON_MASS, "charge": 1, "mode": "positive"},
    "[M+CH3COOH-H]-": {"multiplier": 1, "delta": ACETATE_MASS, "charge": -1, "mode": "negative"},
    "[M+H-H2O]+": {"multiplier": 1, "delta": PROTON_MASS - WATER_MASS, "charge": 1, "mode": "positive"},
    "[M+Cl]-": {"multiplier": 1, "delta": 34.969402, "charge": -1, "mode": "negative"},
    "[M-H2O+H]+": {"multiplier": 1, "delta": PROTON_MASS - WATER_MASS, "charge": 1, "mode": "positive"},
    "[M+2H]2+": {"multiplier": 1, "delta": 2 * PROTON_MASS, "charge": 2, "mode": "positive"},
    "[2M+Na]+": {"multiplier": 2, "delta": SODIUM_MASS, "charge": 1, "mode": "positive"},
    "[2M+H]+": {"multiplier": 2, "delta": PROTON_MASS, "charge": 1, "mode": "positive"},
    "[M+H-2H2O]+": {"multiplier": 1, "delta": PROTON_MASS - 2 * WATER_MASS, "charge": 1, "mode": "positive"},
    "[M-2H2O+H]+": {"multiplier": 1, "delta": PROTON_MASS - 2 * WATER_MASS, "charge": 1, "mode": "positive"},
    "[M+ACN+H]+": {"multiplier": 1, "delta": ACETONITRILE_MASS + PROTON_MASS, "charge": 1, "mode": "positive"},
    "[M-e]+": {"multiplier": 1, "delta": -ELECTRON_MASS, "charge": 1, "mode": "positive"},
    "[2M+NH4]+": {"multiplier": 2, "delta": AMMONIUM_MASS, "charge": 1, "mode": "positive"},
    "[M+3H]3+": {"multiplier": 1, "delta": 3 * PROTON_MASS, "charge": 3, "mode": "positive"},
    "[M-H+2Na]+": {"multiplier": 1, "delta": 2 * SODIUM_MASS - PROTON_MASS, "charge": 1, "mode": "positive"},
}

ADDUCT_ALIASES = {
    "[M+FA-H]-": "[M+HCOOH-H]-",
    "[M+CH3COO]-": "[M+CH3COOH-H]-",
}

POSITIVE_ADDUCT_DELTAS = {
    key: spec["delta"] for key, spec in ADDUCT_SPECS.items() if spec["mode"] == "positive" and spec["charge"] == 1
}
NEGATIVE_ADDUCT_DELTAS = {
    key: spec["delta"] for key, spec in ADDUCT_SPECS.items() if spec["mode"] == "negative" and spec["charge"] == -1
}

PRIMARY_POSITIVE_ADDUCTS = ["[M+H]+", "[M+Na]+", "[M+NH4]+", "[M+K]+", "[M]+", "[M+H-H2O]+"]
PRIMARY_NEGATIVE_ADDUCTS = ["[M-H]-", "[M+Cl]-", "[M+HCOOH-H]-", "[M+CH3COOH-H]-"]
