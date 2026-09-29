import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import quote

import requests

from casmi_pipeline.io_utils import write_bytes_atomic, write_json_atomic
from casmi_pipeline.settings import PUBCHEM_CID_URL, PUBCHEM_PNG_URL


def smiles_to_filename(smiles: str, max_length: int = 180) -> str:
    encoded = quote(smiles, safe="")
    if not encoded:
        encoded = "unknown_smiles"
    if len(encoded) > max_length:
        digest = hashlib.sha256(smiles.encode("utf-8")).hexdigest()[:12]
        encoded = encoded[: max_length - 13] + "_" + digest
    return f"{encoded}.png"


def fetch_pubchem_cid(session: requests.Session, smiles: str, timeout: int) -> Optional[int]:
    variants = [
        ("POST", {"data": {"smiles": smiles}}),
        ("GET", {"params": {"smiles": smiles}}),
    ]
    for method, kwargs in variants:
        try:
            if method == "POST":
                response = session.post(PUBCHEM_CID_URL, timeout=timeout, **kwargs)
            else:
                response = session.get(PUBCHEM_CID_URL, timeout=timeout, **kwargs)
            response.raise_for_status()
            data = response.json()
            cids = data.get("IdentifierList", {}).get("CID", [])
            if cids:
                return int(cids[0])
        except Exception:
            continue
    return None


def fetch_pubchem_png(session: requests.Session, cid: int, timeout: int) -> Optional[bytes]:
    try:
        response = session.get(PUBCHEM_PNG_URL.format(cid=cid), timeout=timeout)
        response.raise_for_status()
        if response.content:
            return response.content
    except Exception:
        return None
    return None


def render_smiles_with_rdkit(smiles: str, output_path: Path) -> bool:
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
    except Exception:
        return False

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    try:
        Draw.MolToFile(mol, str(output_path), size=(500, 500))
        return output_path.exists()
    except Exception:
        return False


def get_structure_image(
    session: requests.Session,
    smiles: str,
    image_dir: Path,
    timeout: int,
    force: bool = False,
) -> Dict[str, Any]:
    image_dir.mkdir(parents=True, exist_ok=True)
    filename = smiles_to_filename(smiles)
    image_path = image_dir / filename
    meta_path = image_dir / f"{Path(filename).stem}.meta.json"
    cached_image_exists = image_path.exists()

    def write_meta(meta: Dict[str, Any]) -> None:
        write_json_atomic(meta_path, meta)

    def read_meta() -> Dict[str, Any]:
        if not meta_path.exists():
            return {}
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            return {}

    if image_path.exists() and not force:
        cached_meta = read_meta()
        result = {
            "image_path": str(image_path),
            "image_source": "cached",
            "image_available": True,
        }
        if "pubchem_cid" in cached_meta:
            result["pubchem_cid"] = cached_meta["pubchem_cid"]
        if cached_meta.get("original_image_source"):
            result["original_image_source"] = cached_meta["original_image_source"]
        return result

    if smiles:
        cid = fetch_pubchem_cid(session, smiles, timeout)
        if cid is not None:
            png_bytes = fetch_pubchem_png(session, cid, timeout)
            if png_bytes:
                write_bytes_atomic(image_path, png_bytes)
                write_meta({"pubchem_cid": cid, "original_image_source": "PubChem Official"})
                return {
                    "image_path": str(image_path),
                    "image_source": "PubChem Official",
                    "pubchem_cid": cid,
                    "image_available": True,
                }

    if smiles and render_smiles_with_rdkit(smiles, image_path):
        write_meta({"original_image_source": "RDKit Rendered"})
        return {
            "image_path": str(image_path),
            "image_source": "RDKit Rendered",
            "image_available": True,
        }

    if cached_image_exists:
        cached_meta = read_meta()
        result = {
            "image_path": str(image_path),
            "image_source": "cached_fallback",
            "image_available": True,
        }
        if "pubchem_cid" in cached_meta:
            result["pubchem_cid"] = cached_meta["pubchem_cid"]
        if cached_meta.get("original_image_source"):
            result["original_image_source"] = cached_meta["original_image_source"]
        return result

    return {
        "image_path": str(image_path),
        "image_source": "unavailable",
        "image_available": False,
    }
