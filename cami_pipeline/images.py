import hashlib
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import quote

import requests

from casmi_pipeline.io_utils import write_bytes_atomic, write_json_atomic
from casmi_pipeline.settings import PUBCHEM_CID_URL, PUBCHEM_PNG_URL


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def is_png_bytes(data: bytes) -> bool:
    return bool(data) and data.startswith(PNG_SIGNATURE)


def is_valid_png_file(path: Path) -> bool:
    try:
        with path.open("rb") as stream:
            return stream.read(len(PNG_SIGNATURE)) == PNG_SIGNATURE
    except OSError:
        return False


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
        if is_png_bytes(response.content):
            return response.content
    except Exception:
        return None
    return None


def render_smiles_with_rdkit(smiles: str, output_path: Path) -> bool:
    try:
        from rdkit import Chem, rdBase
        from rdkit.Chem import Draw
    except Exception:
        return False

    with rdBase.BlockLogs():
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    try:
        Draw.MolToFile(mol, str(output_path), size=(500, 500))
        return is_valid_png_file(output_path)
    except Exception:
        return False


def render_smiles_with_rdkit_relaxed(smiles: str, output_path: Path) -> bool:
    """Render unusual valence states without changing the original molecular graph."""
    try:
        from rdkit import Chem, rdBase
        from rdkit.Chem import Draw, rdDepictor
    except Exception:
        return False

    try:
        with rdBase.BlockLogs():
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                return False
            # Preserve aromatic/ring/stereo processing but skip strict valence-property checks.
            sanitize_ops = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
            Chem.SanitizeMol(mol, sanitizeOps=sanitize_ops, catchErrors=True)
        mol.UpdatePropertyCache(strict=False)
        rdDepictor.Compute2DCoords(mol)
        Draw.MolToFile(mol, str(output_path), size=(700, 500))
        return is_valid_png_file(output_path)
    except Exception:
        return False


def render_structure_placeholder(smiles: str, output_path: Path) -> bool:
    """Create a valid, explicit placeholder instead of aborting an evaluation."""
    try:
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (900, 360), "white")
        draw = ImageDraw.Draw(image)
        draw.text((30, 30), "Structure image unavailable", fill="black")
        draw.text((30, 65), "The original SMILES is preserved below:", fill="black")
        wrapped = textwrap.wrap(smiles or "unknown", width=105)
        draw.multiline_text((30, 105), "\n".join(wrapped[:10]), fill="black", spacing=8)
        image.save(output_path, format="PNG")
        return is_valid_png_file(output_path)
    except Exception:
        return False


def get_structure_image(
    session: requests.Session,
    smiles: str,
    image_dir: Path,
    timeout: int,
    force: bool = False,
    prefer_rdkit: bool = False,
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
        except Exception:
            return {}

    if image_path.exists() and not force and is_valid_png_file(image_path):
        cached_meta = read_meta()
        original_source = cached_meta.get("original_image_source")
        result = {
            "image_path": str(image_path),
            "image_source": original_source or "cached",
            "cache_hit": True,
            "image_available": True,
        }
        if "pubchem_cid" in cached_meta:
            result["pubchem_cid"] = cached_meta["pubchem_cid"]
        if cached_meta.get("original_image_source"):
            result["original_image_source"] = cached_meta["original_image_source"]
        if cached_meta.get("structure_image_warning"):
            result["structure_image_warning"] = cached_meta["structure_image_warning"]
        return result

    if smiles and prefer_rdkit and render_smiles_with_rdkit(smiles, image_path):
        write_meta({"original_image_source": "RDKit Rendered"})
        return {
            "image_path": str(image_path),
            "image_source": "RDKit Rendered",
            "original_image_source": "RDKit Rendered",
            "image_available": True,
        }

    if smiles and prefer_rdkit and render_smiles_with_rdkit_relaxed(smiles, image_path):
        warning = "RDKit strict sanitization failed; rendered from the original graph without property sanitization."
        write_meta(
            {
                "original_image_source": "RDKit Rendered (non-strict)",
                "structure_image_warning": warning,
            }
        )
        return {
            "image_path": str(image_path),
            "image_source": "RDKit Rendered (non-strict)",
            "original_image_source": "RDKit Rendered (non-strict)",
            "structure_image_warning": warning,
            "image_available": True,
        }

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
                    "original_image_source": "PubChem Official",
                    "pubchem_cid": cid,
                    "image_available": True,
                }

    if smiles and render_smiles_with_rdkit(smiles, image_path):
        write_meta({"original_image_source": "RDKit Rendered"})
        return {
            "image_path": str(image_path),
            "image_source": "RDKit Rendered",
            "original_image_source": "RDKit Rendered",
            "image_available": True,
        }

    if smiles and render_smiles_with_rdkit_relaxed(smiles, image_path):
        warning = "RDKit strict sanitization failed; rendered from the original graph without property sanitization."
        write_meta(
            {
                "original_image_source": "RDKit Rendered (non-strict)",
                "structure_image_warning": warning,
            }
        )
        return {
            "image_path": str(image_path),
            "image_source": "RDKit Rendered (non-strict)",
            "original_image_source": "RDKit Rendered (non-strict)",
            "structure_image_warning": warning,
            "image_available": True,
        }

    if cached_image_exists and is_valid_png_file(image_path):
        cached_meta = read_meta()
        original_source = cached_meta.get("original_image_source")
        result = {
            "image_path": str(image_path),
            "image_source": original_source or "cached_fallback",
            "cache_hit": True,
            "image_available": True,
        }
        if "pubchem_cid" in cached_meta:
            result["pubchem_cid"] = cached_meta["pubchem_cid"]
        if cached_meta.get("original_image_source"):
            result["original_image_source"] = cached_meta["original_image_source"]
        if cached_meta.get("structure_image_warning"):
            result["structure_image_warning"] = cached_meta["structure_image_warning"]
        return result

    if render_structure_placeholder(smiles, image_path):
        warning = "PubChem and RDKit structure rendering failed; generated a text placeholder."
        write_meta(
            {
                "original_image_source": "Structure Placeholder",
                "structure_image_warning": warning,
            }
        )
        return {
            "image_path": str(image_path),
            "image_source": "Structure Placeholder",
            "original_image_source": "Structure Placeholder",
            "structure_image_warning": warning,
            "image_available": True,
        }

    return {
        "image_path": str(image_path),
        "image_source": "unavailable",
        "structure_image_warning": "All image-generation methods failed; continue with text-only structure input.",
        "image_available": False,
    }
