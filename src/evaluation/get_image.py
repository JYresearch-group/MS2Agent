# -*- conding: utf-8 -*-
# @Time    : 2026/4/5  20:04
# @Author  : psi

from casmi_pipeline.images import get_structure_image
import requests
import argparse
import os
import time
from pathlib import Path
import json
from tqdm import tqdm


DEFAULT_IMAGE_DIR = "./data/image"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SMILES images, call two LLM prompts, build embeddings, and compute similarities."
    )
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR, help="Image directory.")
    parser.add_argument("--timeout", type=int, default=180, help="HTTP timeout.")
    parser.add_argument("--force", action="store_true", help="Ignore cache where possible.")
    return parser.parse_args()


def get_image(file):
    args = parse_args()

    data = json.load(open(file, 'r', encoding='utf-8'))

    session = requests.Session()
    image_dir = Path(args.image_dir)

    for d in tqdm(data, desc='process ...'):
        smiles = d['smiles']
        image_meta = get_structure_image(
            session=session,
            smiles=smiles,
            image_dir=image_dir,
            timeout=args.timeout,
            force=args.force,
        )


if __name__ == '__main__':
    file = "./data/CASMI2016_Cat2and3_Challenge_1-pos.instrument_fields_smiles.json"
    file = './data/casmi2022_all_in_one_structured.instrument_fields_smiles.json'
    get_image(file)
