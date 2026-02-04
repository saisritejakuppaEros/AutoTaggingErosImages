#!/usr/bin/env python3

import argparse
import toml
from pathlib import Path
import ray

# ==============================
# Import pipeline entry functions
# (ALIAS to avoid name collision)
# ==============================
from scripts.segmentation import (
    segment_source as run_segment_source,
    segment_target as run_segment_target,
)

# Placeholders (keep commented until implemented)
# from scripts.embeddings import (
#     embed_source_dataset,
#     embed_target_dataset,
# )
#
# from scripts.tagging import (
#     build_faiss_index,
#     tag_target_dataset,
# )


# ==============================
# Config Loader
# ==============================
def load_config(config_path: str) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        return toml.load(f)


# ==============================
# Ray Initialization
# ==============================
def init_ray(pipeline_cfg: dict):
    if ray.is_initialized():
        return

    ray.init(
        num_cpus=pipeline_cfg["num_cpus"],
        num_gpus=pipeline_cfg["num_gpus"],
        object_store_memory=pipeline_cfg["object_store_gb"] * 1024**3,
        ignore_reinit_error=True,
    )


# ==============================
# Command Handlers
# ==============================
def cmd_segment_source(cfg):
    init_ray(cfg["tool"]["pipeline"])
    run_segment_source(cfg)


def cmd_segment_target(cfg, folder):
    init_ray(cfg["tool"]["pipeline"])
    run_segment_target(folder, cfg)


# (Keep disabled until implemented)
# def cmd_embed_source(cfg):
#     init_ray(cfg["tool"]["pipeline"])
#     embed_source_dataset(cfg)
#
#
# def cmd_embed_target(cfg, folder):
#     init_ray(cfg["tool"]["pipeline"])
#     embed_target_dataset(cfg, folder)
#
#
# def cmd_tag_target(cfg, folder):
#     init_ray(cfg["tool"]["pipeline"])
#     tag_target_dataset(cfg, folder)
#
#
# def cmd_process_target(cfg, folder):
#     init_ray(cfg["tool"]["pipeline"])
#     run_segment_target(folder, cfg)
#     embed_target_dataset(cfg, folder)
#     tag_target_dataset(cfg, folder)


# ==============================
# CLI Definition
# ==============================
def build_parser():
    parser = argparse.ArgumentParser(
        description="Eros AutoTagging Pipeline (Ray-based)"
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to config.toml",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # -------- Source --------
    subparsers.add_parser("segment-source")

    # -------- Target --------
    p = subparsers.add_parser("segment-target")
    p.add_argument("--folder", required=True)

    return parser


# ==============================
# Main Entrypoint
# ==============================
def main():
    parser = build_parser()
    args = parser.parse_args()

    cfg = load_config(args.config)

    print("Loaded config:")
    print(cfg)

    if args.command == "segment-source":
        cmd_segment_source(cfg)

    elif args.command == "segment-target":
        cmd_segment_target(cfg, args.folder)

    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
