"""Azure ML Pipeline for the H&M two-tower recommender.

Full end-to-end pipeline: raw data → sample → preprocess → train → evaluate → register.
No local data prep required — everything runs on Azure compute.

Prerequisites (one-time setup):
  Register the raw CSVs as a Data Asset:
    az ml data create --name hm-raw-data --type uri_folder \\
      --path azureml://datastores/workspaceblobstore/paths/raw/

  (Optional, for --use-clip) Register pre-computed CLIP embeddings:
    az ml data create --name hm-clip-embeddings --type uri_folder \\
      --path azureml://datastores/workspaceblobstore/paths/clip-embeddings/

Usage:
    # Full pipeline, 26-week window, 30 epochs:
    python azure/pipeline.py --weeks 26 --epochs 30

    # Full pipeline with CLIP multi-modal embeddings:
    python azure/pipeline.py --weeks 26 --epochs 30 --use-clip

    # Skip sample + preprocess; use an existing processed Data Asset:
    python azure/pipeline.py --data-version 52w --epochs 30
"""

import argparse
from datetime import datetime

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.entities import Data, Model
from azure.identity import AzureCliCredential

SUBSCRIPTION_ID = "d9431e53-775d-40b1-9936-6dfa5af12ee4"
RESOURCE_GROUP  = "resource_jhu_rec_sys"
WORKSPACE_NAME  = "JHU_rec_sys"
CPU_COMPUTE     = "large-cluster"
GPU_COMPUTE     = "hm-gpu-cluster"
ENVIRONMENT     = "azureml:hm-recommender-training:1"
EXPERIMENT      = "hm-two-tower-recommender"
DATASTORE_PATH  = "azureml://datastores/workspaceblobstore/paths"


def get_client() -> MLClient:
    # AzureCliCredential goes directly to the active `az login` session.
    # DefaultAzureCredential probes ~8 credential providers sequentially,
    # causing 30-60 second hangs before falling through to the CLI token.
    return MLClient(
        AzureCliCredential(),
        SUBSCRIPTION_ID,
        RESOURCE_GROUP,
        WORKSPACE_NAME,
    )


def run_full_pipeline(client: MLClient, weeks: int, epochs: int, use_clip: bool, use_gpu: bool) -> None:
    """raw data → sample → preprocess → train → evaluate → register."""
    version = f"{weeks}w-{datetime.now().strftime('%Y%m%d-%H%M')}"
    compute = GPU_COMPUTE if use_gpu else CPU_COMPUTE

    base_path = f"{DATASTORE_PATH}/pipeline/{version}"
    sample_path     = f"{base_path}/sample/"
    processed_path  = f"{base_path}/processed/"
    checkpoint_path = f"{base_path}/checkpoints/"

    raw_asset = client.data.get(name="hm-raw-data", label="latest")
    raw_input = Input(
        type=AssetTypes.URI_FOLDER,
        path=f"azureml:hm-raw-data:{raw_asset.version}",
        mode=InputOutputModes.RO_MOUNT,
    )

    # ── Step 1: Sample ───────────────────────────────────────────────────────
    print(f"\n>>> [{version}] Step 1/4: sample ({weeks} weeks)")
    sample_job = client.jobs.create_or_update(
        command(
            display_name=f"[{version}] sample",
            command=(
                "python -m src.data.sample"
                " --raw-dir ${{inputs.raw_data}}"
                " --output ${{outputs.sample}}/transactions_sample.parquet"
                f" --weeks {weeks}"
            ),
            inputs={"raw_data": raw_input},
            outputs={"sample": Output(type=AssetTypes.URI_FOLDER, path=sample_path)},
            environment=ENVIRONMENT,
            compute=compute,
            code=".",
            experiment_name=EXPERIMENT,
            tags={"version": version, "step": "sample"},
        )
    )
    print(f"    Job : {sample_job.name}")
    print(f"    URL : {sample_job.studio_url}")
    client.jobs.stream(sample_job.name)

    # ── Step 2: Preprocess ───────────────────────────────────────────────────
    print(f"\n>>> [{version}] Step 2/4: preprocess")
    preprocess_job = client.jobs.create_or_update(
        command(
            display_name=f"[{version}] preprocess",
            command=(
                "python -m src.data.preprocess"
                " --raw-dir ${{inputs.raw_data}}"
                " --transactions-path ${{inputs.sample}}/transactions_sample.parquet"
                " --output-dir ${{outputs.processed}}"
            ),
            inputs={
                "raw_data": raw_input,
                "sample": Input(type=AssetTypes.URI_FOLDER, path=sample_path, mode=InputOutputModes.RO_MOUNT),
            },
            outputs={"processed": Output(type=AssetTypes.URI_FOLDER, path=processed_path)},
            environment=ENVIRONMENT,
            compute=compute,
            code=".",
            experiment_name=EXPERIMENT,
            tags={"version": version, "step": "preprocess"},
        )
    )
    print(f"    Job : {preprocess_job.name}")
    print(f"    URL : {preprocess_job.studio_url}")
    client.jobs.stream(preprocess_job.name)

    # Register the processed folder as a versioned Data Asset so future runs can
    # skip sample + preprocess and start directly from here with --data-version.
    print(f"\n>>> Registering hm-processed-data:{version}")
    client.data.create_or_update(
        Data(
            name="hm-processed-data",
            version=version,
            path=processed_path,
            type=AssetTypes.URI_FOLDER,
            description=f"Processed H&M features | {weeks}-week window | run {version}",
        )
    )
    print(f"    Done: hm-processed-data:{version}")

    processed_input = Input(
        type=AssetTypes.URI_FOLDER,
        path=processed_path,
        mode=InputOutputModes.RO_MOUNT,
    )

    # ── Step 3: Train ────────────────────────────────────────────────────────
    # The freshly generated processed folder doesn't have CLIP embeddings —
    # preprocess.py doesn't run embed_images.py. Mount the latest existing
    # hm-processed-data asset (which has articles_clip_embeddings.parquet from
    # the manual upload) as a separate read-only input and point --clip-dir at it.
    train_inputs = {"processed_data": processed_input}
    train_clip_flag = ""
    if use_clip:
        clip_source_asset = client.data.get(name="hm-processed-data", label="latest")
        train_inputs["clip_source"] = Input(
            type=AssetTypes.URI_FOLDER,
            path=f"azureml:hm-processed-data:{clip_source_asset.version}",
            mode=InputOutputModes.RO_MOUNT,
        )
        train_clip_flag = " --use-clip --clip-dir ${{inputs.clip_source}}"

    print(f"\n>>> [{version}] Step 3/4: train ({epochs} epochs, clip={use_clip})")
    train_job = client.jobs.create_or_update(
        command(
            display_name=f"[{version}] train ({epochs} epochs)",
            command=(
                "python -m src.train"
                " --processed-dir ${{inputs.processed_data}}"
                f" --run-name {version}"
                f" --epochs {epochs}"
                " --checkpoint-dir ${{outputs.checkpoint}}"
                f"{train_clip_flag}"
            ),
            inputs=train_inputs,
            outputs={"checkpoint": Output(type=AssetTypes.URI_FOLDER, path=checkpoint_path)},
            environment=ENVIRONMENT,
            compute=compute,
            code=".",
            experiment_name=EXPERIMENT,
            tags={"version": version, "step": "train", "use_clip": str(use_clip)},
        )
    )
    print(f"    Job : {train_job.name}")
    print(f"    URL : {train_job.studio_url}")
    client.jobs.stream(train_job.name)

    # ── Step 4: Evaluate ─────────────────────────────────────────────────────
    eval_inputs = {
        "processed_data": processed_input,
        "checkpoint": Input(type=AssetTypes.URI_FOLDER, path=checkpoint_path, mode=InputOutputModes.RO_MOUNT),
    }
    eval_clip_flag = ""
    if use_clip:
        eval_inputs["clip_source"] = train_inputs["clip_source"]
        eval_clip_flag = " --clip-dir ${{inputs.clip_source}}"

    print(f"\n>>> [{version}] Step 4/4: evaluate")
    eval_job = client.jobs.create_or_update(
        command(
            display_name=f"[{version}] evaluate",
            command=(
                "python -m src.evaluate"
                " --processed-dir ${{inputs.processed_data}}"
                # Double braces: f-string collapses {{ → { giving ${{inputs.checkpoint}}
                # which Azure ML resolves to the actual mount path at runtime.
                f" --checkpoint ${{{{inputs.checkpoint}}}}/two_tower.pt"
                " --split test"
                f"{eval_clip_flag}"
            ),
            inputs=eval_inputs,
            environment=ENVIRONMENT,
            compute=compute,
            code=".",
            experiment_name=EXPERIMENT,
            tags={"version": version, "step": "evaluate"},
        )
    )
    print(f"    Job : {eval_job.name}")
    print(f"    URL : {eval_job.studio_url}")
    client.jobs.stream(eval_job.name)

    # ── Register model ────────────────────────────────────────────────────────
    print(f"\n>>> Registering model: hm-two-tower")
    registered = client.models.create_or_update(
        Model(
            name="hm-two-tower",
            path=checkpoint_path,
            description=f"Two-tower retrieval model | {weeks}w window | {epochs} epochs | clip: {use_clip}",
            tags={
                "checkpoint_blob_path": f"pipeline/{version}/checkpoints/two_tower.pt",
                "version": version,
                "weeks": str(weeks),
                "epochs": str(epochs),
                "use_clip": str(use_clip),
            },
        )
    )
    print(f"    Registered: hm-two-tower:{registered.version}")
    print(f"    → Set MODEL_VERSION={registered.version} in the k8s deployment to serve this model.")
    print(f"\nPipeline complete: {version}")
    print(f"View in Azure ML Studio → Jobs → Experiments → {EXPERIMENT}")


def run_train_pipeline(client: MLClient, data_version: str, epochs: int, use_clip: bool, use_gpu: bool) -> None:
    """Train + evaluate only, starting from an existing hm-processed-data asset."""
    compute = GPU_COMPUTE if use_gpu else CPU_COMPUTE

    all_versions = list(client.data.list(name="hm-processed-data"))
    matching = [a for a in all_versions if data_version in (a.path or "") or a.version == data_version]
    if not matching:
        raise ValueError(
            f"No 'hm-processed-data' asset with version or path containing '{data_version}'.\n"
            f"Run the full pipeline instead: python azure/pipeline.py --weeks N"
        )
    asset_ver = matching[0].version
    print(f"Using hm-processed-data:{asset_ver} (path: {matching[0].path})")

    checkpoint_path = f"{DATASTORE_PATH}/checkpoints/{data_version}/"
    processed_input = Input(
        type=AssetTypes.URI_FOLDER,
        path=f"azureml:hm-processed-data:{asset_ver}",
        mode=InputOutputModes.RO_MOUNT,
    )

    train_inputs = {"processed_data": processed_input}
    train_clip_flag = ""
    if use_clip:
        # The existing hm-processed-data asset already has articles_clip_embeddings.parquet.
        train_inputs["clip_source"] = Input(
            type=AssetTypes.URI_FOLDER,
            path=f"azureml:hm-processed-data:{asset_ver}",
            mode=InputOutputModes.RO_MOUNT,
        )
        train_clip_flag = " --use-clip --clip-dir ${{inputs.clip_source}}"

    print(f"\n>>> [{data_version}] train ({epochs} epochs, clip={use_clip})")
    train_job = client.jobs.create_or_update(
        command(
            display_name=f"[{data_version}] train ({epochs} epochs)",
            command=(
                "python -m src.train"
                " --processed-dir ${{inputs.processed_data}}"
                f" --data-version {data_version}"
                f" --run-name {data_version}"
                f" --epochs {epochs}"
                " --checkpoint-dir ${{outputs.checkpoint}}"
                f"{train_clip_flag}"
            ),
            inputs=train_inputs,
            outputs={"checkpoint": Output(type=AssetTypes.URI_FOLDER, path=checkpoint_path)},
            environment=ENVIRONMENT,
            compute=compute,
            code=".",
            experiment_name=EXPERIMENT,
            tags={"data_version": data_version, "step": "train", "use_clip": str(use_clip)},
        )
    )
    print(f"    Job : {train_job.name}")
    print(f"    URL : {train_job.studio_url}")
    client.jobs.stream(train_job.name)

    eval_inputs = {
        "processed_data": processed_input,
        "checkpoint": Input(type=AssetTypes.URI_FOLDER, path=checkpoint_path, mode=InputOutputModes.RO_MOUNT),
    }
    eval_clip_flag = ""
    if use_clip:
        eval_inputs["clip_source"] = train_inputs["clip_source"]
        eval_clip_flag = " --clip-dir ${{inputs.clip_source}}"

    print(f"\n>>> [{data_version}] evaluate")
    eval_job = client.jobs.create_or_update(
        command(
            display_name=f"[{data_version}] evaluate",
            command=(
                "python -m src.evaluate"
                " --processed-dir ${{inputs.processed_data}}"
                f" --checkpoint ${{{{inputs.checkpoint}}}}/two_tower.pt"
                " --split test"
                f"{eval_clip_flag}"
            ),
            inputs=eval_inputs,
            environment=ENVIRONMENT,
            compute=compute,
            code=".",
            experiment_name=EXPERIMENT,
            tags={"data_version": data_version, "step": "evaluate"},
        )
    )
    print(f"    Job : {eval_job.name}")
    print(f"    URL : {eval_job.studio_url}")
    client.jobs.stream(eval_job.name)

    print(f"\n>>> Registering model: hm-two-tower")
    registered = client.models.create_or_update(
        Model(
            name="hm-two-tower",
            path=checkpoint_path,
            description=f"Two-tower retrieval model | data: {data_version} | epochs: {epochs} | clip: {use_clip}",
            tags={
                "checkpoint_blob_path": f"checkpoints/{data_version}/two_tower.pt",
                "data_version": data_version,
                "epochs": str(epochs),
                "use_clip": str(use_clip),
            },
        )
    )
    print(f"    Registered: hm-two-tower:{registered.version}")
    print(f"    → Set MODEL_VERSION={registered.version} in the k8s deployment to serve this model.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit H&M two-tower training pipeline to Azure ML.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python azure/pipeline.py --weeks 26 --epochs 30\n"
            "  python azure/pipeline.py --weeks 52 --epochs 30 --use-clip\n"
            "  python azure/pipeline.py --data-version 52w --epochs 30\n"
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--weeks", type=int,
        help="Full pipeline: sample N weeks of raw data → preprocess → train → evaluate.",
    )
    mode.add_argument(
        "--data-version", type=str,
        help="Partial pipeline: skip sample + preprocess, train + evaluate on an existing hm-processed-data asset.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--use-clip", action="store_true", help="Include CLIP embeddings (requires hm-clip-embeddings asset)")
    parser.add_argument("--gpu", action="store_true", help="Use GPU compute cluster (hm-gpu-cluster)")
    args = parser.parse_args()

    client = get_client()
    if args.weeks:
        run_full_pipeline(client, args.weeks, args.epochs, use_clip=args.use_clip, use_gpu=args.gpu)
    else:
        run_train_pipeline(client, args.data_version, args.epochs, use_clip=args.use_clip, use_gpu=args.gpu)


if __name__ == "__main__":
    main()
