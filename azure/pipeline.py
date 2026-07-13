"""Azure ML Pipeline for the H&M two-tower recommender.

Submits two sequential command jobs to Azure ML:
  train → evaluate

Sample and preprocess are already done locally and their outputs are
uploaded to Blob Storage as versioned Data Assets (hm-processed-data:1/2/3).
The pipeline mounts those Data Assets as read-only inputs to the compute VM
so the scripts see a local directory path — no Blob Storage SDK needed inside
the training scripts.

The train job writes its checkpoint to a Blob Storage output folder.
The evaluate job reads from that same folder.

Note: the azure-ai-ml @pipeline decorator was not used because it relies on
Python bytecode inspection that is broken in Python 3.13. Sequential
create_or_update + stream() calls achieve the same result.

Usage:
    python azure/pipeline.py --data-version v1
    python azure/pipeline.py --data-version v2
    python azure/pipeline.py --data-version v3
"""

import argparse

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.ai.ml.entities import Model
from azure.identity import AzureCliCredential

SUBSCRIPTION_ID = "d9431e53-775d-40b1-9936-6dfa5af12ee4"
RESOURCE_GROUP  = "resource_jhu_rec_sys"
WORKSPACE_NAME  = "JHU_rec_sys"
COMPUTE_NAME    = "hm-training-cluster"
ENVIRONMENT     = "azureml:hm-recommender-training:1"
EXPERIMENT      = "hm-two-tower-recommender"
DATASTORE_PATH  = "azureml://datastores/workspaceblobstore/paths"

# Maps data_version string to the registered Data Asset version number
ASSET_VERSION = {"v1": "1", "v2": "2", "v3": "3"}


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


def run_pipeline(data_version: str, epochs: int) -> None:
    client = get_client()
    asset_ver = ASSET_VERSION[data_version]

    # Resolve the registered Data Asset for this version (e.g. hm-processed-data:1 for v1).
    # RO_MOUNT means Azure ML mounts the Blob Storage folder as a read-only local directory
    # on the compute VM — the training script just sees a normal file path, no SDK calls needed.
    processed_input = Input(
        type=AssetTypes.URI_FOLDER,
        path=f"azureml:hm-processed-data:{asset_ver}",
        mode=InputOutputModes.RO_MOUNT,
    )

    # Where the train job will write its checkpoint in Blob Storage.
    # The evaluate job mounts the same path as an input to read the checkpoint back.
    checkpoint_output_path = f"{DATASTORE_PATH}/checkpoints/{data_version}/"

    # ── Step 1: Train ────────────────────────────────────────────────────────
    print(f"\n>>> Submitting: [{data_version}] train ({epochs} epochs)")
    train_job = client.jobs.create_or_update(
        command(
            display_name=f"[{data_version}] train ({epochs} epochs)",
            # ${{inputs.x}} and ${{outputs.x}} are Azure ML template placeholders —
            # the SDK substitutes the actual mounted paths on the compute VM at runtime.
            # code="." bundles the entire repo directory as the job's code snapshot
            # and uploads it to Azure ML before the job starts.
            command=(
                "python -m src.train"
                " --processed-dir ${{inputs.processed_data}}"  # path to mounted Data Asset
                f" --data-version {data_version}"              # logged to Azure ML Experiments for traceability
                f" --run-name {data_version}-26w"
                f" --epochs {epochs}"
                " --checkpoint-dir ${{outputs.checkpoint}}"   # Azure ML writes this folder to Blob Storage
            ),
            inputs={"processed_data": processed_input},
            outputs={
                # URI_FOLDER output: Azure ML creates the folder in Blob Storage and
                # mounts it as a writable local path on the VM during the job.
                "checkpoint": Output(
                    type=AssetTypes.URI_FOLDER,
                    path=checkpoint_output_path,
                )
            },
            environment=ENVIRONMENT,
            compute=COMPUTE_NAME,
            code=".",
            experiment_name=EXPERIMENT,
            tags={"data_version": data_version, "step": "train"},
        )
    )
    assert train_job.name, "Azure ML did not return a job name for train"
    print(f"    Job name : {train_job.name}")
    print(f"    Studio   : {train_job.studio_url}")
    # stream() blocks until the job finishes — this is what makes the steps sequential.
    # The evaluate job must not start until the checkpoint exists in Blob Storage.
    client.jobs.stream(train_job.name)
    print(f"    Done: train")

    # ── Step 2: Evaluate ─────────────────────────────────────────────────────
    # Mounts the same checkpoint folder the train job just wrote as a read-only input.
    print(f"\n>>> Submitting: [{data_version}] evaluate")
    eval_job = client.jobs.create_or_update(
        command(
            display_name=f"[{data_version}] evaluate",
            command=(
                "python -m src.evaluate"
                " --processed-dir ${{inputs.processed_data}}"
                # Double braces: the outer f-string collapses {{ → {, producing
                # ${{inputs.checkpoint}} which Azure ML then resolves to the mount path.
                f" --checkpoint ${{{{inputs.checkpoint}}}}/two_tower.pt"
                " --split test"
            ),
            inputs={
                "processed_data": processed_input,
                "checkpoint": Input(
                    type=AssetTypes.URI_FOLDER,
                    path=checkpoint_output_path,
                    mode=InputOutputModes.RO_MOUNT,
                ),
            },
            environment=ENVIRONMENT,
            compute=COMPUTE_NAME,
            code=".",
            experiment_name=EXPERIMENT,
            tags={"data_version": data_version, "step": "evaluate"},
        )
    )
    assert eval_job.name, "Azure ML did not return a job name for evaluate"
    print(f"    Job name : {eval_job.name}")
    print(f"    Studio   : {eval_job.studio_url}")
    client.jobs.stream(eval_job.name)
    print(f"    Done: evaluate")

    # ── Step 3: Register model ────────────────────────────────────────────────
    # Tags are how serve.py looks up the exact Blob paths at pod startup.
    # Omitting `version` lets Azure ML auto-increment (1, 2, 3, ...) so every
    # pipeline run is a distinct, recoverable entry in the registry.
    print(f"\n>>> Registering model: hm-two-tower")
    registered = client.models.create_or_update(
        Model(
            name="hm-two-tower",
            path=checkpoint_output_path,
            description=f"Two-tower retrieval model | data: {data_version} | epochs: {epochs}",
            tags={
                "checkpoint_blob_path": f"checkpoints/{data_version}/two_tower.pt",
                "data_version": data_version,
                "epochs": str(epochs),
            },
        )
    )
    print(f"    Registered: hm-two-tower:{registered.version}")
    print(f"    → Set MODEL_VERSION={registered.version} in the k8s deployment to serve this model.")

    print(f"\nAll steps complete for {data_version}.")
    print(f"View in Azure ML Studio → Jobs → Experiments → {EXPERIMENT}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-version", required=True, choices=["v1", "v2", "v3"])
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    run_pipeline(args.data_version, args.epochs)


if __name__ == "__main__":
    main()
