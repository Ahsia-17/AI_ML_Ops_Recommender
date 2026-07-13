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

Usage:
    python azure/pipeline.py --data-version v1
    python azure/pipeline.py --data-version v2
    python azure/pipeline.py --data-version v3
"""

import argparse

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.constants import AssetTypes, InputOutputModes
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
    return MLClient(
        AzureCliCredential(),
        SUBSCRIPTION_ID,
        RESOURCE_GROUP,
        WORKSPACE_NAME,
    )


def run_pipeline(data_version: str, epochs: int) -> None:
    client = get_client()
    asset_ver = ASSET_VERSION[data_version]

    processed_input = Input(
        type=AssetTypes.URI_FOLDER,
        path=f"azureml:hm-processed-data:{asset_ver}",
        mode=InputOutputModes.RO_MOUNT,  # mounted as a read-only local directory on the VM
    )

    checkpoint_output_path = f"{DATASTORE_PATH}/checkpoints/{data_version}/"

    # ── Step 1: Train ────────────────────────────────────────────────────────
    print(f"\n>>> Submitting: [{data_version}] train ({epochs} epochs)")
    train_job = client.jobs.create_or_update(
        command(
            display_name=f"[{data_version}] train ({epochs} epochs)",
            command=(
                "python -m src.train"
                " --processed-dir ${{inputs.processed_data}}"
                f" --data-version {data_version}"
                f" --run-name {data_version}-26w"
                f" --epochs {epochs}"
                " --checkpoint-dir ${{outputs.checkpoint}}"
            ),
            inputs={"processed_data": processed_input},
            outputs={
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
    client.jobs.stream(train_job.name)
    print(f"    Done: train")

    # ── Step 2: Evaluate ─────────────────────────────────────────────────────
    print(f"\n>>> Submitting: [{data_version}] evaluate")
    eval_job = client.jobs.create_or_update(
        command(
            display_name=f"[{data_version}] evaluate",
            command=(
                "python -m src.evaluate"
                " --processed-dir ${{inputs.processed_data}}"
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
