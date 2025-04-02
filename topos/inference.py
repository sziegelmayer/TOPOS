# Copyright 2025 The TOPOS Authors
# Licensed under the Apache License, Version 2.0

import os
import json
import shutil
import torch
import numpy as np
import nibabel as nib
import nnunetv2
from torch import device
from .downloader import get_checkpoint_path
from .env import setup_nnunet_env
setup_nnunet_env()

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager



def predict(input_images_dir, output_dir, model_config="2d", threshold=0.5):
    """
    Runs nnU-Net inference and postprocessing in one step.

    Args:
        input_images_dir (str): Path to folder with input .nii.gz files.
        output_dir (str): Path to final folder for saving labeled .nii.gz segmentations.
        model_config (str): nnU-Net model config (e.g. "2d", "3d_fullres")
        threshold (float): Threshold for binarizing probability maps.
    """
    ckpt_path, plans_path, dataset_json_path = get_checkpoint_path()
    ckpt_dir = os.path.dirname(ckpt_path)

    # Temporary folder for raw nnUNet predictions
    intermediate_npz_dir = os.path.join(output_dir, "_tmp_npz")
    os.makedirs(intermediate_npz_dir, exist_ok=True)

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device('cuda' if torch.cuda.is_available() else 'cpu'),
        verbose=False,
        allow_tqdm=True
    )

    # Load model weights manually
    checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'), weights_only=False)

    trainer_name = checkpoint['trainer_name']
    configuration_name = checkpoint['init_args']['configuration']
    inference_allowed_mirroring_axes = checkpoint.get('inference_allowed_mirroring_axes', None)

    # Load plans and dataset.json from disk
    with open(plans_path) as f:
        plans = json.load(f)

    with open(dataset_json_path) as f:
        dataset_json = json.load(f)

    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration(configuration_name)

    trainer_class = nnunetv2.utilities.find_class_by_name.recursive_find_python_class(
        os.path.join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
        trainer_name,
        "nnunetv2.training.nnUNetTrainer"
    )

    network = trainer_class.build_network_architecture(
        configuration_manager.network_arch_class_name,
        configuration_manager.network_arch_init_kwargs,
        configuration_manager.network_arch_init_kwargs_req_import,
        nnunetv2.utilities.label_handling.label_handling.determine_num_input_channels(
            plans_manager, configuration_manager, dataset_json),
        plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
        enable_deep_supervision=False
    )
    network.load_state_dict(checkpoint['network_weights'])

    # Final setup
    predictor.manual_initialization(
        network=network,
        plans_manager=plans_manager,
        configuration_manager=configuration_manager,
        parameters=[checkpoint['network_weights']],
        dataset_json=dataset_json,
        trainer_name=trainer_name,
        inference_allowed_mirroring_axes=inference_allowed_mirroring_axes
    )

    predictor.predict_from_files(
        list_of_lists_or_source_folder=input_images_dir,
        output_folder_or_list_of_truncated_output_files=intermediate_npz_dir,
        save_probabilities=True,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2
    )

    print("Inference complete. Running postprocessing...")

    def transform_mask(mask):
        mask_rot = np.rot90(mask, k=-1)
        return np.fliplr(mask_rot)

    labels = {
    "aorta": 1,
    "clavicula_left": 2,
    "clavicula_right": 3,
    "colon": 4,
    "femur_left": 5,
    "femur_right": 6,
    "heart": 7,
    "hip_left": 8,
    "hip_right": 9,
    "humerus_left": 10,
    "humerus_right": 11,
    "kidney_left": 12,
    "kidney_right": 13,
    "liver": 14,
    "lung": 15,
    "cervical_spine": 16,
    "thoracic_spine": 17,
    "lumbar_spine": 18,
    "sacrum": 19,
    "scapula_left": 20,
    "scapula_right": 21,
    "spleen": 22,
    "stomach": 23,
    "trachea": 24,
    "brain": 25,
    "skull": 26
}
    sorted_labels = sorted(labels, key=lambda k: labels[k])
    npz_files = [f for f in os.listdir(intermediate_npz_dir) if f.startswith("la_") and f.endswith(".npz")]

    for npz_file in npz_files:
        subject_id = npz_file.split("_")[1].split(".")[0]
        npz_path = os.path.join(intermediate_npz_dir, npz_file)
        ref_seg_path = os.path.join(intermediate_npz_dir, f"la_{subject_id}.nii.gz")

        if not os.path.exists(ref_seg_path):
            print(f" Reference segmentation not found for {subject_id}, skipping...")
            continue

        subject_output_folder = os.path.join(output_dir, subject_id)
        os.makedirs(subject_output_folder, exist_ok=True)

        data = np.load(npz_path)
        probabilities = data['probabilities']
        if probabilities.ndim == 4:
            probabilities = np.squeeze(probabilities, axis=1)

        affine_ref = nib.load(ref_seg_path).affine

        for idx, label_name in enumerate(sorted_labels):
            prob_map = probabilities[idx]
            binary_mask = (prob_map > threshold).astype(np.uint8)
            binary_mask = transform_mask(binary_mask)
            if binary_mask.ndim == 2:
                binary_mask = binary_mask[..., np.newaxis]

            seg_img = nib.Nifti1Image(binary_mask, affine_ref)
            seg_img = nib.as_closest_canonical(seg_img)

            out_path = os.path.join(subject_output_folder, f"{label_name}.nii.gz")
            nib.save(seg_img, out_path)
            

    print(f"All subjects processed. Final outputs in: {output_dir}")
    shutil.rmtree(intermediate_npz_dir)

def main():
    import argparse

    parser = argparse.ArgumentParser(description="TOPOS: Target Organ Prediction CLI")
    parser.add_argument("-i", "--input", required=True, help="Path to folder containing input .nii.gz images")
    parser.add_argument("-o", "--output", required=True, help="Path to output folder for predicted segmentations")
    parser.add_argument("--config", default="2d", help="nnU-Net model configuration (default: 2d)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for segmentation (default: 0.5)")

    args = parser.parse_args()

    predict(
        input_images_dir=args.input,
        output_dir=args.output,
        model_config=args.config,
        threshold=args.threshold
    )
