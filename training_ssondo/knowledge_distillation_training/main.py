"""Refactored training pipeline - orchestrates all components."""

# Standard library imports
import argparse
import os
import warnings
from pprint import pprint

# Local package imports - refactored modules
from .data_pipeline import setup_data_pipeline
from .model import build_student_model
from .training_components import setup_optimizer, setup_learning_rate_scheduler, setup_loss_fct, configure_trainer, get_checkpoint_callback

# Local package imports - existing modules
from .config import conf, common_parameters
from .system import KnowledgeDistillationSystem
from .utils import merge_dicts, set_random_seeds, save_config_file, save_best_models


def main(conf: dict) -> None:
    """
    Main orchestration function for the training pipeline.

    This function coordinates all training components:
    1. Sets random seeds for reproducibility
    2. Sets up data pipeline (preprocessing, datasets, dataloaders)
    3. Builds student model (backbone + classification head)
    4. Configures training components (optimizer, scheduler, losses)
    5. Configures PyTorch Lightning trainer
    6. Initializes training system
    7. Runs training
    8. Saves results

    Parameters
    ----------
    conf : dict
        Configuration dictionary containing all experiment settings.
    """
    print(f"Experiment: {conf['conf_id']}, Job ID: {conf['job_id']}")
    
    # Step 1: Reproducibility Setup
    generator = set_random_seeds(conf["seed"])

    # Step 2: Data Pipeline
    train_loader, val_loader = setup_data_pipeline(conf, generator)

    # Step 3: Model Building
    student_model = build_student_model(conf)

    # Step 4: Training Components
    optimizer = setup_optimizer(conf, student_model)
    scheduler = setup_learning_rate_scheduler(conf, optimizer, train_loader)
    pred_loss_func, kd_loss_func = setup_loss_fct(conf)

    # Step 5: Logging and Checkpointing Setup
    log_dir = os.path.join(conf["exp_dir"], conf["conf_id"], conf["job_id"])
    save_config_file(conf, log_dir)

    # Step 6: Trainer Configuration
    trainer, callbacks = configure_trainer(conf, log_dir)

    # Step 7: Initialize Training System
    system = KnowledgeDistillationSystem(
        config=conf,
        student_model=student_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        pred_loss_func=pred_loss_func,
        kd_loss_func=kd_loss_func,
    )
    # Step 8: Start Training
    checkpoint_path = conf["checkpoint_path"]
    if checkpoint_path is None:
        print("Training from scratch...")
        trainer.fit(system)
    else:
        print(f"Resuming from checkpoint: {checkpoint_path}")
        trainer.fit(system, ckpt_path=checkpoint_path)

    # Step 9: Save Results
    checkpoint_callback = get_checkpoint_callback(callbacks)
    save_best_models(checkpoint_callback, log_dir)
    print(f"TRAINING COMPLETE. Results saved to: {log_dir}\n")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Train knowledge distillation model with refactored pipeline")
    parser.add_argument("--conf_id", required=True, help="Configuration ID to select the right config from config.py")
    args = parser.parse_args()
    args = vars(args)
    # Merge configurations
    experiment_conf = merge_dicts(common_parameters, conf[args["conf_id"]])
    experiment_conf = {**experiment_conf, **args}
    pprint(experiment_conf)
    main(experiment_conf)