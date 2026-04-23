"""Model building functions for knowledge distillation."""

from training_ssondo.utils.student_models.MobileNetV3.model import (
    get_model as get_mobilenet,
)
from training_ssondo.utils.student_models.dymn.model import get_model as get_dymn
from training_ssondo.utils.student_models.ERes2Net.model import ERes2Net
from training_ssondo.utils.student_models.model_utils import (
    LinearClassifer,
    MLPClassifer,
    RNNClassifer,
    AttentionRNNClassifer,
    ModelWrapper,
)


def build_student_model(conf: dict):
    """
    Build the complete student model with backbone and classification head.

    Parameters
    ----------
    conf : dict
        Configuration dictionary containing model architecture settings.

    Returns
    -------
    ModelWrapper
        Complete student model ready for training.
    """
    print("=" * 80)
    print("BUILDING STUDENT MODEL")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. Build Backbone Network
    # -------------------------------------------------------------------------
    print("\n[1/2] Building backbone network...")
    model_name = conf["student_model"]["model_name"]
    print(f"  Model: {model_name}")

    # MobileNet variants (excluding DyMN)
    if "mn" in model_name and "dy" not in model_name:
        print("  Architecture: MobileNetV3")
        model = get_mobilenet(
            pretrained_name=conf["student_model"]["pretrained_name"],
            width_mult=conf["student_model"]["width_mult"],
            reduced_tail=conf["student_model"]["reduced_tail"],
            dilated=conf["student_model"]["dilated"],
            strides=conf["student_model"]["strides"],
            relu_only=conf["student_model"]["relu_only"],
            input_dim_f=conf["student_model"]["input_dim_f"],
            input_dim_t=conf["student_model"]["input_dim_t"],
            se_dims=conf["student_model"]["se_dims"],
            se_agg=conf["student_model"]["se_agg"],
            se_r=conf["student_model"]["se_r"],
        )

    # Dynamic MobileNet (DyMN)
    elif "dy" in model_name:
        print("  Architecture: DyMN (Dynamic MobileNet)")
        model = get_dymn(
            pretrained_name=conf["student_model"]["pretrained_name"],
            width_mult=conf["student_model"]["width_mult"],
            strides=conf["student_model"]["strides"],
            pretrain_final_temp=conf["student_model"]["pretrain_final_temp"],
        )

    # ERes2Net
    else:
        print("  Architecture: ERes2Net")
        model = ERes2Net(
            m_channels=conf["student_model"]["m_channels"],
            feat_dim=conf["student_model"]["feat_dim"],
            num_blocks=conf["student_model"]["num_blocks"],
            pooling_func=conf["student_model"]["pooling_func"],
            add_layer=conf["student_model"]["add_layer"],
        )

    print(f"  ✓ Backbone built - Embedding size: {model.emb_size}")

    # -------------------------------------------------------------------------
    # 2. Build Classification Head
    # -------------------------------------------------------------------------
    print("\n[2/2] Building classification head...")
    head_type = conf["classification_head"]["head_type"]
    print(f"  Head type: {head_type}")

    # MLP Head
    if head_type == "mlp":
        try:
            hidden_features_size = model.last_channel
        except AttributeError:
            hidden_features_size = conf["classification_head"]["hidden_features_size"]

        class_head = MLPClassifer(
            emb_size=model.emb_size,
            n_classes=conf["classification_head"]["n_classes"],
            hidden_features_size=hidden_features_size,
            pooling=conf["classification_head"]["pooling"],
            activation_att=conf["classification_head"]["activation_att"],
            last_activation=conf["classification_head"]["last_activation"],
        )

    # RNN-based Heads (LSTM, GRU, RNN)
    elif head_type in ["lstm", "gru", "rnn"]:
        class_head = RNNClassifer(
            rnn_type=head_type,
            emb_size=model.emb_size,
            hidden_size=conf["classification_head"]["hidden_size"],
            n_classes=conf["classification_head"]["n_classes"],
            num_layers=conf["classification_head"]["num_layers"],
            bidirectional=conf["classification_head"]["bidirectional"],
            n_last_elements=conf["classification_head"]["n_last_elements"],
            last_activation=conf["classification_head"]["last_activation"],
        )

    # Attention RNN Heads
    elif head_type in ["attention_lstm", "attention_gru", "attention_rnn"]:
        rnn_type = head_type.split("_")[1]  # Extract lstm/gru/rnn
        class_head = AttentionRNNClassifer(
            rnn_type=rnn_type,
            emb_size=model.emb_size,
            hidden_size=conf["classification_head"]["hidden_size"],
            n_classes=conf["classification_head"]["n_classes"],
            num_layers=conf["classification_head"]["num_layers"],
            bidirectional=conf["classification_head"]["bidirectional"],
            n_last_elements=conf["classification_head"]["n_last_elements"],
            last_activation=conf["classification_head"]["last_activation"],
        )

    # Linear Head (default)
    else:
        class_head = LinearClassifer(
            emb_size=model.emb_size,
            n_classes=conf["classification_head"]["n_classes"],
            pooling=conf["classification_head"]["pooling"],
            activation_att=conf["classification_head"]["activation_att"],
            last_activation=conf["classification_head"]["last_activation"],
        )

    print(
        f"  ✓ Classification head built - Output classes: {conf['classification_head']['n_classes']}"
    )

    # -------------------------------------------------------------------------
    # 3. Combine Backbone + Head
    # -------------------------------------------------------------------------
    print("\nCombining backbone and classification head...")
    student_model = ModelWrapper(model=model, classification_head=class_head)
    print("  ✓ Complete student model assembled")

    print("\n" + "=" * 80)
    print("MODEL BUILDING COMPLETE")
    print("=" * 80 + "\n")

    return student_model
