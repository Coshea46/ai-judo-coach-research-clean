import torch
import torch.nn as nn


class JudoClipClassifierModel(nn.Module):
    """
    LSTM classifier producing one throw-attempt logit per clip.

    Expected input:
        [batch_size, sequence_length, num_features_per_frame]

    Output:
        [batch_size]

    The architecture consists of an LSTM followed by a classification head.
    """

    def __init__(
        self,
        num_features_per_frame: int,
        num_hidden_state_features_lstm: int,
        num_layers: int,
        classifier_hidden_size: int,
        dropout_rate: float,
        bidirectional: bool,
    ) -> None:
        super().__init__()

        if num_features_per_frame <= 0:
            raise ValueError(
                "num_features_per_frame must be greater than zero"
            )

        if num_hidden_state_features_lstm <= 0:
            raise ValueError(
                "num_hidden_state_features_lstm must be greater than zero"
            )

        if num_layers <= 0:
            raise ValueError(
                "num_layers must be greater than zero"
            )

        if classifier_hidden_size <= 0:
            raise ValueError(
                "classifier_hidden_size must be greater than zero"
            )

        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError(
                "dropout_rate must be between 0.0 and 1.0"
            )

        self.num_layers = num_layers
        self.hidden_size = num_hidden_state_features_lstm
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=num_features_per_frame,
            hidden_size=num_hidden_state_features_lstm,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        classifier_input_size = (
            num_hidden_state_features_lstm * self.num_directions
        )

        self.fc1 = nn.Linear(
            classifier_input_size,
            classifier_hidden_size,
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        self.fc_out = nn.Linear(
            classifier_hidden_size,
            1,
        )

    def forward(
        self,
        input_sequence: torch.Tensor,
    ) -> torch.Tensor:
        """Produce one raw throw-attempt logit for each input clip."""

        _, (hidden_state, _) = self.lstm(input_sequence)

        if self.bidirectional:
            final_forward_state = hidden_state[-2]
            final_backward_state = hidden_state[-1]

            clip_representation = torch.cat(
                (final_forward_state, final_backward_state),
                dim=1,
            )
        else:
            clip_representation = hidden_state[-1]

        classifier_output = self.fc1(clip_representation)
        classifier_output = self.relu(classifier_output)
        classifier_output = self.dropout(classifier_output)
        logits = self.fc_out(classifier_output)

        return logits.squeeze(-1)
