"""
uv pip install torchaudio
version used: 2.8.0+cpu
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torchaudio.models import Conformer

import torch_neuronx


class FakeASRDataset:
    """
    Generates random spectrograms + random token sequences
    """

    def __init__(
        self,
        n_mels=80,
        vocab_size=1000,
        min_audio_len=100,
        max_audio_len=400,
        min_text_len=10,
        max_text_len=50,
        seed=0,
    ):
        self.n_mels = n_mels
        self.vocab_size = vocab_size
        self.min_audio_len = min_audio_len
        self.max_audio_len = max_audio_len
        self.min_text_len = min_text_len
        self.max_text_len = max_text_len
        self.generator = torch.Generator().manual_seed(seed)

    def __iter__(self):
        while True:
            audio_len = 512  # NOTE seq len needs to be multiple of 512 for FA
            text_len = int(
                torch.randint(
                    self.min_text_len, self.max_text_len + 1, (1,), generator=self.generator
                )
            )

            audio_features = torch.randn(audio_len, self.n_mels, generator=self.generator)
            text_tokens = torch.randint(
                1, self.vocab_size + 1, (text_len,), generator=self.generator
            )  # 1..vocab_size

            yield {
                "audio_features": audio_features,
                "audio_length": audio_len,
                "text_tokens": text_tokens,
                "text_length": text_len,
            }


# --------------------------
# Conformer + CTC head
# --------------------------
class ConformerCTC(nn.Module):
    def __init__(
        self,
        n_mels=80,
        d_model=256,
        num_layers=2,
        num_heads=4,
        ffn_dim=256,
        depthwise_conv_kernel_size=15,
        vocab_size=1000,
        dropout=0.0,
    ):
        super().__init__()
        self.in_proj = nn.Linear(n_mels, d_model)
        self.encoder = Conformer(
            input_dim=d_model,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            depthwise_conv_kernel_size=depthwise_conv_kernel_size,
            dropout=dropout,
        )
        self.classifier = nn.Linear(d_model, vocab_size + 1)  # +1 for CTC blank

    def forward(self, x, x_lens):
        x = self.in_proj(x)  # (B, T, d_model)
        enc, out_lens = self.encoder(x, x_lens)  # (B, T, d_model), (B,)
        logits = self.classifier(enc)  # (B, T, V)
        return logits.transpose(0, 1), out_lens  # (T, B, V), (B,)


def train(args):
    device = "neuron"
    dataset = FakeASRDataset()
    model = ConformerCTC().to(device)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)

    for i, sample in enumerate(dataset):
        if i >= args.iterations:
            break

        # Prepare single-sample batch (B=1)
        x = sample["audio_features"].unsqueeze(0).to(device)  # (1, T, F)
        x_lens = torch.tensor([sample["audio_length"]], dtype=torch.long, device=device)
        y = sample["text_tokens"].to(device)
        y_lens = torch.tensor([sample["text_length"]], dtype=torch.long, device=device)

        # Forward
        logits, out_lens = model(x, x_lens)  # (T, 1, V), (1,)
        log_probs = functional.log_softmax(logits, dim=-1)
        loss = ctc_loss(log_probs, y, out_lens, y_lens)

        print(f"Iteration {i + 1}/{args.iterations} - Forward complete, Loss: {loss.item():.4f}")

        # Backward
        opt.zero_grad()
        loss.backward()
        print(f"Iteration {i + 1}/{args.iterations} - Backward complete")

        # Optimizer step
        opt.step()
        print(f"Iteration {i + 1}/{args.iterations} - Optimizer step complete\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=2, help="Number of training iterations")
    args = parser.parse_args()

    torch.manual_seed(0)
    train(args)
