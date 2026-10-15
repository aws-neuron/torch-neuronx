import argparse
import logging
import time

import torch
from transformers import AutoTokenizer, WhisperForConditionalGeneration

import torch_neuronx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run Whisper on Neuron")
    parser.add_argument(
        "--model", type=str, default="openai/whisper-tiny", help="Whisper model name"
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(42)

    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float32, attn_implementation="eager"
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    num_mel_bins = model.config.num_mel_bins
    input_features = torch.randn(args.batch_size, num_mel_bins, 3000, dtype=torch.float32)
    gen_kwargs = {
        "max_new_tokens": 64,
        "do_sample": False,
        "cache_implementation": "static",
        "eos_token_id": -1,
    }

    # Run once to establish shapes before compile
    with torch.no_grad():
        _ = model.generate(input_features=input_features, **gen_kwargs)

    model.forward = torch.compile(model.forward, backend="neuron", fullgraph=True)

    # Warmup
    warmup_start = time.time()
    with torch.no_grad():
        output = model.generate(input_features=input_features, **gen_kwargs)
    warmup_time = time.time() - warmup_start

    # Run
    run_start = time.time()
    with torch.no_grad():
        output = model.generate(input_features=input_features, **gen_kwargs)
    run_time = time.time() - run_start

    logger.info(f"Warmup: {warmup_time:.2f}s, Run: {run_time:.4f}s")
    logger.info(f"Output: {tokenizer.batch_decode(output, skip_special_tokens=True)}")


if __name__ == "__main__":
    main()
