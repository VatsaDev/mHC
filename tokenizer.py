# Most likely will use Qwen 128k tokenizer in the big one, small model 4096 helps

from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

# BPE tokenizer

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

# 4k smol

trainer = trainers.BpeTrainer(
    vocab_size=1024,
    special_tokens=["<|endoftext|>", "<|padding|>"],
    min_frequency=2
)

# json

files = ["data/synth_text/synth_part_000.txt"]
tokenizer.train(files, trainer)
tokenizer.save("tokenizers/mhc_1k.json")

# full

tokenizer = PreTrainedTokenizerFast(
    tokenizer_file="tokenizers/mhc_1k.json",
    model_max_length=4096  # Match this to your LLM's context length
)

tokenizer.pad_token = "<|padding|>"
tokenizer.eos_token = "<|endoftext|>"
tokenizer.bos_token = "<|endoftext|>"
tokenizer.unk_token = "<|endoftext|>" # Fallback

tokenizer.save_pretrained("tokenizers/mhc_1k")

tok = AutoTokenizer.from_pretrained("tokenizers/mhc_1k")

# quick check
text = "The winds were the Anemoi, powerful deities like Boreas (N), Zephyrus (W), Notus (S), and Eurus (E)"
encoded = tok(text, padding="max_length", truncation=True, max_length=200)
print("IDs:", encoded["input_ids"])
print("Tokens:", tok.convert_ids_to_tokens(encoded["input_ids"]))
print("Decoded:", tok.decode(encoded["input_ids"], skip_special_tokens=True))

