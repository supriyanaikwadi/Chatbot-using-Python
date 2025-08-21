# Simple CLI Chatbot

This is a minimal command-line chatbot built using Hugging Face
Transformers.\
It supports both causal language models (GPT-2, DialoGPT) and
seq2seq models (Flan-T5, BART, etc.).

By default, it runs with `google/flan-t5-large`, which gives much
better factual answers than GPT-2.

------------------------------------------------------------------------

## Installation

1.  Clone or download this project.
2.  Install Python dependencies:

``` bash
pip install torch transformers
```

(Optionally, install `accelerate` for faster inference if you have a
GPU.)

------------------------------------------------------------------------

##  Usage



Run the chatbot with:

python model_loader.py

python chat_memory.py

python interface.py --model google/flan-t5-large


Example session:

    Chatbot ready! Type '/exit' to quit, '/clear' to reset.

    User: What is the capital of India?
    Bot: The capital of India is New Delhi.

    User: What is the capital of France?
    Bot: The capital of France is Paris.

    User: /exit
    Exiting chatbot. Goodbye!

------------------------------------------------------------------------

## Options

You can customize parameters:

``` bash
python interface.py --model google/flan-t5-base --memory 6 --dtype fp32
```

-   `--model` → Hugging Face model name (e.g., `distilgpt2`,
    `microsoft/DialoGPT-medium`, `google/flan-t5-large`)\
-   `--memory` → Number of dialogue turns to keep in memory\
-   `--dtype` → Torch precision (`fp16`, `bf16`, `fp32`)

------------------------------------------------------------------------

## Notes

-   Flan-T5 models (`flan-t5-base`, `flan-t5-large`, etc.) are
    recommended for Q&A and instruction-following.\
-   GPT-2 / DialoGPT models will run, but they often produce random
    or unrealistic answers.\
-   On CPU, `flan-t5-small` or `flan-t5-base` are faster;
    `flan-t5-large` is slower but more accurate.\
-   If you have a GPU, larger models (`flan-t5-xl`,
    `Mistral-7B-Instruct`, `Falcon-7B-Instruct`) will give better
    results.

------------------------------------------------------------------------

## Example Commands

``` bash
# Small and fast (CPU-friendly)
python interface.py --model google/flan-t5-small

# Better accuracy
python interface.py --model google/flan-t5-large

# Using DialoGPT (chat-style but weaker knowledge)
python interface.py --model microsoft/DialoGPT-medium
```

