"""
interface.py
------------
Command-line chatbot using Hugging Face models.
Now outputs clean Q&A style answers like:
User: What is the capital of France?
Bot: The capital of France is Paris.
"""

import argparse
from model_loader import load_model_and_tokenizer
from chat_memory import ChatMemory


def generate_reply(pipe, question: str, is_seq2seq: bool = False) -> str:
    """
    Generate a reply from the model.
    Uses few-shot Q&A examples for accuracy.
    """

    # Few-shot prompt to guide the model
    prompt = (
        f"Question: {question}\nAnswer:"
    )

    if is_seq2seq:
        outputs = pipe(
            prompt,
            max_length=64,
            num_beams=4,
            do_sample=False,
            truncation=True,
        )
    else:
        outputs = pipe(
            prompt,
            max_length=64,
            do_sample=False,
            temperature=None,
            truncation=True,
        )

    text = outputs[0]["generated_text"]
    # Extract only the part after "Answer:"
    reply = text.split("Answer:")[-1].strip()
    return reply


def chat_loop(model_name: str, dtype: str, memory_size: int):
    bundle = load_model_and_tokenizer(model_name=model_name, dtype=dtype)
    pipe = bundle["pipe"]

    is_seq2seq = pipe.task == "text2text-generation"
    memory = ChatMemory(max_turns=memory_size)

    print("🤖 Chatbot ready! Type '/exit' to quit, '/clear' to reset.\n")

    while True:
        user_input = input("User: ").strip()
        if not user_input:
            continue

        if user_input.lower() in {"/exit", "/quit"}:
            print("Exiting chatbot. Goodbye!")
            break
        elif user_input.lower() == "/clear":
            memory.clear()
            print("Memory cleared.")
            continue
        elif user_input.lower() == "/help":
            print("Commands: /exit, /clear, /help")
            continue

        memory.add_turn("User", user_input)
        bot_reply = generate_reply(pipe, user_input, is_seq2seq=is_seq2seq)
        memory.add_turn("Bot", bot_reply)

        print(f"Bot: {bot_reply}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple CLI chatbot")
    parser.add_argument("--model", type=str, default="google/flan-t5-large", help="Model name")
    parser.add_argument("--dtype", type=str, default=None, help="torch dtype (fp16, bf16, fp32)")
    parser.add_argument("--memory", type=int, default=4, help="Number of turns to keep in memory")
    args = parser.parse_args()

    chat_loop(model_name=args.model, dtype=args.dtype, memory_size=args.memory)
