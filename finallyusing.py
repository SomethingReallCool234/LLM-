from prev import GPTModel, generate
import tiktoken
import torch
from gpt import download_and_load_gpt2
from prev import GPTModel, load_weights_into_gpt

settings, params = download_and_load_gpt2("124M", "./gpt2_models")

GPT_CONFIG = {
    "vocab_size": settings["n_vocab"],
    "context_length": settings["n_ctx"],
    "emb_dim": settings["n_embd"],
    "n_heads": settings["n_head"],
    "n_layers": settings["n_layer"],
    "drop_rate": 0.1,
    "qkv_bias": True
}

tokenizer = tiktoken.get_encoding("gpt2")

model = GPTModel(GPT_CONFIG)
model.load_state_dict(torch.load("chatbot_finetuned_gpt2.pth", map_location="cpu"))
model.eval()

print("Custom LLM Chatbot ready! Type 'exit' or 'quit' to stop.\n")

while True:
    user_input = input("You: ")
    if user_input.strip().lower() in {"exit", "quit"}:
        break

    # Build the role-token prompt
    prompt = f"<|user|>\n{user_input}\n<|assistant|>\n"
    encoded = tokenizer.encode(prompt)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    # Generate output (stop at <|end|>)
    output_ids = generate(
        model,
        encoded_tensor,
        max_new_tokens=64,
        context_size=GPT_CONFIG["context_length"],
        temperature=0.7,
        top_k=20,
        eos_id=tokenizer.encode("<|end|>")[0]
    )

    response = tokenizer.decode(output_ids[0].tolist())
    print("Bot:", response)
