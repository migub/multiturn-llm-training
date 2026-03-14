import torch
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    bias="none",
)

tokenizer.pad_token = tokenizer.eos_token

# Prepare a simple test input
messages = [{"role": "user", "content": "Say hello in exactly 3 words."}]
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

FastLanguageModel.for_inference(model)

# Test 1: With LoRA
print("Test 1: LoRA ENABLED")
model.enable_adapter_layers()
out1 = model.generate(**inputs, max_new_tokens=20, temperature=0.01, do_sample=True)
print(tokenizer.decode(out1[0], skip_special_tokens=True))

# Test 2: Without LoRA
print("\nTest 2: LoRA DISABLED")
model.disable_adapter_layers()
out2 = model.generate(**inputs, max_new_tokens=20, temperature=0.01, do_sample=True)
print(tokenizer.decode(out2[0], skip_special_tokens=True))

# Test 3: Back with LoRA
print("\nTest 3: LoRA RE-ENABLED")
model.enable_adapter_layers()
out3 = model.generate(**inputs, max_new_tokens=20, temperature=0.01, do_sample=True)
print(tokenizer.decode(out3[0], skip_special_tokens=True))

print("\n--- Results ---")
print(f"Test 1 == Test 3 (both LoRA on): {torch.equal(out1, out3)}")
print(f"Test 1 == Test 2 (LoRA on vs off): {torch.equal(out1, out2)}")
print("No crash = disable/enable works on quantized model!")