"""
Interactive inference example for fine-tuned Llama-3 models with LoRA
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_llama3_lora(adapter_path, base_model="meta-llama/Llama-2-7b-hf"):
    """Load Llama-3 with LoRA adapter"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading base model: {base_model}")
    base_model_obj = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model_obj, adapter_path)
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    return model, tokenizer, device


def generate(model, tokenizer, device, instruction, input_text, max_tokens=256):
    """Generate output"""
    prompt = f"{instruction} {input_text}\n\nOutput:\n"

    inputs = tokenizer.encode(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    generated_text = full_output[len(prompt):]
    return generated_text.strip()


# Example 1: quote_with_contents
print("\n" + "="*80)
print("EXAMPLE 1: Llama-3 with quote_with_contents adapter")
print("="*80)

model_wc, tokenizer_wc, device_wc = load_llama3_lora(
    "/mnt/Datasets/NoteVLM/src/llama_model/quote_with_contents/final_model"
)

instruction_wc = "You are a helpful assistant. Generate an engaging introduction of the content with quotation based on the following input."
input_text_wc = """
Organic farming emphasizes soil health and biodiversity. Studies show that neonicotinoid pesticides harm bee populations
and weaken immune systems. France banned these pesticides in 2020 after discovering their harmful effects.
Sustainable farming practices offer alternatives to chemical-based agriculture.
"""

print("\nGenerating output...")
output_wc = generate(model_wc, tokenizer_wc, device_wc, instruction_wc, input_text_wc)

print(f"\nInstruction:\n{instruction_wc}")
print(f"\nInput:\n{input_text_wc.strip()}")
print(f"\nGenerated Output:\n{output_wc}")

# Example 2: quote_only
print("\n" + "="*80)
print("EXAMPLE 2: Llama-3 with quote_only adapter")
print("="*80)

model_qo, tokenizer_qo, device_qo = load_llama3_lora(
    "/mnt/Datasets/NoteVLM/src/llama_model/quote_only/final_model"
)

instruction_qo = "You are a helpful assistant. Generate an engaging introduction of the content with quotation based on the following input."
input_text_qo = """
Technology advances rapidly. Robots can now assist in classrooms. Students report higher engagement with interactive learning.
Some worry about replacing human teachers, but robots work best alongside educators.
"""

print("\nGenerating output...")
output_qo = generate(model_qo, tokenizer_qo, device_qo, instruction_qo, input_text_qo)

print(f"\nInstruction:\n{instruction_qo}")
print(f"\nInput:\n{input_text_qo.strip()}")
print(f"\nGenerated Output:\n{output_qo}")

print("\n" + "="*80 + "\n")
