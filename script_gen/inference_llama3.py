"""
Inference script for fine-tuned Llama-3 models with LoRA
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


class Llama3InferenceModel:
    def __init__(self, model_path, base_model="meta-llama/Llama-2-7b-hf", device=None):
        """
        Load a fine-tuned Llama-3 model with LoRA for inference

        Args:
            model_path: Path to the LoRA adapter directory
            base_model: Base model name
            device: Device to run inference on (cuda or cpu)
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading base model {base_model}...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        print(f"Loading LoRA adapter from {model_path}...")
        self.model = PeftModel.from_pretrained(self.base_model, model_path)
        self.model = self.model.to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        print(f"Model loaded on {self.device}")

    def generate(
        self,
        instruction,
        input_text,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
    ):
        """
        Generate output from the model

        Args:
            instruction: The instruction prompt
            input_text: The input text/content
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Temperature for generation (0 = greedy)
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling

        Returns:
            Generated text
        """
        # Format prompt
        prompt = f"{instruction} {input_text}\n\nOutput:\n"

        # Tokenize
        inputs = self.tokenizer.encode(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode
        full_output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Extract only the generated part (remove prompt)
        generated_text = full_output[len(prompt):]

        return generated_text.strip()


def main():
    parser = argparse.ArgumentParser(description="Run inference on fine-tuned Llama-3 models")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the LoRA adapter (e.g., /path/to/llama_model/quote_with_contents/final_model)"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Base model name (default: meta-llama/Llama-3.1-8B-Instruct)"
    )
    parser.add_argument("--instruction", type=str, required=True, help="Instruction prompt")
    parser.add_argument("--input", type=str, required=True, help="Input text/content")
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate (default: 256)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation (default: 0.7)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Nucleus sampling parameter (default: 0.95)"
    )
    parser.add_argument(
        "--no_sample",
        action="store_true",
        help="Use greedy decoding instead of sampling"
    )

    args = parser.parse_args()

    # Load model
    model = Llama3InferenceModel(args.model_path, args.base_model)

    # Run inference
    print(f"\n{'='*80}")
    print(f"Model: {args.model_path}")
    print(f"Base Model: {args.base_model}")
    print(f"{'='*80}")
    print(f"Instruction: {args.instruction}")
    print(f"Input: {args.input[:100]}..." if len(args.input) > 100 else f"Input: {args.input}")
    print(f"{'='*80}")

    output = model.generate(
        args.instruction,
        args.input,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=not args.no_sample,
    )

    print(f"Output:\n{output}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
