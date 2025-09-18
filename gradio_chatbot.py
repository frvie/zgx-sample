#!/usr/bin/env python3
"""
Gradio Chatbot Interface for Testing Fine-tuned Llama Models

This script creates an interactive web interface using Gradio to test your
fine-tuned Llama-3.2-1B models with a chat-like experience.
"""

import gradio as gr
import torch
from unsloth import FastLanguageModel
import argparse
import os
from pathlib import Path
import time


class LlamaChatbot:
    def __init__(self, model_path="./llama32_mixed_finetuned"):
        """Initialize the chatbot with a fine-tuned model."""
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.load_model()
    
    def load_model(self):
        """Load the fine-tuned model."""
        print(f"🤖 Loading model from: {self.model_path}")
        
        try:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_path,
                max_seq_length=2048,
                dtype=torch.bfloat16,
                load_in_4bit=True,
            )
            
            # Enable inference mode
            FastLanguageModel.for_inference(self.model)
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            # Fallback to base model
            print("🔄 Falling back to base model...")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name="unsloth/Llama-3.2-1B-Instruct",
                max_seq_length=2048,
                dtype=torch.bfloat16,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(self.model)
    
    def format_prompt(self, instruction, input_text=""):
        """Format the prompt in the training format."""
        if input_text.strip():
            return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            return f"### Instruction:\n{instruction}\n\n### Response:\n"
    
    def generate_response(self, message, history, max_tokens=200, temperature=0.7, top_p=0.9):
        """Generate a response to the user's message."""
        # Format the prompt
        prompt = self.format_prompt(message)
        
        # Tokenize
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        
        # Generate
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )
        
        generation_time = time.time() - start_time
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = response[len(prompt):].strip()
        
        # Add generation info
        info = f"\n\n*Generated in {generation_time:.2f}s*"
        
        return generated_text + info


def create_chatbot_interface():
    """Create the Gradio chatbot interface."""
    
    def get_available_models():
        """Get list of available fine-tuned models."""
        models = []
        base_dir = Path(".")
        
        # Look for model directories
        for path in base_dir.glob("*finetuned*"):
            if path.is_dir() and (path / "adapter_config.json").exists():
                models.append(str(path))
        
        # Add base model option
        models.append("unsloth/Llama-3.2-1B-Instruct")
        
        return models if models else ["./llama32_mixed_finetuned"]
    
    # Initialize chatbot
    available_models = get_available_models()
    chatbot = LlamaChatbot(available_models[0])
    
    def load_model(model_path):
        """Load a different model."""
        nonlocal chatbot
        chatbot = LlamaChatbot(model_path)
        return f"✅ Loaded model: {model_path}"
    
    def respond(message, history, max_tokens, temperature, top_p):
        """Generate response and update chat history."""
        if not message.strip():
            return history, ""
        
        # Generate response
        response = chatbot.generate_response(message, history, max_tokens, temperature, top_p)
        
        # Update history
        history.append((message, response))
        return history, ""
    
    # Create the interface
    with gr.Blocks(title="🦙 Llama Chatbot", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🦙 Fine-tuned Llama Chatbot")
        gr.Markdown("Chat with your fine-tuned Llama-3.2-1B model!")
        
        with gr.Row():
            with gr.Column(scale=3):
                # Chat interface
                chatbot_ui = gr.Chatbot(
                    label="Chat with Llama",
                    height=500,
                    show_label=True
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="Your message",
                        placeholder="Type your message here...",
                        scale=4
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                
                # Example prompts
                gr.Examples(
                    examples=[
                        "Write a short story about a robot learning to paint.",
                        "Explain quantum computing in simple terms.",
                        "Write a Python function to calculate factorial.",
                        "What are the benefits of renewable energy?",
                        "How do I bake a chocolate cake?",
                        "Explain the difference between machine learning and AI.",
                    ],
                    inputs=msg,
                    label="Example prompts"
                )
            
            with gr.Column(scale=1):
                # Model selection
                model_dropdown = gr.Dropdown(
                    choices=available_models,
                    value=available_models[0],
                    label="Select Model",
                    interactive=True
                )
                
                load_btn = gr.Button("Load Model", variant="secondary")
                model_status = gr.Textbox(label="Model Status", interactive=False)
                
                # Generation parameters
                gr.Markdown("### Generation Settings")
                
                max_tokens = gr.Slider(
                    minimum=50,
                    maximum=500,
                    value=200,
                    step=10,
                    label="Max Tokens"
                )
                
                temperature = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.7,
                    step=0.1,
                    label="Temperature"
                )
                
                top_p = gr.Slider(
                    minimum=0.1,
                    maximum=1.0,
                    value=0.9,
                    step=0.05,
                    label="Top P"
                )
                
                # Clear chat
                clear_btn = gr.Button("Clear Chat", variant="stop")
        
        # Event handlers
        send_btn.click(
            respond,
            inputs=[msg, chatbot_ui, max_tokens, temperature, top_p],
            outputs=[chatbot_ui, msg]
        )
        
        msg.submit(
            respond,
            inputs=[msg, chatbot_ui, max_tokens, temperature, top_p],
            outputs=[chatbot_ui, msg]
        )
        
        load_btn.click(
            load_model,
            inputs=[model_dropdown],
            outputs=[model_status]
        )
        
        clear_btn.click(
            lambda: ([], ""),
            outputs=[chatbot_ui, msg]
        )
    
    return interface


def main():
    parser = argparse.ArgumentParser(description="Launch Gradio Chatbot Interface")
    parser.add_argument("--model_path", type=str, default="./llama32_mixed_finetuned",
                       help="Path to fine-tuned model")
    parser.add_argument("--share", action="store_true",
                       help="Create public sharing link")
    parser.add_argument("--port", type=int, default=7860,
                       help="Port to run the interface on")
    
    args = parser.parse_args()
    
    print("🚀 Starting Gradio Chatbot Interface...")
    print(f"📂 Model path: {args.model_path}")
    
    # Create and launch interface
    interface = create_chatbot_interface()
    
    interface.launch(
        share=args.share,
        server_port=args.port,
        server_name="0.0.0.0"
    )


if __name__ == "__main__":
    main()