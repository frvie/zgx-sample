#!/usr/bin/env python3
"""
Streamlit Chatbot Interface for Testing Fine-tuned Llama Models

This script creates an interactive web application using Streamlit to test your
fine-tuned Llama-3.2-1B models with a modern chat interface.
"""

import streamlit as st
import torch
from unsloth import FastLanguageModel
import time
from pathlib import Path
import json


# Page configuration
st.set_page_config(
    page_title="🦙 Llama Chatbot",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_model(model_path):
    """Load the fine-tuned model (cached for performance)."""
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            dtype=torch.bfloat16,
            load_in_4bit=True,
        )
        
        # Enable inference mode
        FastLanguageModel.for_inference(model)
        return model, tokenizer, True
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        # Fallback to base model
        st.warning("Falling back to base model...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/Llama-3.2-1B-Instruct",
            max_seq_length=2048,
            dtype=torch.bfloat16,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        return model, tokenizer, False


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


def format_prompt(instruction, input_text=""):
    """Format the prompt in the training format."""
    if input_text.strip():
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    else:
        return f"### Instruction:\n{instruction}\n\n### Response:\n"


def generate_response(model, tokenizer, message, max_tokens, temperature, top_p):
    """Generate a response to the user's message."""
    # Format the prompt
    prompt = format_prompt(message)
    
    # Tokenize
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    
    # Generate
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )
    
    generation_time = time.time() - start_time
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text = response[len(prompt):].strip()
    
    return generated_text, generation_time


def main():
    """Main Streamlit application."""
    
    # Title and description
    st.title("🦙 Fine-tuned Llama Chatbot")
    st.markdown("Chat with your fine-tuned Llama-3.2-1B model!")
    
    # Sidebar for model selection and settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        available_models = get_available_models()
        selected_model = st.selectbox(
            "Select Model",
            available_models,
            index=0
        )
        
        # Load model
        if "current_model" not in st.session_state or st.session_state.current_model != selected_model:
            with st.spinner(f"Loading model: {selected_model}"):
                model, tokenizer, success = load_model(selected_model)
                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
                st.session_state.current_model = selected_model
                
                if success:
                    st.success(f"✅ Loaded: {selected_model}")
                else:
                    st.warning("⚠️ Using fallback model")
        
        st.divider()
        
        # Generation parameters
        st.subheader("🎛️ Generation Settings")
        
        max_tokens = st.slider(
            "Max Tokens",
            min_value=50,
            max_value=500,
            value=200,
            step=10,
            help="Maximum number of tokens to generate"
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Controls randomness (lower = more focused)"
        )
        
        top_p = st.slider(
            "Top P",
            min_value=0.1,
            max_value=1.0,
            value=0.9,
            step=0.05,
            help="Nucleus sampling parameter"
        )
        
        st.divider()
        
        # Model info
        if "model" in st.session_state:
            st.subheader("📊 Model Info")
            
            # Count parameters
            total_params = sum(p.numel() for p in st.session_state.model.parameters())
            trainable_params = sum(p.numel() for p in st.session_state.model.parameters() if p.requires_grad)
            
            st.metric("Total Parameters", f"{total_params:,}")
            st.metric("Trainable Parameters", f"{trainable_params:,}")
            
            # GPU memory
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                st.metric("GPU Memory", f"{memory_used:.1f}GB / {memory_total:.1f}GB")
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Display chat messages
        chat_container = st.container(height=500)
        
        with chat_container:
            for i, message in enumerate(st.session_state.messages):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    
                    # Show generation time for assistant messages
                    if message["role"] == "assistant" and "generation_time" in message:
                        st.caption(f"*Generated in {message['generation_time']:.2f}s*")
        
        # Chat input
        if prompt := st.chat_input("Type your message here..."):
            # Add user message to chat
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Generate response
            if "model" in st.session_state:
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response, gen_time = generate_response(
                            st.session_state.model,
                            st.session_state.tokenizer,
                            prompt,
                            max_tokens,
                            temperature,
                            top_p
                        )
                    
                    st.markdown(response)
                    st.caption(f"*Generated in {gen_time:.2f}s*")
                    
                    # Add assistant message to chat
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "generation_time": gen_time
                    })
                    
                    st.rerun()
    
    with col2:
        # Example prompts
        st.subheader("💡 Example Prompts")
        
        examples = [
            "Write a short story about a robot learning to paint.",
            "Explain quantum computing in simple terms.",
            "Write a Python function to calculate factorial.",
            "What are the benefits of renewable energy?",
            "How do I bake a chocolate cake?",
            "Explain the difference between machine learning and AI.",
        ]
        
        for i, example in enumerate(examples):
            if st.button(example, key=f"example_{i}", help="Click to use this prompt"):
                # Add the example as a user message
                st.session_state.messages.append({"role": "user", "content": example})
                st.rerun()
        
        st.divider()
        
        # Chat statistics
        if st.session_state.messages:
            st.subheader("📈 Chat Stats")
            user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
            assistant_messages = len([m for m in st.session_state.messages if m["role"] == "assistant"])
            
            st.metric("User Messages", user_messages)
            st.metric("Assistant Messages", assistant_messages)
            
            # Average generation time
            gen_times = [m.get("generation_time", 0) for m in st.session_state.messages if m["role"] == "assistant"]
            if gen_times:
                avg_time = sum(gen_times) / len(gen_times)
                st.metric("Avg Response Time", f"{avg_time:.2f}s")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        Built with ❤️ using Streamlit and Unsloth | 
        <a href='https://github.com/unslothai/unsloth' target='_blank'>Unsloth GitHub</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()