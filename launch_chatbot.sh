#!/bin/bash
"""
Launch Script for Chatbot Interfaces

This script provides easy commands to launch either the Gradio or Streamlit
chatbot interface for testing your fine-tuned models.
"""

# Default values
MODEL_PATH="./llama32_mixed_finetuned"
INTERFACE="gradio"
PORT=7860
SHARE=false

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Launch chatbot interface for testing fine-tuned Llama models"
    echo ""
    echo "Options:"
    echo "  -i, --interface TYPE    Interface type: 'gradio' or 'streamlit' (default: gradio)"
    echo "  -m, --model PATH        Path to fine-tuned model (default: ./llama32_mixed_finetuned)"
    echo "  -p, --port PORT         Port number (default: 7860)"
    echo "  -s, --share             Create public sharing link (Gradio only)"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Launch Gradio interface with default model"
    echo "  $0 -i streamlit                      # Launch Streamlit interface"
    echo "  $0 -m ./my_model -p 8080             # Use custom model and port"
    echo "  $0 -s                                # Launch with public sharing"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--interface)
            INTERFACE="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -s|--share)
            SHARE=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate interface type
if [[ "$INTERFACE" != "gradio" && "$INTERFACE" != "streamlit" ]]; then
    echo "Error: Interface must be 'gradio' or 'streamlit'"
    exit 1
fi

# Check if model path exists (for local models)
if [[ "$MODEL_PATH" != unsloth/* && ! -d "$MODEL_PATH" ]]; then
    echo "Warning: Model path '$MODEL_PATH' does not exist"
    echo "Will attempt to use base model as fallback"
fi

# Activate UV environment
echo "🔧 Activating UV environment..."
source .venv/bin/activate || {
    echo "❌ Error: Could not activate UV environment"
    echo "Make sure you're in the project directory and the environment exists"
    exit 1
}

# Launch the appropriate interface
echo "🚀 Launching $INTERFACE chatbot interface..."
echo "📂 Model: $MODEL_PATH"
echo "🌐 Port: $PORT"

if [[ "$INTERFACE" == "gradio" ]]; then
    if [[ "$SHARE" == "true" ]]; then
        echo "🔗 Creating public sharing link..."
        python gradio_chatbot.py --model_path "$MODEL_PATH" --port "$PORT" --share
    else
        python gradio_chatbot.py --model_path "$MODEL_PATH" --port "$PORT"
    fi
elif [[ "$INTERFACE" == "streamlit" ]]; then
    if [[ "$SHARE" == "true" ]]; then
        echo "⚠️  Note: Public sharing not supported with Streamlit"
    fi
    
    # Set Streamlit config
    export STREAMLIT_SERVER_PORT="$PORT"
    export STREAMLIT_SERVER_ADDRESS="0.0.0.0"
    
    streamlit run streamlit_chatbot.py
fi