import os
import time
import threading
from argparse import ArgumentParser


def parse_arguments():
    """Parse and return the command line arguments."""

    parser = ArgumentParser(description="Start the Ollama server with specified configurations.")
    parser.add_argument('--port', '-p', type=int, required=True, help='Port to run the server on')
    parser.add_argument('--model_path', '-mp', type=str, default='~/.ollama/models', help='Path to the models directory')
    parser.add_argument('--keep_alive', '-t', type=str, default='5m', help='The duration that models stay loaded in memory')

    return parser.parse_args()


def construct_command(host, port, model_path, keep_alive):
    """Construct and return the command to start the Ollama server."""

    return (
        # f"CUDA_VISIBLE_DEVICES=0 "
        f"OLLAMA_HOST={host}:{port} "
        f"OLLAMA_MODELS={model_path} "
        f"OLLAMA_KEEP_ALIVE={keep_alive} "
        f"OLLAMA_NUM_PARALLEL=64 "
        f"OLLAMA_FLASH_ATTENTION=1 "
        f"OLLAMA_MAX_LOADED_MODELS=1 "
        f"OLLAMA_DEBUG=0 "
        f"ollama serve"
    )


def run_command(cmd):
    """Execute the given command."""
    os.system(cmd)


def main():
    """Main function to parse arguments, construct command, and start the server."""
    args = parse_arguments()
    
    hostname = os.uname().nodename
    host = f"{hostname}.snu.vision"
    cmd = construct_command(host, args.port, args.model_path, args.keep_alive)
    run_command(cmd)


if __name__ == "__main__":
    main()

