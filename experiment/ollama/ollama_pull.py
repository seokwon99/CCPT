import os
import time
import threading
import subprocess
from argparse import ArgumentParser


def parse_arguments():
    """Parse and return the command line arguments."""
    parser = ArgumentParser(description="Start the Ollama server with specified configurations.")
    parser.add_argument('--port', '-p', type=int, required=True, help='Port to run the server on')
    parser.add_argument('--model_path', '-mp', type=str, default='~/.ollama/models', help='Path to the models directory')
    parser.add_argument('--model', '-m', type=str, help='the model name to pull')
    return parser.parse_args()


def run_init(host, port, model_path, event):
    """Construct and return the command to start the Ollama server."""
    cmd = (
        f"OLLAMA_HOST={host}:{port} "
        f"OLLAMA_MODELS={model_path} "
        f"OLLAMA_DEBUG=0 "
        f"ollama serve"
    )
    process = subprocess.Popen(cmd, shell=True)
    event.wait()  # Wait until the run_pull event is set
    process.terminate()  # Send kill signal when run_pull is done


def run_pull(worker, port, model, event):
    """Construct and return the command to run the Ollama model."""
    cmd = (
        f"OLLAMA_HOST=http://{worker}.snu.vision:{port} "
        f"ollama pull {model}"
    )
    subprocess.run(cmd, shell=True)
    event.set()  # Set the event to signal that run_pull has finished


def main():
    """Main function to parse arguments, construct command, and start the server."""
    args = parse_arguments()
    
    hostname = os.uname().nodename
    host = f"{hostname}.snu.vision"
    event = threading.Event()
    
    init_thread = threading.Thread(target=run_init, args=(host, args.port, args.model_path, event))
    command_thread = threading.Thread(target=run_pull, args=(hostname, args.port, args.model, event))
    
    init_thread.start()
    time.sleep(5)  # Ensure run_init starts first
    command_thread.start()

    # Join threads to the main thread
    command_thread.join()
    event.set()  # Ensure run_init finishes after run_pull
    init_thread.join()

if __name__ == "__main__":
    main()
