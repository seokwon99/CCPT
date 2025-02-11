import os
from datasets import load_dataset
import concurrent.futures
import backoff
import anthropic
from anthropic import Anthropic

class ParallelClaude():
    def __init__(self, model_id):
        self.model_id = model_id
        self.client = Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
        
    # @backoff.on_exception(backoff.expo, (anthropic.RateLimitError, anthropic.APIError, anthropic.Timeout, anthropic.BadRequestError, anthropic.APIConnectionError, anthropic.InternalServerError))
    def generate(self, text, max_new_tokens=2048, temperature=1, system_prompt=None, num_return_sequences=1, **kwargs):
        if isinstance(text, str):
            text = [text]
        def process_text(t, idx):
            responses_for_instance = []
            for seq in range(num_return_sequences):
                completion = self.client.messages.create(
                    model=self.model_id,
                    messages=[
                        {"role": "user", "content": f"{t} Sequence: {seq}"}
                    ],
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    system=system_prompt
                )
                responses_for_instance.append(completion.content[0].text)
            return (responses_for_instance, idx)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_text, t, idx) for idx, t in enumerate(text)]
            completions = []
            for future in concurrent.futures.as_completed(futures):
                completions.append(future.result())

        completions_sorted = sorted(completions, key=lambda x: x[1])
        responses = [completion[0] for completion in completions_sorted]

        return {'responses': responses}