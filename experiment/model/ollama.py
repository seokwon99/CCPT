import base64
import concurrent.futures
from litellm import completion

def encode_image(image_path):
    if image_path.startswith("http"):
        return image_path
    else:
        with open(image_path, "rb") as image_file:
            return f"data:image/jpeg;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"
        
class ParallelOllama():
    def __init__(self, model_id):
        self.model_id = model_id
        self.total_requests = 0

    # @backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIError, openai.Timeout, openai.BadRequestError, openai.APIConnectionError, openai.InternalServerError))
    def completion_with_backoff(self, **kwargs):
        return completion(**kwargs)

    def generate(self, text, image=None, max_new_tokens=1024, temperature=0.7, num_return_sequences=1, system_prompt = None, batch_size=1000, **kwargs):
        print(f"Input length: {len(text)}")
        output = self._generate(text, image=image, max_new_tokens=max_new_tokens, temperature=temperature, num_return_sequences=num_return_sequences, system_prompt=system_prompt, batch_size=batch_size, **kwargs)
        self.total_requests = 0
        return output
    
    def _generate(self, text, image=None, max_new_tokens=1024, temperature=0.7, num_return_sequences=1, system_prompt = None, batch_size=1000, **kwargs):
        print(f"Done length: {self.total_requests}")
        if len(text) > batch_size:
            conquer = self._generate(text[:batch_size], image=image, max_new_tokens=max_new_tokens, temperature=temperature, num_return_sequences=num_return_sequences, system_prompt=system_prompt, **kwargs)
            rest = self._generate(text[batch_size:], image=image, max_new_tokens=max_new_tokens, temperature=temperature, num_return_sequences=num_return_sequences, system_prompt=system_prompt, **kwargs)
            return {'responses': conquer['responses'] + rest['responses'], 'completions': conquer['completions'] + rest['completions']}
        else:
            self.total_requests += len(text)
            if isinstance(text, str):
                text = [text]
            if image is not None:
                if isinstance(image, str):
                    image = [image]
                assert len(text) == len(image)

                def process_text_and_image(t, i, idx):
                    base64_image = encode_image(i)
                    completion = self.completion_with_backoff(
                        model=self.model_id,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text", "text": t
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": 
                                        {
                                            "url": base64_image,
                                        },
                                    },
                                ],
                            }
                        ],
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        n=num_return_sequences,
                        **kwargs
                    )
                    return (completion, idx)


                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(process_text_and_image, t, i, idx) for idx, t, i in zip(range(len(text)), text, image)]
                    completions = []
                    for future in concurrent.futures.as_completed(futures):
                        completions.append(future.result())

                completions_sorted = sorted(completions, key=lambda x: x[1])
                responses = [[completion[0].choices[i].message.content for i in range(num_return_sequences)] for completion in completions_sorted]
                completions = [completion[0] for completion in completions_sorted]

                return {'responses': responses, 'completions': completions}

            else:
                MAX_RETURN_SEQUENCES = 1

                responses = []
                for i in range(0, num_return_sequences, MAX_RETURN_SEQUENCES):
                    print(f"step {i}/{num_return_sequences}")
                    def process_text(t, idx):
                        completion = self.completion_with_backoff(
                            model=self.model_id,
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text", "text": t
                                        },
                                    ],
                                }
                            ],
                            **kwargs
                        )
                            
                        return (completion, idx)


                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        futures = [executor.submit(process_text, t, idx) for idx, t in enumerate(text)]
                        completions = []
                        for future in concurrent.futures.as_completed(futures):
                            completions.append(future.result())
                    completions = sorted(completions, key=lambda x: x[1])
                    if responses == []:
                        responses = [[completion[0].choices[i].message.content for i in range(min(MAX_RETURN_SEQUENCES, num_return_sequences-i))] for completion in completions]
                    else:
                        for idx, completion in enumerate(completions):
                            responses[idx] += [completion[0].choices[i].message.content for i in range(min(MAX_RETURN_SEQUENCES, num_return_sequences-i))]
                
                print(f"Done length: {self.total_requests}")
                return {'responses': responses, 'completions': None}