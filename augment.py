import argparse
import json
import os
import urllib.error
import urllib.request

from rag import pieces, retrieve

def augment(q, k=2):
    found = [pieces[i] for i in retrieve(q, k)]
    return '\n'.join([
        'Answer the question using only this context. If the context does not contain the answer, say so.',
        *found,
        'Question: ' + q,
        'Answer:',
    ])

def generate(prompt):
    host = os.getenv('LLAMA_HOST', 'http://127.0.0.1:9931').rstrip('/')
    model = os.getenv('LLAMA_MODEL', 'ggml-org/LFM2.5-24B-A2B-GGUF:Q4_K_M')
    url = host + '/v1/chat/completions'
    body = json.dumps({
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'stream': False,
    }).encode('utf-8')
    request = urllib.request.Request(
        url,
        data=body,
        headers={'Content-Type': 'application/json'},
    )

    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as error:
        detail = error.read().decode('utf-8', errors='replace')
        raise RuntimeError(f'llama returned HTTP {error.code}: {detail}') from error
    except urllib.error.URLError as error:
        raise RuntimeError(
            f'Could not connect to the model server at {host}. Check that it is running.'
        ) from error

    return result['choices'][0]['message']['content'].strip()

def main():
    parser = argparse.ArgumentParser(description='Ask a question about the sample contract.')
    parser.add_argument('question', nargs='?', help='question to ask')
    args = parser.parse_args()

    while True:
        q = args.question or input('Ask a question about the contract: ').strip()
        if not q:
            parser.error('a question is required')
    
        prompt = augment(q)
        print(generate(prompt))


if __name__ == '__main__':
    main()
