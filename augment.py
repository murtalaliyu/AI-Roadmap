from rag import pieces, retrieve

q = 'How many vacation days do I get per year?'

found = [pieces[i] for i in retrieve(q)]            # step 1

prompt = '\n'.join(['Answer using this context:', 
                    *found, 'Question: ' + q])      # step 2: augment

print(prompt)       # step 3: a language model gets this
