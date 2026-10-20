import numpy as np
from mails import mails, spam       # 1000 emails

num_emails = 1000

words = sorted(set(' '.join(mails).split()))
col = {w: i for i, w in enumerate(words)}
X = np.zeros((num_emails, len(words)))     # word counts

for n, m in enumerate(mails):
    for w in m.split():
        X[n, col[w]] += 1

y = np.array(spam)
S = X[:300][y[:300] == 1].sum(0) + 1
H = X[:300][y[:300] == 0].sum(0) + 1

score = np.log(S / S.sum()) - np.log(H / H.sum())
guess = X[300:] @ score > 0         # add, decide
print((guess == y[300:]).mean())    # unseen mails

new = "you won a free trip claim it now"
print(sum(score[col[w]] for w in new.split()))
print(words[score.argmax()])        # most spammy
