import torch
import torch.nn as nn

class CRF(nn.Module):

    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))

    def forward(self, logits, tags=None):
        if tags is None:
            return self._decode(logits)

        return -self._calculate_score(logits, tags)

    def _calculate_score(self, logits, tags):
        score = 0
        for i in range(tags.shape[0]):
            score += logits[i, tags[i]]
            if i > 0:
                score += self.transitions[tags[i - 1], tags[i]]

        return score

    def _decode(self, logits):
        backpointers = []
        scores = []

        for i in reversed(range(logits.shape[0])):
            v = logits[i]
            bp = []
            for prev in range(v.shape[0]):
                score = v[prev]
                if len(backpointers) > i:
                    score += self.transitions[backpointers[i][prev], prev]

                bp.append((prev, score))

            scores.append(bp)
            backpointers.append(scores[-1][torch.argmax(torch.stack([x[1] for x in scores[-1]]), dim=0)])

        result = []
        current = backpointers[-1][torch.argmax(torch.stack([x[1] for x in backpointers[-1]]), dim=0)][0]
        for bp in reversed(backpointers[:-1]):
            result.append(current)
            current = bp[current][0]

        result.reverse()
        return result
