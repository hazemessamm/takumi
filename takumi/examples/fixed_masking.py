import os
os.environ["TAKUMI_MASKING_BACKEND"] = "torch"
from takumi.masking import Corruptor, Bernoulli, Mixed, BetaRate
import torch

corruptor = Corruptor(
    rate=BetaRate(0.15, 4.0),
    select=Bernoulli(),
    corrupt=Mixed(mask_prob=0.8, mutate_prob=0.1, mask_token_id=3, valid_token_ids=[i for i in range(4, 100)]),
    special_token_ids=[0, 1, 2, 3],
)


x = torch.randint(4, 100, (1, 200))
output = corruptor(x, 10)

print(output)