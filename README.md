# sequential_gan

Some vague attempt at making a Generative Adversarial Network  that works on sequences.

There are a few problems to solve, they are not yet satisfactorily solved.

The most obvious way to train the generator is using policy gradients, but this doesn't work very well
(it is devastatingly slow so the discrimnator gets far too much of a head start). One suggested solution
is to pre-train in the [standard](https://arxiv.org/abs/1308.0850) manner and try and fine tune
adversarially, but some this is fundamentally unsatisfying.

Other solutions are changing the structure of the networks so we can just backpropagate all the way
through, but coming up with a useful way to do this is still a work in progress. In fact, all of this
is very much a work in progress, so don't expect it to make much sense.
