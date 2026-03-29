""" Your SkipGram Word2Vec model. """

import torch


class Model(torch.nn.Module):
    """Your SkipGram Word2Vec model."""

    def __init__(self, vocab_size):
        super(Model, self).__init__()
        self.embedding_dim = 64
        ############################################################
        # STUDENT IMPLEMENTATION START
        # Note 1: see model.png. We recommend using a single torch.nn.Embedding 
        # (vocab size by embedding dim) and a single torch.nn.Linear layer with no bias.
        # See:
        # - https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html
        # - https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        # Note 2: We recommend initializing the embedding matrix with a uniform distribution, between -0.1 and 0.1
        # See: https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.uniform_
        ############################################################
        self.embeddings = torch.nn.Embedding(vocab_size, self.embedding_dim)
        self.linear = torch.nn.Linear(self.embedding_dim, vocab_size, bias=False)
        torch.nn.init.uniform_(self.embeddings.weight, -0.1, 0.1)
        

        ############################################################
        # STUDENT IMPLEMENTATION END
        ############################################################
        return

    def forward(self, input_ids):
        """Forward pass of the model.

        Args:
        input_ids: torch.Tensor, the input token indices [batch_size]

        Returns:
        logits: torch.Tensor, the output logits [batch_size, vocab_size]
        """
        ############################################################
        # STUDENT IMPLEMENTATION START
        # 1. see model.png.
        ############################################################
        embedded = self.embeddings(input_ids)
        logits = self.linear(embedded)


        ############################################################
        # STUDENT IMPLEMENTATION END
        ############################################################
        return logits

    def get_embeddings(self):
        """Return the embedding matrix.
        Returns:
        torch.Tensor, the embedding [vocab_size, embedding_dim]
        """
        return self.embeddings.weight
