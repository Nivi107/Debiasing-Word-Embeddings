from __future__ import print_function, division
from debiaswe import we
import json
import numpy as np
import argparse
import sys
import torch
if sys.version_info[0] < 3:
    import io
    open = io.open
"""
Hard-debias embedding

Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings
Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, and Adam Kalai
2016
"""


def debias(E, gender_specific_words, definitional, equalize):
    gender_direction = we.doPCA(definitional, E).components_[0]
    specific_set = set(gender_specific_words)
    for i, w in enumerate(E.words):
        if w not in specific_set:
            E.vecs[i] = we.drop(E.vecs[i], gender_direction)
    E.normalize()
    candidates = {x for e1, e2 in equalize for x in [(e1.lower(), e2.lower()),
                                                     (e1.title(), e2.title()),
                                                     (e1.upper(), e2.upper())]}
    print(candidates)
    for (a, b) in candidates:
        if (a in E.index and b in E.index):
            y = we.drop((E.v(a) + E.v(b)) / 2, gender_direction)
            z = np.sqrt(1 - np.linalg.norm(y)**2)
            if (E.v(a) - E.v(b)).dot(gender_direction) < 0:
                z = -z
            E.vecs[E.index[a]] = z * gender_direction + y
            E.vecs[E.index[b]] = -z * gender_direction + y
    E.normalize()

"""
soft-debiasing
"""
def soft_debias(E, gender_specific_words, definitional, equalize, lambda_strength=0.2, lr=0.0001, epochs=10):
    gender_direction = we.doPCA(definitional, E).components_[0]
    specific_set = set(gender_specific_words)

    # Convert embeddings to PyTorch tensors
    vocabIndex = {i: w for i, w in enumerate(E.words)}
    vocabVectors = E.vecs
    embedding_dim = vocabVectors.shape[1]

    Neutrals = torch.tensor([E.v(w) for w in gender_specific_words if w in E.index]).float().t()
    Words = torch.tensor(vocabVectors).float().t()

    # Perform SVD on Words
    u, s, _ = torch.svd(Words)
    s = torch.diag(s)

    # Precompute matrices
    t1 = s.mm(u.t())
    t2 = u.mm(s)
    
    # Initialize Transform and BiasSpace
    Transform = torch.randn(embedding_dim, embedding_dim).float()
    BiasSpace = torch.tensor(gender_direction).view(embedding_dim, -1).float()

    Neutrals.requires_grad = False
    Words.requires_grad = False
    BiasSpace.requires_grad = False
    Transform.requires_grad = True

    # Optimizer
    optimizer = torch.optim.SGD([Transform], lr=lr)

    for i in range(epochs):
        # Calculate norms
        TtT = torch.mm(Transform.t(), Transform)
        norm1 = (t1.mm(TtT - torch.eye(embedding_dim)).mm(t2)).norm(p=2)
        norm2 = (Neutrals.t().mm(TtT).mm(BiasSpace)).norm(p=2)

        # Loss function
        loss = norm1 + lambda_strength * norm2
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print("Optimization Completed, normalizing vector transform")

     # Update the embedding vectors in-place
    for i, w in enumerate(Words.t()):
        transformedVec = torch.mm(Transform, w.view(-1, 1))
        E.vecs[i] = (transformedVec / transformedVec.norm(p=2)).detach().numpy().flatten()

    E.reindex()  # Ensure internal state reflects the updated vectors

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("embedding_filename", help="The name of the embedding")
    parser.add_argument("definitional_filename", help="JSON of definitional pairs")
    parser.add_argument("gendered_words_filename", help="File containing words not to neutralize (one per line)")
    parser.add_argument("equalize_filename", help="???.bin")
    parser.add_argument("debiased_filename", help="???.bin")

    args = parser.parse_args()
    print(args)

    with open(args.definitional_filename, "r") as f:
        defs = json.load(f)
    print("definitional", defs)

    with open(args.equalize_filename, "r") as f:
        equalize_pairs = json.load(f)

    with open(args.gendered_words_filename, "r") as f:
        gender_specific_words = json.load(f)
    print("gender specific", len(gender_specific_words), gender_specific_words[:10])

    E = we.WordEmbedding(args.embedding_filename)

    print("Debiasing...")
    debias(E, gender_specific_words, defs, equalize_pairs)

    print("Saving to file...")
    if args.embedding_filename[-4:] == args.debiased_filename[-4:] == ".bin":
        E.save_w2v(args.debiased_filename)
    else:
        E.save(args.debiased_filename)

    print("\n\nDone!\n")
