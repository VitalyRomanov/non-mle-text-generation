import numpy as np

from SeqT5 import SeqT5


def main():
    model = SeqT5.from_pretrained('t5-small')
    weights = model.encoder.embed_tokens.weight.cpu().detach().numpy()
    # add zero-th vector for compatibility with dictionary
    weights = np.concatenate([np.random.rand(1, weights.shape[1]) * 1e-7, weights], axis=0)
    np.save("t5_embeddings", weights)


if __name__ == "__main__":
    main()