from s2s import Seq2Seq

def main():
    # Seq2Seq model using Tensorflow / Keras
    s2s = Seq2Seq(batch_size = 64, epochs = 100, latent_dim = 256, 
        num_samples = 10000, data_path = "Datasets/infix_dataset.tsv"
        )
    s2s.prepareData()
    s2s.learnModel()
    print("model learned")
    s2s.testModel()
    print("model tested")


main()
    