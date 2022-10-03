## Non MLE Text Generation

Repozitory of [nazim1021](https://github.com/nazim1021/neural-machine-translation-using-gan) is used as a starting point for this project. The changes are

1) Added scripts for converting summarization and translation datasets into binary format
2) Reimplemented ModelTrainer class
3) Added models: 
    - T5 trained with MLE
    - T5 trained with Reinforcement Learning
    - T5 trained with Gumbel Softmax
    - T5 evaluated with BLEURT score
    - Continuous Space Text Generation 
