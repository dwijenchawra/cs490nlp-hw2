**Dwijen** | CS 490: Natural Language Processing | Spring 2026

# Homework 2

## Task 5: Experimental Analysis

![Per-Step Training Loss](../task5_loss_curves.png)

The plot shows per-step training loss (smoothed to reduce noise) for all four models. BERT converges to the lowest loss the fastest, dropping sharply within the first ~2000 steps and stabilizing near zero, because its pre-trained contextual embeddings already encode rich semantic information so fine-tuning only needs small adjustments. BoW MLP converges next, reaching low loss by around 3000–4000 steps, since its frequency-based representation still captures useful signals. One-hot MLP follows a similar trajectory but requires more gradient steps (~5000+) to settle, as its vectors carry no pre-learned semantic information and the network must learn all useful representations from scratch.

\newpage

## Question 1: Contextual Embeddings

In the first sentence "bank" means a financial institution, and in the second it means the edge of a river. Word2vec assigns the same fixed vector to "bank" in both cases because it only learns one embedding per word regardless of context. So for both sentences, the closest word in word2vec space would be something like "account," since financial usage dominates the training data. BERT uses self-attention to look at surrounding words, so it produces a different embedding for "bank" in each sentence. In the first sentence "bank" would be closest to something like "vault" because of words like "deposit" and "money," and in the second it would be closest to something like "shore" because of "river."

\newpage

## Question 2: The [CLS] Token

1. During pre-training, the [CLS] token is used for the next sentence prediction task, where the model predicts whether two sentences appear next to each other. Its final hidden state is what gets fed into that prediction, so BERT learns to pack a summary of the entire input into that one token's representation.

2. We use [CLS] instead of averaging all token embeddings because it is specifically trained to represent the whole sequence. Averaging treats every token equally, which dilutes the signal with padding and unimportant words. The [CLS] token already aggregates information from every other token through self-attention, so it stores a useful summary of the sentence's meaning without needing to average anything.

\newpage

## Question 3: Tokenization

With whole-word tokenization, any word not in the vocabulary gets replaced with \<UNK\> and all its meaning is lost. Subword tokenization fixes this by breaking unknown words into smaller known pieces. This keeps the vocabulary small (around 30,000 subwords) while still being able to represent any word. For example, "unfathomable" gets split into "un," "fat," "##hom," "##able," so the model still gets useful information from each part instead of throwing the whole word away.

\newpage

## Question 4: Fine-Tuning vs. Feature Extraction

1. Full fine-tuning typically yields higher accuracy because updating all of BERT's weights lets the model adapt its internal representations to the specific task. When you freeze BERT, the model is stuck with general-purpose representations that are not optimized for something like sarcasm detection, so the classifier has less to work with.

2. Freezing BERT and only training the linear layer is much cheaper. This is because you only need to compute gradients for the linear layer's weights instead of all ~110 million parameters in BERT. Full fine-tuning requires computing and storing gradients for every layer on the backward pass, which takes significantly more memory and time.
