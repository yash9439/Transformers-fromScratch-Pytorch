import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import math
import matplotlib.pyplot as plt
from tqdm import tqdm


# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

# Define a function to load and preprocess data from files


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.read().strip().split('\n')
    return lines


# Load and preprocess data for training and testing
train_en = load_data('ted-talks-corpus/train.en')
train_fr = load_data('ted-talks-corpus/train.fr')
test_en = load_data('ted-talks-corpus/test.en')
test_fr = load_data('ted-talks-corpus/test.fr')
dev_en = load_data('ted-talks-corpus/dev.en')
dev_fr = load_data('ted-talks-corpus/dev.fr')

train_set = []
test_set = []
dev_set = []
for i in range(len(train_en)):
    if len(train_en[i].split()) > 90 or len(train_fr[i].split()) > 90:
        continue
    train_set.append({'en': train_en[i], 'fr': train_fr[i]})
for i in range(len(test_en)):
    if len(test_en[i].split()) > 90 or len(test_fr[i].split()) > 90:
        continue
    test_set.append({'en': test_en[i], 'fr': test_fr[i]})
for i in range(len(dev_en)):
    if len(dev_en[i].split()) > 90 or len(dev_fr[i].split()) > 90:
        continue
    dev_set.append({'en': dev_en[i], 'fr': dev_fr[i]})



class BilingualDataset(Dataset):

    def __init__(self, dataset, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len

        self.dataset = dataset
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.sos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor(
            [tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        src_target_pair = self.dataset[idx]
        src_text = src_target_pair[self.src_lang]
        tgt_text = src_target_pair[self.tgt_lang]

        # Transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # Add sos, eos and padding to each sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # Add <s> and </s> tokens
        encoder_input = [self.sos_token] + enc_input_tokens + \
            [self.eos_token] + [self.pad_token] * enc_num_padding_tokens
        encoder_input = torch.tensor(encoder_input, dtype=torch.int64)

        # Add only <s> token
        decoder_input = [self.sos_token] + dec_input_tokens + \
            [self.pad_token] * dec_num_padding_tokens
        decoder_input = torch.tensor(decoder_input, dtype=torch.int64)

        # Add only </s> token
        label = dec_input_tokens + [self.eos_token] + \
            [self.pad_token] * dec_num_padding_tokens
        label = torch.tensor(label, dtype=torch.int64)

        return {
            "encoder_input": encoder_input,  # (seq_len)
            "decoder_input": decoder_input,  # (seq_len)
            # (1, 1, seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            # (1, seq_len) & (1, seq_len, seq_len),
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            "label": label,  # (seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


class LayerNormalization(nn.Module):
    def __init__(self, features, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        # alpha is a learnable parameter
        self.alpha = nn.Parameter(torch.ones(features))
        # bias is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
        # Keep the dimension for broadcasting
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, d_ff, dropout) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len)
        position = torch.arange(
            0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float(
        ) * (-math.log(10000.0) / d_model))  # (d_model / 2)
        # Apply sine to even indices
        # sin(position * (10000 ** (2i / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        # cos(position * (10000 ** (2i / d_model))
        pe[:, 1::2] = torch.cos(position * div_term)
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        # (batch, seq_len, d_model)
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class ResidualConnection(nn.Module):
    def __init__(self, features, dropout) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, h, dropout) -> None:
        super().__init__()
        self.d_model = d_model  # Embedding vector size
        self.h = h  # Number of heads
        self.d_k = d_model // h  # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        # (batch, h, seq_len, seq_len) # Apply softmax
        attention_scores = attention_scores.softmax(dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        query = self.w_q(q)
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(k)
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(v)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(
            query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1],
                       self.h, self.d_k).transpose(1, 2)
        value = value.view(
            value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout)

        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(
            x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)


class EncoderBlock(nn.Module):
    def __init__(self, features, self_attention_block, feed_forward_block, dropout) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, features, layers) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, features, self_attention_block, cross_attention_block, feed_forward_block, dropout) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(
            x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, features, layers) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, d_model=512, N=2, h=8, dropout=0.1, d_ff=2048) -> None:
        super().__init__()
        self.inputEmbedding = nn.Embedding(src_vocab_size, d_model)
        self.targetEmbedding = nn.Embedding(tgt_vocab_size, d_model)
        self.d_model = d_model
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
        self.tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
        self.projection_layer = nn.Linear(d_model, tgt_vocab_size)

        # Create the encoder blocks
        encoder_blocks = []
        for _ in range(N):
            encoder_self_attention_block = MultiHeadAttentionBlock(
                d_model, h, dropout)
            feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
            encoder_block = EncoderBlock(
                d_model, encoder_self_attention_block, feed_forward_block, dropout)
            encoder_blocks.append(encoder_block)

        # Create the encoder
        self.encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))

        # Create the decoder blocks
        decoder_blocks = []
        for _ in range(N):
            decoder_self_attention_block = MultiHeadAttentionBlock(
                d_model, h, dropout)
            decoder_cross_attention_block = MultiHeadAttentionBlock(
                d_model, h, dropout)
            feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
            decoder_block = DecoderBlock(
                d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
            decoder_blocks.append(decoder_block)

        # Create the decoder
        self.decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    def encode(self, src, src_mask):
        # (batch, seq_len, d_model)
        src = self.inputEmbedding(src) * math.sqrt(self.d_model)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        # (batch, seq_len, d_model)
        tgt = self.targetEmbedding(tgt) * math.sqrt(self.d_model)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        # (batch, seq_len, vocab_size)
        return self.projection_layer(x)
    


# Input Tokenizer
tokenizer_src = Tokenizer(WordLevel(unk_token="[UNK]"))
tokenizer_src.pre_tokenizer = Whitespace()
trainer = WordLevelTrainer(
    special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
tokenizer_src.train_from_iterator(
    [item['en'] for item in train_set], trainer=trainer)

# Output Tokenizer
tokenizer_tgt = Tokenizer(WordLevel(unk_token="[UNK]"))
tokenizer_tgt.pre_tokenizer = Whitespace()
trainer = WordLevelTrainer(
    special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
tokenizer_tgt.train_from_iterator(
    [item['fr'] for item in train_set], trainer=trainer)



# Define the device
import sacrebleu
from sacrebleu.metrics import BLEU
bleu = BLEU()

device = "cuda" if torch.cuda.is_available() else "cpu"

device = torch.device(device)

# Initialize lists to store loss and BleuScore for different hyperparameters
loss_data = []
bleu_data = []

# Hyperparameters
lr_list= [10**-2, 10**-3, 10**-4]
num_epochs = 2
batch_size_list = [32, 64]
dropout_list = [0.1, 0.3]

ct = 0
for lr in lr_list:
    for batch_size in batch_size_list:
        for dropout in dropout_list:
            ct += 1
            d_model = 512
            seq_len = 128
            N = 2
            h = 4
            d_ff = 2048

            train_ds = BilingualDataset(train_set, tokenizer_src, tokenizer_tgt, 'en', 'fr', seq_len)
            dev_ds = BilingualDataset(dev_set, tokenizer_src, tokenizer_tgt, 'en', 'fr', seq_len)
            test_ds = BilingualDataset(test_set, tokenizer_src, tokenizer_tgt, 'en', 'fr', seq_len)

            train_dataloader = DataLoader(train_ds, batch_size=batch_size)

            model = Transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(), seq_len, seq_len, d_model, N, h, dropout, d_ff)

            # Initialize the parameters
            for p in model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            model = model.to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr, eps=1e-9)

            loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id(
                '[PAD]'), label_smoothing=0.1).to(device)

            # Storing the losses after every epoch
            losses = []
            val_bleu = []

            for epoch in range(num_epochs):
                torch.cuda.empty_cache()
                epoch_loss = 0.0
                model.train()
                batch_iterator = tqdm(
                    train_dataloader, desc=f"Processing Epoch {epoch:02d}")
                for batch in batch_iterator:

                    encoder_input = batch['encoder_input'].to(device)  # (b, seq_len)
                    decoder_input = batch['decoder_input'].to(device)  # (B, seq_len)
                    encoder_mask = batch['encoder_mask'].to(device)  # (B, 1, 1, seq_len)
                    decoder_mask = batch['decoder_mask'].to(
                        device)  # (B, 1, seq_len, seq_len)

                    # Run the tensors through the encoder, decoder and the projection layer
                    encoder_output = model.encode(
                        encoder_input, encoder_mask)  # (B, seq_len, d_model)
                    decoder_output = model.decode(
                        encoder_output, encoder_mask, decoder_input, decoder_mask)  # (B, seq_len, d_model)
                    proj_output = model.project(decoder_output)  # (B, seq_len, vocab_size)

                    # Compare the output with the label
                    label = batch['label'].to(device)  # (B, seq_len)

                    # Compute the loss using a simple cross entropy
                    loss = loss_fn(
                        proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
                    batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

                    # Add the loss to the epoch loss
                    epoch_loss += loss.item()

                    # Backpropagate the loss
                    loss.backward()

                    # Update the weights
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                # Calculate the average loss for the epoch
                avg_epoch_loss = epoch_loss / len(train_dataloader)

                # Append the average loss to the list
                losses.append(avg_epoch_loss)

                model.eval()
                with torch.no_grad():
                    true = []
                    preds = []
                    for sent in dev_ds:
                        # Precompute the encoder output and reuse it for every generation step
                        source = tokenizer_src.encode(sent['src_text'])

                        # Adding Start and End Token with Padding
                        source = torch.cat([
                            torch.tensor([tokenizer_src.token_to_id('[SOS]')],
                                        dtype=torch.int64),
                            torch.tensor(source.ids, dtype=torch.int64),
                            torch.tensor([tokenizer_src.token_to_id('[EOS]')],
                                        dtype=torch.int64),
                            torch.tensor([tokenizer_src.token_to_id('[PAD]')] *
                                        (seq_len - len(source.ids) - 2), dtype=torch.int64)
                        ], dim=0).to(device)

                        # Making input mask
                        source_mask = (source != tokenizer_src.token_to_id(
                            '[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)

                        encoder_output = model.encode(source, source_mask)

                        # Initialize the decoder input with the sos token
                        decoder_input = torch.empty(1, 1).fill_(
                            tokenizer_tgt.token_to_id('[SOS]')).type_as(source).to(device)

                        pred_trans = []
                        while decoder_input.size(1) < seq_len:
                            # build mask for target and calculate output
                            decoder_mask = torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(
                                1))), diagonal=1).type(torch.int).type_as(source_mask).to(device)
                            out = model.decode(encoder_output, source_mask,
                                            decoder_input, decoder_mask)

                            # project next token
                            prob = model.project(out[:, -1])
                            _, next_word = torch.max(prob, dim=1)
                            decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(
                                source).fill_(next_word.item()).to(device)], dim=1)

                            # # print the translated word
                            # print(f"{tokenizer_tgt.decode([next_word.item()])}", end=' ')

                            pred_trans.append(tokenizer_tgt.decode([next_word.item()]))

                            # break if we predict the end of sentence token
                            if next_word == tokenizer_tgt.token_to_id('[EOS]') or decoder_input.size(1) == seq_len:
                                preds.append(' '.join(pred_trans))
                                true.append(sent['tgt_text'])
                                break
 
                    print(len(preds))
                    print(len(true))
                    bleuScore = bleu.corpus_score(preds, true)
                    val_bleu.append(bleuScore)
                    print(bleuScore)

            # Save the model at the end of every epoch
            model_filename = "SavedModel" + str(ct)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_filename)

            val_bleu_plot = []
            for i in val_bleu:
                val_bleu_plot.append(i.score)

            loss_data.append((ct, losses))
            bleu_data.append((ct, val_bleu_plot))



import matplotlib.pyplot as plt

# Create the Loss Graph with a white background
plt.figure(figsize=(12, 6), facecolor='w')
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")

for label, loss_values in loss_data:
    ct = label
    plt.plot(range(len(loss_values)), loss_values, label=f"ct={ct}")

plt.legend()
plt.grid(True)

# Save the Loss Graph with a white background
plt.tight_layout()
plt.savefig('loss_graph1.png', facecolor='w')


# Create the BleuScore Graph with a white background
plt.figure(figsize=(12, 6), facecolor='w')
plt.title("Bleu Score Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Bleu Score")

for label, bleu in bleu_data:
    ct, bleu_values = label, bleu
    plt.plot(range(len(bleu_values)), bleu_values, label=f"ct={ct}")

plt.legend()
plt.grid(True)

# Save the BleuScore Graph with a white background
plt.tight_layout()
plt.savefig('bleu_score1.png', facecolor='w')



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
seq_len = 128
model = Transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(
), seq_len, seq_len, d_model=512, N=2, h=16, dropout=0.3, d_ff=2048)

# Load the pretrained weights
model_filename = "SavedModel"
state = torch.load(model_filename)
model.load_state_dict(state['model_state_dict'])





def translate(sentence, tokenizer_src, tokenizer_tgt, seq_len, d_model):
    # Define the device, tokenizers, and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = Transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(
    ), seq_len, seq_len, d_model=512, N=2, h=16, dropout=0.3, d_ff=2048)

    model = model.to(device)

    # Load the pretrained weights
    model_filename = "SavedModel"
    state = torch.load(model_filename)
    model.load_state_dict(state['model_state_dict'])

    # translate the sentence
    model.eval()
    with torch.no_grad():
        # Precompute the encoder output and reuse it for every generation step
        source = tokenizer_src.encode(sentence)

        # Adding Start and End Token with Padding
        source = torch.cat([
            torch.tensor([tokenizer_src.token_to_id('[SOS]')],
                         dtype=torch.int64),
            torch.tensor(source.ids, dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[EOS]')],
                         dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[PAD]')] *
                         (seq_len - len(source.ids) - 2), dtype=torch.int64)
        ], dim=0).to(device)

        # Making input mask
        source_mask = (source != tokenizer_src.token_to_id(
            '[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)

        encoder_output = model.encode(source, source_mask)

        # Initialize the decoder input with the sos token
        decoder_input = torch.empty(1, 1).fill_(
            tokenizer_tgt.token_to_id('[SOS]')).type_as(source).to(device)

        # Print the source sentence and target start prompt
        print(f"   SOURCE: {sentence}")
        print("PREDICTED: ", end='')
        # Generate the translation word by word
        while decoder_input.size(1) < seq_len:
            # build mask for target and calculate output
            decoder_mask = torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(
                1))), diagonal=1).type(torch.int).type_as(source_mask).to(device)
            out = model.decode(encoder_output, source_mask,
                               decoder_input, decoder_mask)

            # project next token
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(
                source).fill_(next_word.item()).to(device)], dim=1)

            # print the translated word
            print(f"{tokenizer_tgt.decode([next_word.item()])}", end=' ')

            # break if we predict the end of sentence token
            if next_word == tokenizer_tgt.token_to_id('[EOS]'):
                break

    # convert ids to tokens
    return tokenizer_tgt.decode(decoder_input[0].tolist())




t = translate("Once upon a time, in the clearing of the magical forest.",
              tokenizer_src, tokenizer_tgt, seq_len, d_model)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = Transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(
), seq_len, seq_len, d_model=512, N=2, h=16, dropout=0.3, d_ff=2048)

model = model.to(device)

# Load the pretrained weights
model_filename = "SavedModel"
state = torch.load(model_filename)
model.load_state_dict(state['model_state_dict'])

test_file_data = []
# translate the sentence
model.eval()
with torch.no_grad():
    for sent in tqdm(test_ds):
        local_dict = {}
        sentence = sent['src_text']
        local_dict["input"] = sentence
        local_dict["true"] = sent['tgt_text']
        # Precompute the encoder output and reuse it for every generation step
        source = tokenizer_src.encode(sentence)

        # Adding Start and End Token with Padding
        source = torch.cat([
            torch.tensor([tokenizer_src.token_to_id('[SOS]')],
                        dtype=torch.int64),
            torch.tensor(source.ids, dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[EOS]')],
                        dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[PAD]')] *
                        (seq_len - len(source.ids) - 2), dtype=torch.int64)
        ], dim=0).to(device)

        # Making input mask
        source_mask = (source != tokenizer_src.token_to_id(
            '[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)

        encoder_output = model.encode(source, source_mask)

        # Initialize the decoder input with the sos token
        decoder_input = torch.empty(1, 1).fill_(
            tokenizer_tgt.token_to_id('[SOS]')).type_as(source).to(device)
        sentpred = ""

        while decoder_input.size(1) < seq_len:
            # build mask for target and calculate output
            decoder_mask = torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(
                1))), diagonal=1).type(torch.int).type_as(source_mask).to(device)
            out = model.decode(encoder_output, source_mask,
                            decoder_input, decoder_mask)

            # project next token
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(
                source).fill_(next_word.item()).to(device)], dim=1)

            # print the translated word
            sentpred += f"{tokenizer_tgt.decode([next_word.item()])} "
            # print(f"{tokenizer_tgt.decode([next_word.item()])}", end=' ')

            # break if we predict the end of sentence token
            if next_word == tokenizer_tgt.token_to_id('[EOS]') or decoder_input.size(1) == seq_len:
                local_dict["pred"] = sentpred
                local_dict["BleuScore"] = sacrebleu.sentence_bleu(sentpred, [sent['tgt_text']]).score
                test_file_data.append(local_dict)
                break
            



import json

with open("Test_Sent.txt", 'w', encoding='utf-8') as json_file:
    json.dump(test_file_data, json_file, ensure_ascii=False, indent=4)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = Transformer(tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size(
), seq_len, seq_len, d_model=512, N=2, h=16, dropout=0.3, d_ff=2048)

model = model.to(device)

# Load the pretrained weights
model_filename = "SavedModel"
state = torch.load(model_filename)
model.load_state_dict(state['model_state_dict'])

train_file_data = []
# translate the sentence
model.eval()
with torch.no_grad():
    for sent in tqdm(train_ds):
        local_dict = {}
        sentence = sent['src_text']
        local_dict["input"] = sentence
        local_dict["true"] = sent['tgt_text']
        # Precompute the encoder output and reuse it for every generation step
        source = tokenizer_src.encode(sentence)

        # Adding Start and End Token with Padding
        source = torch.cat([
            torch.tensor([tokenizer_src.token_to_id('[SOS]')],
                        dtype=torch.int64),
            torch.tensor(source.ids, dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[EOS]')],
                        dtype=torch.int64),
            torch.tensor([tokenizer_src.token_to_id('[PAD]')] *
                        (seq_len - len(source.ids) - 2), dtype=torch.int64)
        ], dim=0).to(device)

        # Making input mask
        source_mask = (source != tokenizer_src.token_to_id(
            '[PAD]')).unsqueeze(0).unsqueeze(0).int().to(device)

        encoder_output = model.encode(source, source_mask)

        # Initialize the decoder input with the sos token
        decoder_input = torch.empty(1, 1).fill_(
            tokenizer_tgt.token_to_id('[SOS]')).type_as(source).to(device)
        sentpred = ""

        while decoder_input.size(1) < seq_len:
            # build mask for target and calculate output
            decoder_mask = torch.triu(torch.ones((1, decoder_input.size(1), decoder_input.size(
                1))), diagonal=1).type(torch.int).type_as(source_mask).to(device)
            out = model.decode(encoder_output, source_mask,
                            decoder_input, decoder_mask)

            # project next token
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat([decoder_input, torch.empty(1, 1).type_as(
                source).fill_(next_word.item()).to(device)], dim=1)

            # print the translated word
            sentpred += f"{tokenizer_tgt.decode([next_word.item()])} "
            # print(f"{tokenizer_tgt.decode([next_word.item()])}", end=' ')

            # break if we predict the end of sentence token
            if next_word == tokenizer_tgt.token_to_id('[EOS]') or decoder_input.size(1) == seq_len:
                local_dict["pred"] = sentpred
                local_dict["BleuScore"] = sacrebleu.sentence_bleu(sentpred, [sent['tgt_text']]).score
                train_file_data.append(local_dict)
                break
            


import json

with open("Train_Sent.txt", 'w', encoding='utf-8') as json_file:
    json.dump(train_file_data, json_file, ensure_ascii=False, indent=4)