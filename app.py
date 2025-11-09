from flask import Flask, render_template, request, jsonify
import torch
from torchtext.data.utils import get_tokenizer
import torch.nn as nn
import os, logging

# -------------------------------------------------
# 1. Initialize Flask app and logging
# -------------------------------------------------
app = Flask(__name__, template_folder='templates', static_folder='static')
logging.basicConfig(level=logging.INFO)

# -------------------------------------------------
# 2. Define LSTM model
# -------------------------------------------------
class LSTM(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, num_layers, bidirectional, dropout, output_dim, pad_idx):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            emb_dim,
            hid_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hid_dim * 2, output_dim)

    def forward(self, text, text_length):
        embedded = self.embedding(text)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_length.to('cpu'), enforce_sorted=False, batch_first=True
        )
        packed_output, (hn, _) = self.lstm(packed_embedded)
        hn = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        return self.fc(hn)

# -------------------------------------------------
# 3. Load model and vocab locally
# -------------------------------------------------
try:
    vocab = torch.load('models/vocab.pt', map_location='cpu')
    pad_idx = vocab.get_stoi()['<pad>']

    input_dim = len(vocab)
    hid_dim = 256
    emb_dim = 300
    output_dim = 4
    num_layers = 2
    bidirectional = True
    dropout = 0.5

    model = LSTM(input_dim, emb_dim, hid_dim, num_layers, bidirectional, dropout, output_dim, pad_idx)
    model.load_state_dict(torch.load('models/LSTM.pt', map_location='cpu'))
    model.eval()
    logging.info("✅ Model and vocab loaded successfully.")
except Exception as e:
    logging.error(f"❌ Failed to load model or vocab: {e}")
    model = None
    vocab = None

# -------------------------------------------------
# 4. Tokenizer and label mapping
# -------------------------------------------------
tokenizer = get_tokenizer("basic_english")
label_map = {
    0: "World 🌍",
    1: "Sports 🏅",
    2: "Business 💼",
    3: "Science/Tech 🔬"
}

def text_pipeline(text, vocab):
    tokens = tokenizer(text.lower())
    stoi = vocab.get_stoi()
    ids = [stoi.get(token, stoi.get('<unk>')) for token in tokens]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

# -------------------------------------------------
# 5. Routes
# -------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vocab is None:
        return jsonify({'error': 'Model not loaded.'}), 500
    try:
        data = request.get_json()
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided.'}), 400
        with torch.no_grad():
            text_tensor = text_pipeline(text, vocab)
            text_length = torch.tensor([text_tensor.size(1)])
            outputs = model(text_tensor, text_length)
            pred = torch.argmax(outputs, dim=1).item()
        return jsonify({'label': label_map.get(pred, "No prediction available")})
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        return jsonify({'error': f'Prediction failed: {e}'}), 500

# -------------------------------------------------
# 6. Run app
# -------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
