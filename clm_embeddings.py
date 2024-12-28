import os
import torch
import torchaudio
import pretty_midi
import numpy as np
import torch.nn as nn
from transformers import AutoModel, AutoProcessor, AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
import argparse

# Dataset Class
class AudioTextMelodyDataset(Dataset):
    def __init__(self, audio_dir, melodies_dir, captions_file, processor, emotion_tokenizer, emotion_model):
        self.audio_dir = audio_dir
        self.melodies_dir = melodies_dir
        self.processor = processor
        self.emotion_tokenizer = emotion_tokenizer
        self.emotion_model = emotion_model

        # Load captions
        with open(captions_file, 'r', encoding='utf-8') as f:
            self.captions = [line.strip() for line in f if line.strip()]

        # Get sorted list of audio and melody files
        self.audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
        self.melody_files = sorted([f for f in os.listdir(melodies_dir) if f.endswith('.mid')])

        # Check consistency
        assert len(self.captions) == len(self.audio_files) == len(self.melody_files), "Mismatch in counts!"

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.audio_files[idx])
        melody_path = os.path.join(self.melodies_dir, self.melody_files[idx])
        caption = self.captions[idx]

        # Load and process audio
        waveform, sample_rate = torchaudio.load(audio_path)
        expected_sample_rate = self.processor.feature_extractor.sampling_rate
        if sample_rate != expected_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=expected_sample_rate)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        audio_features = self.processor.feature_extractor(
            waveform.squeeze(0).numpy(),
            sampling_rate=expected_sample_rate,
            return_tensors="pt"
        )
        input_features = audio_features['input_features'].squeeze(0)

        # Melody
        melody_embedding = self.extract_melody(melody_path)

        # Text
        text_tokens = self.processor.tokenizer(
            caption,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128,
        )
        input_ids = text_tokens['input_ids'].squeeze(0)
        attention_mask = text_tokens['attention_mask'].squeeze(0)

        # Emotion
        with torch.no_grad():
            emotion_inputs = self.emotion_tokenizer(caption, return_tensors="pt", truncation=True, padding="max_length")
            emotion_logits = self.emotion_model(**emotion_inputs).logits.squeeze(0)

        return {
            'input_features': input_features,
            'melody_embedding': melody_embedding,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'emotion_logits': emotion_logits
        }

    def extract_melody(self, melody_path):
        midi_data = pretty_midi.PrettyMIDI(melody_path)
        notes = []
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    notes.append((note.pitch, note.end - note.start))
        # Sort by start time
        notes = sorted(notes, key=lambda x: x[1])

        melody = []
        for pitch, duration in notes[:16]:
            melody.extend([pitch, duration])

        while len(melody) < 32:
            melody.append(0)
        melody = melody[:32]
        melody = np.array(melody, dtype=np.float32)
        return torch.tensor(melody, dtype=torch.float32)

# CLaMP Model
class CLaMP(nn.Module):
    def __init__(self, clap_model, embedding_dim=512):
        super(CLaMP, self).__init__()
        self.audio_model = clap_model.audio_model
        self.text_model = clap_model.text_model

        # Extract hidden sizes
        audio_hidden_size = self.audio_model.config.hidden_size
        text_hidden_size = self.text_model.config.hidden_size

        # Debugging statement
        print(f"Configuring projection layers with audio_hidden_size={audio_hidden_size} and text_hidden_size={text_hidden_size}")

        # Projection layers for audio, text, melody, and emotion
        self.audio_proj = nn.Linear(audio_hidden_size, embedding_dim)
        self.text_proj = nn.Linear(text_hidden_size, embedding_dim)
        self.melody_proj = nn.Linear(32, embedding_dim)  # Assuming melody_embedding size is 32
        self.emotion_proj = nn.Linear(6, embedding_dim)  # Assuming emotion logits have 6 dimensions

    def forward(self, inputs):
        # Extract audio features
        audio_inputs = inputs['input_features']
        audio_outputs = self.audio_model(input_features=audio_inputs)
        audio_features = audio_outputs.last_hidden_state.mean(dim=(-2, -1))
        audio_embeddings = self.audio_proj(audio_features)

        # Extract melody features
        melody_embedding = inputs['melody_embedding']
        melody_embeddings = self.melody_proj(melody_embedding)

        # Extract text features
        text_inputs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        }
        text_outputs = self.text_model(**text_inputs)
        text_features = text_outputs.last_hidden_state.mean(dim=1)
        text_embeddings = self.text_proj(text_features)

        # Extract emotion features
        emotion_logits = inputs['emotion_logits']
        emotion_embeddings = self.emotion_proj(emotion_logits)

        # Combine embeddings
        combined_embeddings = torch.cat((audio_embeddings, melody_embeddings, text_embeddings, emotion_embeddings), dim=1)
        return combined_embeddings

# Main Function
def main():
    parser = argparse.ArgumentParser(description="Batch Embeddings Extraction")
    parser.add_argument('--audio_path', type=str, required=True, help='Path to directory of .wav files')
    parser.add_argument('--melody_path', type=str, required=True, help='Path to directory of .mid files')
    parser.add_argument('--caption', type=str, required=True, help='Path to captions.txt')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory for embeddings')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
    clap_model = AutoModel.from_pretrained("laion/clap-htsat-unfused").to(device)

    emotion_tokenizer = AutoTokenizer.from_pretrained("nateraw/bert-base-uncased-emotion")
    emotion_model = AutoModelForSequenceClassification.from_pretrained("nateraw/bert-base-uncased-emotion").to(device)

    model = CLaMP(clap_model, embedding_dim=512).to(device)

    # Load checkpoint and remove unexpected keys
    state_dict = torch.load(args.checkpoint, map_location=device)
    keys_to_remove = [
        "melody_encoder.pitch_emb.weight", "melody_encoder.duration_emb.weight",
        "melody_encoder.mlp.0.weight", "melody_encoder.mlp.0.bias",
        "melody_encoder.mlp.2.weight", "melody_encoder.mlp.2.bias"
    ]
    for key in keys_to_remove:
        if key in state_dict['model_state_dict']:
            print(f"Removing key: {key}")
            state_dict['model_state_dict'].pop(key)

    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()

    dataset = AudioTextMelodyDataset(
        args.audio_path, args.melody_path, args.caption, processor,
        emotion_tokenizer=emotion_tokenizer, emotion_model=emotion_model
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    audio_embeddings_list = []
    melody_embeddings_list = []
    text_embeddings_list = []
    emotion_embeddings_list = []

    embedding_dim = 512

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            embeddings = model(batch)  # (B, 2048)

            # Split embeddings
            audio_emb = embeddings[:, :embedding_dim].cpu().numpy()
            melody_emb = embeddings[:, embedding_dim:2*embedding_dim].cpu().numpy()
            text_emb = embeddings[:, 2*embedding_dim:3*embedding_dim].cpu().numpy()
            emotion_emb = embeddings[:, 3*embedding_dim:].cpu().numpy()

            audio_embeddings_list.append(audio_emb)
            melody_embeddings_list.append(melody_emb)
            text_embeddings_list.append(text_emb)
            emotion_embeddings_list.append(emotion_emb)

    audio_embeddings = np.concatenate(audio_embeddings_list, axis=0)
    melody_embeddings = np.concatenate(melody_embeddings_list, axis=0)
    text_embeddings = np.concatenate(text_embeddings_list, axis=0)
    emotion_embeddings = np.concatenate(emotion_embeddings_list, axis=0)

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "audio_embeddings.npy"), audio_embeddings)
    np.save(os.path.join(args.output_dir, "melody_embeddings.npy"), melody_embeddings)
    np.save(os.path.join(args.output_dir, "text_embeddings.npy"), text_embeddings)
    np.save(os.path.join(args.output_dir, "emotion_embeddings.npy"), emotion_embeddings)

    print("Embeddings saved successfully!")

if __name__ == '__main__':
    main()
