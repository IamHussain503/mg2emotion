import os
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, AutoProcessor, AutoModelForSequenceClassification
import torch.multiprocessing as mp


###############################################
# Melody Encoder
###############################################
class MelodyEncoder(nn.Module):
    def __init__(self, num_pitch=128, num_duration=512, pitch_emb_dim=64, dur_emb_dim=64, hidden_dim=256, proj_dim=32):
        super(MelodyEncoder, self).__init__()
        self.pitch_emb = nn.Embedding(num_pitch, pitch_emb_dim)
        self.duration_emb = nn.Embedding(num_duration, dur_emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(pitch_emb_dim + dur_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim)
        )

    def forward(self, pitch_tokens, duration_tokens):
        pitch_vecs = self.pitch_emb(pitch_tokens)
        dur_vecs = self.duration_emb(duration_tokens)
        note_repr = torch.cat([pitch_vecs, dur_vecs], dim=-1)
        melody_repr = note_repr.mean(dim=1)
        melody_embs = self.mlp(melody_repr)
        return melody_embs

###############################################
# CLMP Module
###############################################
class CLMPModel(nn.Module):
    def __init__(self, clap_model, melody_encoder, final_dim=512):
        super(CLMPModel, self).__init__()
        self.audio_model = clap_model.audio_model
        self.text_model = clap_model.text_model
        self.melody_encoder = melody_encoder

        audio_hidden = self.audio_model.config.hidden_size
        text_hidden = self.text_model.config.hidden_size

        self.audio_proj = nn.Linear(audio_hidden, final_dim)
        self.text_proj = nn.Linear(text_hidden, final_dim)
        self.melody_proj = nn.Linear(32, final_dim)
        self.emotion_proj = nn.Linear(7, final_dim)

    def forward(self, audio_features, input_ids, attention_mask, pitch_tokens, duration_tokens, emotion_logits):
        audio_outputs = self.audio_model(**audio_features)
        audio_emb = audio_outputs.last_hidden_state.mean(dim=(2, 3))
        audio_emb = self.audio_proj(audio_emb)

        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_emb = text_outputs.last_hidden_state.mean(dim=1)
        text_emb = self.text_proj(text_emb)

        melody_emb = self.melody_encoder(pitch_tokens, duration_tokens)
        melody_emb = self.melody_proj(melody_emb)

        emotion_emb = self.emotion_proj(emotion_logits)

        return audio_emb, text_emb, melody_emb, emotion_emb

###############################################
# Symmetric Triple-Contrastive Loss
###############################################
class ExtendedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ExtendedContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, audio_embs, text_embs, melody_embs, emotion_embs):
        a = F.normalize(audio_embs, p=2, dim=1)
        t = F.normalize(text_embs, p=2, dim=1)
        m = F.normalize(melody_embs, p=2, dim=1)
        e = F.normalize(emotion_embs, p=2, dim=1)

        batch_size = a.size(0)
        labels = torch.arange(batch_size, device=a.device)

        sim_a_t = (a @ t.T) / self.temperature
        sim_a_m = (a @ m.T) / self.temperature
        sim_a_e = (a @ e.T) / self.temperature
        sim_t_m = (t @ m.T) / self.temperature
        sim_t_e = (t @ e.T) / self.temperature
        sim_m_e = (m @ e.T) / self.temperature

        loss_a_t = F.cross_entropy(sim_a_t, labels)
        loss_a_m = F.cross_entropy(sim_a_m, labels)
        loss_a_e = F.cross_entropy(sim_a_e, labels)
        loss_t_m = F.cross_entropy(sim_t_m, labels)
        loss_t_e = F.cross_entropy(sim_t_e, labels)
        loss_m_e = F.cross_entropy(sim_m_e, labels)

        loss = (loss_a_t + loss_a_m + loss_a_e + loss_t_m + loss_t_e + loss_m_e) / 6.0
        return loss

###############################################
# Dataset
###############################################
class AudioTextMelodyDataset(Dataset):
    def __init__(self, audio_dir, melody_dir, captions_file, processor, tokenizer, emotion_tokenizer, emotion_model, max_length=128):
        self.audio_dir = audio_dir
        self.melody_dir = melody_dir
        self.processor = processor
        self.tokenizer = tokenizer
        self.emotion_tokenizer = emotion_tokenizer
        self.emotion_model = emotion_model  # Keep the model reference
        self.max_length = max_length

        with open(captions_file, 'r', encoding='utf-8') as f:
            self.captions = [l.strip() for l in f if l.strip()]

        self.audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
        self.melody_files = sorted([f for f in os.listdir(melody_dir) if f.endswith('.mid')])

        assert len(self.captions) == len(self.audio_files) == len(self.melody_files), "Mismatch in data lengths."

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.audio_files[idx])
        waveform, sr = torchaudio.load(audio_path)
        if sr != 48000:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=48000)
            waveform = resampler(waveform)
        waveform = waveform.mean(dim=0, keepdim=True)

        audio_features = self.processor.feature_extractor(
            waveform.squeeze(0).numpy(),
            sampling_rate=48000,
            return_tensors="pt",
        )
        audio_features = {k: v.squeeze(0) for k, v in audio_features.items()}

        melody_path = os.path.join(self.melody_dir, self.melody_files[idx])
        pitch_tokens, duration_tokens = self.extract_melody_tokens(melody_path)

        caption = self.captions[idx]
        text_encoded = self.tokenizer(
            caption, truncation=True, max_length=self.max_length, padding='max_length', return_tensors='pt'
        )
        input_ids = text_encoded['input_ids'].squeeze(0)
        attention_mask = text_encoded['attention_mask'].squeeze(0)

        # Process emotion logits on CPU to avoid CUDA errors in subprocesses
        with torch.no_grad():
            emotion_encoded = self.emotion_tokenizer(caption, return_tensors='pt')
            emotion_logits = self.emotion_model(**emotion_encoded).logits.cpu()

        return {
            'audio_features': audio_features,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pitch_tokens': pitch_tokens,
            'duration_tokens': duration_tokens,
            'emotion_logits': emotion_logits.squeeze(0),
        }

    def extract_melody_tokens(self, melody_path):
        # Implement melody token extraction (unchanged)
        import pretty_midi
        midi_data = pretty_midi.PrettyMIDI(melody_path)
        notes = []
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    pitch = note.pitch
                    duration = int((note.end - note.start) / 6.3 * 511)
                    duration = max(0, min(duration, 511))
                    notes.append((pitch, duration))

        N = 32
        if len(notes) > N:
            notes = notes[:N]
        while len(notes) < N:
            notes.append((0, 0))

        pitches = [n[0] for n in notes]
        durations = [n[1] for n in notes]

        pitch_tokens = torch.tensor(pitches, dtype=torch.long)
        duration_tokens = torch.tensor(durations, dtype=torch.long)
        return pitch_tokens, duration_tokens


###############################################
# Training
###############################################
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        audio_features = {k: v.to(device) for k, v in batch['audio_features'].items()}
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        pitch_tokens = batch['pitch_tokens']
        duration_tokens = batch['duration_tokens']
        emotion_logits = batch['emotion_logits']

        optimizer.zero_grad()
        audio_emb, text_emb, melody_emb, emotion_emb = model(
            audio_features, input_ids, attention_mask, pitch_tokens, duration_tokens, emotion_logits
        )
        loss = criterion(audio_emb, text_emb, melody_emb, emotion_emb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def main():
    mp.set_start_method("spawn", force=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")
    clap_model = AutoModel.from_pretrained("laion/clap-htsat-unfused").to(device)
    tokenizer = processor.tokenizer
    emotion_tokenizer = AutoTokenizer.from_pretrained("nateraw/bert-base-uncased-emotion")
    emotion_model = AutoModelForSequenceClassification.from_pretrained("nateraw/bert-base-uncased-emotion").to(device)

    melody_encoder = MelodyEncoder()
    clmp_model = CLMPModel(clap_model, melody_encoder, final_dim=512).to(device)

    dataset = AudioTextMelodyDataset(
        audio_dir="/root/m2music/data/audio",
        melody_dir="/root/m2music/data/melodies",
        captions_file="/root/m2music/data/captions.txt",
        processor=processor,
        tokenizer=tokenizer,
        emotion_tokenizer=emotion_tokenizer,
        emotion_model=emotion_model  # Pass emotion_model here
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True)

    criterion = ExtendedContrastiveLoss(temperature=0.07)
    optimizer = torch.optim.AdamW(clmp_model.parameters(), lr=3e-4)

    checkpoint_dir = "clm_checkpoint"
    os.makedirs(checkpoint_dir, exist_ok=True)

    num_epochs = 10
    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(clmp_model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

    # Save the final checkpoint
    final_checkpoint_path = os.path.join(checkpoint_dir, f"clmp_final.pth")
    torch.save({
        'model_state_dict': clmp_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss
    }, final_checkpoint_path)
    print(f"Saved final model checkpoint to {final_checkpoint_path}")

if __name__ == "__main__":
    main()

