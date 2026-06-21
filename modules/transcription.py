from transformers import WhisperProcessor,WhisperForConditionalGeneration
import torch

def transcribe(waveform,sr):
    processor=WhisperProcessor.from_pretrained("openai/whisper-small")
    model=WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

    chunks=[]
    step_size=30*sr
    overlap_len=5*sr

    device="cuda" if torch.cuda.is_available() else "cpu"
    model=model.to(device)

    for i in range(0,waveform.shape[0],step_size-overlap_len):
        chunk=waveform[i:i+step_size]
        chunks.append(chunk)

    text=""
    for chunk in chunks:
        inputs=processor(chunk,sampling_rate=16000,return_tensors="pt",return_attention_mask=True)
        input_features=inputs["input_features"].to(device)

        with torch.no_grad():
            predicted_ids=model.generate(input_features)

        transcription=processor.batch_decode(predicted_ids,skip_special_tokens=True)[0]

        text+=transcription


    return text