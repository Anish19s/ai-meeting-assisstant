import torch

def diarize(audio_path, HF_TOKEN):

    from faster_whisper import WhisperModel
    from pyannote.audio import Pipeline   # 👈 MOVE HERE

    device = "cuda" if torch.cuda.is_available() else "cpu"

    whisper = WhisperModel(
        "small",
        device=device,
        compute_type="float16" if device == "cuda" else "int8"
    )

    segments, info = whisper.transcribe(audio_path, beam_size=5)

    transcript_segments = [
        {"start": s.start, "end": s.end, "text": s.text.strip()}
        for s in segments
    ]

    pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.0",
    token=HF_TOKEN
    )

    import torchaudio

    waveform, sr = torchaudio.load(audio_path)

    diarization = pipeline({"waveform": waveform, "sample_rate": sr})

    speaker_segments = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })

    final_transcript = []

    for t in transcript_segments:
        best_speaker = "UNKNOWN"
        max_overlap = 0

        for s in speaker_segments:
            overlap = max(0, min(t["end"], s["end"]) - max(t["start"], s["start"]))
            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = s["speaker"]

        final_transcript.append(f"{best_speaker}: {t['text']}")

    return "\n".join(final_transcript)