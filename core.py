import whisper
from whisper import Whisper

def core(audio_path:str)->str:
  device:str = "cuda"
  compute_type:str = "float16"

  model:Whisper = whisper.load_model(name="models/whisper-large-v3.en", device=device, compute_type=compute_type)
  results:str = model.transcribe(audio_path=audio_path, temperature=0.0, word_timestamps=False)

  # delete model if low on GPU resources
  # import gc; gc.collect(); torch.cuda.empty_cache(); del model

  return results["text"].lower()