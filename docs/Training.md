# Training

Текущий репозиторий содержит baseline training scaffold.

## Face model

`src/face_model/train_face_emotion.py` принимает JSON dataset и сохраняет:

- class priors;
- sample counts;
- baseline metrics manifest;
- checkpoint JSON.

Это совместимо с дальнейшей заменой на PyTorch ResNet18 fine-tuning loop.

## План production fine-tune

1. собрать manifest для FER2013 / AffectNet / русских локальных данных;
2. реализовать dataset class и augmentation;
3. обучить ResNet18 (224x224, AdamW, lr=1e-4, batch_size=16, epochs=30, early stopping=5);
4. экспортировать best checkpoint в ONNX.

## Voice model

`src/voice_emotion.py` пока использует prosody heuristics. Следующий шаг — заменить на CNN/GRU или Aniemore/SpeechBrain backend.
