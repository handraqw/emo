# Evaluation

## Что проверяется сейчас

- корректность правил `CONFLICT`;
- агрегация face/voice/text сигналов;
- экспорт результатов UI pipeline;
- стабильность запуска без внешних ML зависимостей.

## Что нужно для итоговой приёмки

- промаркированный recorded dataset;
- отдельный live-stream protocol;
- macro F1 / accuracy / per-class precision-recall;
- latency/FPS logs на Windows + NVIDIA 4060.
