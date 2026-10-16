# PR12 — WhisperX alignment и Pyannote diarization

**Этап 1, чат 02, GitHub PR #12.** Продолжение PR #11 (русский язык, VAD).

## Контракт

- WhisperX Small, ru/transcribe, VAD, alignment и diarization включены.
- WhisperX alignment получает исходный звук и принудительный ru после восстановления таймкодов VAD.
- Pyannote получает исходный звук и выровненные слова через уже существующий WhisperXWrapper; второй ASR и worker не создаются.
- Для gated diarization нужен HF_TOKEN с доступом read в приватном окружении или локальном .env. Перед запуском нужно принять условия pyannote/speaker-diarization-3.1 и pyannote/segmentation-3.0 на Hugging Face.
- По умолчанию CPU / float32 для Pyannote. CUDA допускается только при реально доступном NVIDIA CUDA. AMD Radeon не является CUDA.

## Машинные коды ошибок

| Код | Пояснение |
|---|---|
| HF_TOKEN_MISSING | Токен не передан — задать в приватной среде |
| HF_MODEL_ACCESS | Не удалось загрузить gated-модель — проверить права и условия использования |
| DEVICE_UNSUPPORTED | Выбрать CPU или NVIDIA CUDA |
| CUDA_UNAVAILABLE | CUDA недоступна — выбрать CPU |
| DIARIZATION_MODEL_LOAD | Ошибка инициализации Pyannote — проверить память и совместимость |
| DIARIZATION_FAILED | Ошибка выполнения Pyannote |
| ALIGNMENT_MODEL_UNAVAILABLE | Не удалось загрузить alignment-модель |
| ALIGNMENT_FAILED | Не удалось выполнить alignment |

TranscriptionService сохраняет эти коды как TranscriptionServiceError без распространения текста исключений сторонних библиотек. Не сохранять токен в YAML, JSON, логах и PR. WhisperXWrapper хранит только отпечаток токена в оперативной памяти для выбора переинициализации. Отсутствие времён выравнивания или speaker label по-прежнему блокирует успешный результат.

## Реальная проверка двух голосов

Короткая собственная запись двух русскоязычных спикеров, WAV/MP3, до 90 секунд:

```powershell
$env:HF_TOKEN = "<private-read-token>"
python scripts/smoke_whisperx.py --audio "C:\Private\two-speakers.wav" --device cpu --compute-type int8 --batch-size 1 --min-speakers 2
```

Требования приёмки: минимум две различные метки SPEAKER_*, непустой текст, словные таймкоды alignment; дополнительно вручную проверить соответствие реплик говорящим по исходному звуку. Число меток не доказывает точность идентификации.

Тесты offline с синтетическими Segment и заглушкой Pyannote проверяют только контракты. Реальный Windows 11 / Ryzen 5 5500U / 8 ГБ RAM и память/время/доступ HF не проверены.

Следующий GitHub PR13 займётся нормализацией speaker/text и экспортом JSON/TXT; очередь и Gradio UI относятся к последующим этапам.
