# P09 — TranscriptionService без зависимости от браузерной сессии

**Этап 1 / Чат 02 / P09**. Использует модули из P07 (MVP preset) и P08
(ValidatedInput). Код P09 не создаёт новую реализацию ASR, не скачивает модели
при импорте и не меняет старые обработчики Gradio.

## Контракт

~~~python
from modules.jobs import InputSource, validate_input, TranscriptionService

source = validate_input(
    InputSource.local_file("C:/Records/meeting.wav"),
    local_roots=["C:/Records"],  # доверенная конфигурация сервера
)
service = TranscriptionService()
result = service.transcribe(source)
segments = result.segments  # исходные Segment из WhisperX, без потери слов/спикеров
~~~

В показанном вызове исходный файл должен предварительно пройти P08; это
**синхронная** операция, не background task. При вызове используется
`WhisperFactory.create_whisper_inference(whisper_type="whisperx")`
внутри существующего `BaseTranscriptionPipeline.run(...)`. Специальный
`HeadlessProgress` удовлетворяет callable-контракту текущего конвейера
вместо привязки к активной сессии Gradio. Прогресс ASR опционально
передаётся `progress_callback`; проценты/ETA не выдумываются.

Настройки по умолчанию — единый P07 профиль `small/ru/transcribe`,
`VAD/alignment/diarization`, `CPU/int8/batch_size=1`. Передаётся
`persist_settings=False` — отслеживаемый YAML не переписывается.
HF_TOKEN сервис не читает, не записывает и не возвращает: существующий
WhisperX/Pyannote делает это только во время инференции.

## Выход и критерии

`TranscriptionResult` предоставляет:
- `segments: tuple[Segment, ...]` — **те же** объекты Segment, с полями
  `start/end/text/speaker/words`, которые вернул уже существующий pipeline.
- `elapsed_seconds`, `input_name`, `model_size`,
  `language`, `diarization_requested`.
- `diagnostic_only=True` при `diarization_enabled=False`.
  Такой вывод **не является прошедшим приёмку MVP**.

Нет пустого «успеха»: обязательно распознанный текст и валидные времена
сегментов, минимум одно слово с таймкодами alignment, а при включённой
диаризации — минимум одна метка спикера. Отказы кодируются
`TranscriptionServiceError.code`:
`INPUT_CHANGED`, `BUSY`, `PIPELINE_FAILED`,
`EMPTY_TRANSCRIPT`, `INVALID_SEGMENTS`,
`ALIGNMENT_MISSING`, `DIARIZATION_MISSING`,
`INVALID_DURATION`. Подробная классификация ошибок alignment/
Pyannote и тест на реальных спикерах остаются в P11.

## Управление ресурсами и границы PR

Один экземпляр `TranscriptionService` повторно использует один pipeline.
Неблокирующая блокировка отказывает одновременному вызову с `BUSY`,
не создавая параллельный тяжёлый экземпляр. После каждого запуска
используются существующие `offload()` и offload diarizer.
Это **не очередь**; owner worker и durable SQLite будут в P13–P19.

Сервис не делает копию исходника: перед выполнением проверяет существование,
канонический путь, полученный от P08, и сохранение размера. Каталоги он повторно
не авторизует; замена файла на другие данные той же длины не обнаруживается.
Устранение TOCTOU через повторную валидацию при атомарном копировании — P14. Не использовать напрямую с
пользовательским непроверенным путём; требуется P08 `validate_input()`.

P12 реализует нормализацию JSON (`speaker` отдельно от текста,
устранение унаследованных префиксов и согласованный экспорт) — текущий
сервис преднамеренно не меняет исходные Segment.

## Проверки

`python -m pytest -q tests/unit/test_transcription_service.py`:
15 автономных mock-тестов без модели/доступа в Интернет на проверку
параметров, вызова run, типов выходных Segment, диаризации,
ошибок, повторов, отказа параллельному запросу и отсутствия тяжёлых
импортов при старте.

**Не проверено:** фактический WhisperX Small + Pyannote на Windows/Ryzen
с 8 ГБ ОЗУ, качество двухголосной диаризации, полный CI в GitHub,
close-tab/reconnect, persistence и долговременная очередь. Последние
возможности не относятся к P09.
