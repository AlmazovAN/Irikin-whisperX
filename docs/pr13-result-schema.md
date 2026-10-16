# PR13 — единый формат результатов WhisperX (JSON/TXT)

Этап 1, чат 02. PR13 зависит от PR12 (alignment/diarization). Используем существующие
`Segment` и `Word` из WhisperX без дополнительного ASR или повторной обработки звука.

## Контракт JSON, schema_version=1

```json
{
  "schema_version": 1,
  "source_name": "meeting.wav",
  "model_size": "small",
  "language": "ru",
  "elapsed_seconds": 3.25,
  "diarization_requested": true,
  "diagnostic_only": false,
  "segments": [
    {
      "start": 1.0,
      "end": 2.2,
      "text": "Добрый день",
      "speaker": "SPEAKER_00",
      "words": [
        {"word": "Добрый", "start": 1.1, "end": 1.5, "speaker": "SPEAKER_00", "confidence": 0.0}
      ]
    }
  ]
}
```

Поля сегмента: обязательные `start`, `end`, `text`, `speaker` (строка или null).
Список `words` опционален; каждое слово содержит `word` и при наличии
`start`, `end`, `speaker`, `confidence` (0..1). Значение `confidence=0.0`
сохраняется как число, а не трактуется как отсутствие оценки.
При отсутствии части word timestamps они **не выдумываются**.

Метаданные содержат только базовое имя файла, модель, язык, время работы
и признак диагностического запуска без diarization. Системные пути,
HF_TOKEN и объекты модели в JSON не записываются.

## Особенности Legacy

- В `Segment` удалена только вторая декларация `speaker` без изменения значения
  поля и сериализации Pydantic.
- Старый Gradio/subtitle writer продолжает работать с **неизменёнными** исходными
  объектами Segment. Важно: он по-прежнему может получать текст вида
  `SPEAKER_00|Добрый день`.
- Новый `normalize_segments()` создаёт отдельную копию данных; согласованный
  префикс `SPEAKER_00|` переносится в `speaker`, а из `text` удаляется.
  Несовпадающий с явным полем speaker префикс не удаляется, чтобы избежать
  незаметного искажения исходных данных. Legacy `None|` удаляется только
  если speaker отсутствует. Если в сегменте нет speaker, но у всех
  размеченных слов одинаковая метка, она наследуется сегментом.
- Пустые сегменты без текста исключаются из нового результата; если текста
  в записи нет совсем, экспорт заканчивается явной ошибкой `EMPTY_TRANSCRIPT`.

## TXT и экспорт

```text
[00:00:01.000 --> 00:00:02.200] SPEAKER_00: Добрый день
[00:00:03.000 --> 00:00:04.300] SPEAKER_01: Здравствуйте
```

Интерфейс вызова:

```python
from modules.jobs import build_result_payload, render_json, render_txt, write_result_files

result = service.transcribe(validated_input)
payload = build_result_payload(result)
json_preview = render_json(payload)
txt_preview = render_txt(payload)
files = write_result_files(result, trusted_existing_directory)  # unique result-<uuid>.json/.txt
```

`write_result_files` — **явный** разовый экспорт в доверенную уже существующую
директорию. Используется режим создания без перезаписи (`x`); если второй файл
не удалось создать, вновь созданные файлы удаляются. Это не транзакционное
хранилище с восстановлением после сбоя питания. Durable job storage и
атомарная структура каталога задания относятся к следующему пакету PR.

Тесты не импортируют тяжёлые ML-зависимости и не загружают веса. Реальная
двухголосная обработка и проверка Windows/CPU остаются отдельной аппаратной
приёмкой. Сохранение текста на диск выполняется только по явному вызову.
