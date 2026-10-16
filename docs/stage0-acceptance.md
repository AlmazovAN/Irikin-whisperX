# Этап 0 — итоговый аудит разработки P01–P06

**Дата:** 2026-10-16. **Репозиторий:** `AlmazovAN/Irikin-whisperX`.
**Основная ветка:** `master`.
**Статус пакета:** шесть кодовых PR подготовлены или находятся в работе
ревью; **объединение и эксплуатационная приёмка НЕ подтверждены**.

Документ описывает фактически подготовленные изменения и границы
проверки. Наличие файла/ветки или написанного unit-теста
не считается доказательством работы WhisperX на Windows.

## Реестр изменений

| PR | Ветка PR | База PR | Что подготовлено | Состояние |
| --- | --- | --- | --- | --- |
| [#1](https://github.com/AlmazovAN/Irikin-whisperX/pull/1) | `test/p01-fast-ci` | `master` | Автономные unit-tests, быстрый CI, аудит | OPEN; не слит |
| [#2](https://github.com/AlmazovAN/Irikin-whisperX/pull/2) | `feature/p02-hardware-preflight` | P01 | Диагностика Windows, Python, FFmpeg, GPU, HF | OPEN; не слит |
| [#3](https://github.com/AlmazovAN/Irikin-whisperX/pull/3) | `feature/p03-windows-install` | P02 | Профили CPU/CUDA, кандидатные версии, Install.bat | DRAFT; не слит |
| [#4](https://github.com/AlmazovAN/Irikin-whisperX/pull/4) | `fix/p04-local-security` | P03 | Loopback-only, безопасные пути, токены, backend guard | DRAFT; не слит |
| [#5](https://github.com/AlmazovAN/Irikin-whisperX/pull/5) | `feature/p05-windows-launcher` | P04 | Start.bat, диагностика запуска, логи и shutdown | DRAFT; не слит |
| #6 | `test/p06-smoke-adr` | P05 | Реальная smoke-команда, unit-защита от ложного PASS, ADR-001 | PENDING REVIEW; не слит |

**Ветки намеренно зависимые (stacked PR)**. Ветка P06 содержит весь
разрабатываемый код этапа 0. Не следует направлять все шесть PR
независимо в `master` без обновления их баз/последовательного слияния.

## Что проверено, а что нет

| Контроль | Наблюдаемое состояние | Доказательство / ограничение |
| --- | --- | --- |
| Статический аудит репозитория | Выполнен | Изучены `app.py`, WhisperX pipeline/wrapper, P02–P05 |
| CI workflow | Подготовлен | `.github/workflows/ci.yml` из P05 разрешает stacked PR; **прохождение GitHub Actions не подтверждено** |
| Автономные P01/P02/P05 unit-тесты | Ранее сообщалось об отдельных локальных прогонах | Они не удостоверяют интеграцию реальных моделей |
| P03 dependency resolution и установка | Не подтверждены | `pip check` на целевой Windows 11 не запускался |
| P04 реальный Gradio/FastAPI | Не подтверждён | Новые rules/tests не заменяют live-проверку |
| P05 двойной щелчок, Ctrl+C, порт/браузер | Не подтверждены на Windows | Mock-тесты не моделируют Windows CMD/CTRL_BREAK полностью |
| P06 smoke Small `ru/transcribe` | **NOT_RUN** | Нет реального Windows WAV/MP3 / модели; скрипт сам фиксирует BLOCKED/PARTIAL/FAILED/PASSED |
| P06 HF gated доступ / diarization | Не подтверждён | Наличие токена не доказывает принятие условий модели |
| ADR-001 | **Accepted (design)** | `docs/adr/ADR-001-single-worker.md`; worker/queue ещё не реализованы |
| Результаты jobs, TXT/JSON, close-tab | Не реализованы в этапе 0 | Область MVP P07–P25 |

**Недопустимо:** использовать факт `mock tests passed` как
подтверждение ускорения/совместимости WhisperX/Pyannote, результата
диаризации и производительности GPU. Все такие заявления требуют
реального запуска.

## Известные ограничения целевого ПК

Характеристики, предоставленные владельцем, **не измеренные P02**:
Windows 11 Home, Ryzen 5 5500U, интегрированная Radeon, 8 GiB RAM,
64-bit. Базовый путь — CPU + `int8` + batch size 1.

Вопросы, которые остаются открытыми до ручного smoke:
- Установится ли комбинация `whisperx==3.4.2`,
  `torch==2.7.1`, `torchaudio==2.7.1` и
  `pyannote.audio==3.4.0` на конкретном Windows Python.
- Хватает ли 8 ГБ RAM для Small + forced alignment + Pyannote
  без swap/OOM/неприемлемой задержки.
- Реальный compute/device и возможность загрузить gated Pyannote.
- Разрешение зависимостей/моделей при ограничении Интернета и
  защита от неожиданного повторного скачивания весов.

Эти пункты — **risks / pending evidence**, а не уже обнаруженные
фактические ошибки установки.

## Предлагаемые автоматические проверки до слияния

```bat
venv\Scripts\python.exe -m compileall -q app.py backend modules scripts tests
venv\Scripts\python.exe -m pytest -q tests/unit
venv\Scripts\python.exe -m pip check
```

Запуск `pytest` без интеграционных маркеров не скачивает модели.
Для Windows-прогона smoke см. [P06](smoke-test-windows.md).
При подтверждении успешных проверок зафиксировать дату, версию
Python и commit SHA прямо в этом документе. Не добавлять сюда
частные аудио или значения HF_TOKEN.

## Итоговый gate

- **Кодовая часть P01–P06:** подготовлена к ревью.
- **Чистая Windows-установка:** pending.
- **Smoke + фактическое GPU/CPU/RAM:** pending.
- **CI GitHub Actions:** pending.
- **Ручная приёмка владельцем:** по согласованному процессу
  отложена до пригодного MVP, если не возникнет явный
  технологический блокер.
- **Слияние:** пока не выполнялось; после проверок, ревью
  и согласования следует обновить target branches и
  последовательно объединить #1→#2→#3→#4→#5→#6.

Не считать этап 0 промышленно готовым только по факту шести PR.

## Передача в следующий чат: P07–P12

Предусловия: принять [ADR-001](adr/ADR-001-single-worker.md) как
архитектурное ограничение; продолжать разработку от суммарного
состояния P06 (или от слитого `master` при завершённом review).
Не использовать одновременно независимые Gradio и FastAPI
inference-модели. Для P07 создать **один** MVP preset
`small/ru/transcribe/VAD/alignment/diarization`, далее файл,
проверка результата, JSON/TXT writer. Не реализовывать вторую очередь
или публичный сервер, не добавлять Drive/диктофон до следующих этапов.

После появления MVP восстановить полный acceptance gate:
аудио WAV/MP3, настоящий `SPEAKER_*`, word timestamps,
TXT/JSON на диск, сохраняемый статус после закрытия вкладки,
повторное открытие истории; 10 последовательных прогонов без потерь
в соответствии с исходной дорожной картой.
