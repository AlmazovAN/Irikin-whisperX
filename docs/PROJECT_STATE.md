# Irikin-whisperX — состояние разработки

Последняя проверка репозитория: 2026-10-16.
Базовая ветка: `master`.
Исходный HEAD до P01: `f8d58d1c0af3cfe80d30b8bada4a4a790788a9a9`.

## Этап 0 — фундамент Windows

**Кодовый пакет закрыт: 2026-10-16.** Все PR #1–#6 последовательно
слиты в `master` методом merge, без конфликтов. Полная приёмка
Windows/PyTorch/WhisperX, как договорились, переносится на MVP.
**GitHub Actions остаётся неподтверждённым: 0 workflow runs после слияний.**
Проверки, которых не было, не считаются успешными.


| ID плана | Статус | Ветка | Результат |
| --- | --- | --- | --- |
| P01 | [MERGED, PR #1](https://github.com/AlmazovAN/Irikin-whisperX/pull/1); CI не подтверждён | `test/p01-fast-ci` | Быстрые автономные тесты, облегчённый CI, аудит существующих зависимостей |
| P02 | [MERGED, PR #2](https://github.com/AlmazovAN/Irikin-whisperX/pull/2); Windows-прогон отложен | `feature/p02-hardware-preflight` | CLI-диагностика Windows/Python/FFmpeg/GPU/VRAM/PyTorch/HF и автономные тесты |
| P03 | [MERGED, PR #3](https://github.com/AlmazovAN/Irikin-whisperX/pull/3); Windows-тесты отложены | `feature/p03-windows-install` | CPU/NVIDIA wheel-профили, кандидатные pin-версии, защитный установщик, unit-тесты |
| P04 | [MERGED, PR #4](https://github.com/AlmazovAN/Irikin-whisperX/pull/4); live-тесты отложены | `fix/p04-local-security` | Loopback Gradio/FastAPI, безопасные пути, очистка секретов YAML/Task, .env |
| P05 | [MERGED, PR #5](https://github.com/AlmazovAN/Irikin-whisperX/pull/5); Windows-приёмка отложена | `feature/p05-windows-launcher` | Start.bat, одиночный сервер Gradio, preflight, HTTP readiness, локальные логи, тесты |
| P06 | [MERGED, PR #6](https://github.com/AlmazovAN/Irikin-whisperX/pull/6); smoke на Windows не подтверждён | `test/p06-smoke-adr` | CLI smoke WhisperX Small ru, диагностический отчёт, ADR-001, финальный аудит этапа |

### Не проверено на P01

- Реальная установка Python/PyTorch/WhisperX/Pyannote на Windows и аппаратная
  совместимость; это P02–P03.
- Реальный inference, alignment, diarization и расход VRAM; это P06.
- Готовность CI считается подтверждённой только после фактического результата
  GitHub Actions. Локальные автономные проверки и GitHub Actions фиксируются отдельно.

### Границы P01

Изменения касаются только тестовой инфраструктуры и документации.
Контракты `app.py`, WhisperX pipeline, схемы SQLite и настройки распознавания
не изменены. Следующий этап разработки после закрытия P01–P06: `P07–P12 — MVP ядро`.

### P02 — замечания перед приёмкой

- Исторически P02 был создан поверх P01. При закрытии пакета PR #2 перенаправлен на `master` после слияния PR #1; все зависимости сохранены.
- Linux-тесты диагностики 15/15 проходят в изолированной рабочей копии на Python 3.13. Это не подтверждение запуска на Windows 11 пользователя.
- Для пользовательской приёмки требуется запуск `python scripts\\diagnose_environment.py --json` на целевом ПК и оценка отчёта (секреты в отчёте не выводятся).
- Проверка Pyannote HF-gated лицензий и реальный ASR остаются для P06; P02 не загружает модели.

### P03 — installation candidate

- Историческая базовая ветка P03: `feature/p02-hardware-preflight`; при слиянии PR #3 направлен в `master` после PR #2.
- Версии torch/torchaudio/torchvision (2.7.1/2.7.1/0.22.1) и whisperx==3.4.2 согласованы с опубликованными требованиями; реальный Windows-прогон пока **не подтверждён**.
- Для ноутбука AMD Ryzen 5 5500U / Radeon / 8 ГБ RAM по умолчанию выбран **CPU**, CUDA только по запросу и при обнаружении NVIDIA.
- `Install.bat` не удаляет прежний `venv`; повтор того же профиля разрешён только с `--reuse-venv` при наличии marker-файла.
- Основные прямые зависимости закреплены, но полного lock с транзитивными хешами нет. Проверка `pip check` и реальное распознавание нужны для завершения P03/P06.
- Следующие действия: аудит diff/тестов, фактический `Install.bat --profile cpu --dry-run` на Windows; после осознанного согласования — установка и протокол.

### P04 — локальная безопасность

- Историческая база: `feature/p03-windows-install`; PR #1–#3 слиты до PR #4.
- Добавлена защита loopback-only, убраны `eval()` и shell `start <path>`.
- При сохранении YAML и Task секреты удаляются; существующая БД **не изменялась**.
- Файл `backend/records.db` уже отслеживается Git и **не удалён**: `.gitignore` не отменяет ранее добавленный файл; миграцию выполнять отдельно после аудита данных.
- Автоматическая проверка на Windows/Gradio/FastAPI и реальная инференция ещё не подтверждены.

### P05 — Windows Launcher (код слит)

- Историческая база P05: `fix/p04-local-security`; при слиянии PR #5 перенаправлен в `master` после PR #4.
- `Start.bat` является рекомендуемой точкой входа. `start-webui.bat` вызывает тот же launcher.
- Проверяет `venv`, P02 диагностику (`--skip-torch`), FFmpeg, базовые пакеты, занятый порт и локальный HTTP.
- В отдельном процессе под тем же окном консоли работает **только один** `app.py`; отдельный FastAPI не поднимается.
- Вывод/ошибки записываются в `logs/`; launcher не устанавливает или не удаляет зависимости.
- Локальная изолированная проверка `tests/unit/test_windows_launcher.py`: **26 passed**; Python 3.13 Linux, с mocked subprocess/HTTP, без настоящего WebUI. **Это не проверка Windows CMD/CTRL_BREAK**.
- Файл `.github/workflows/ci.yml` теперь допускает CI на stacked PR (любая целевая ветка). Фактический статус GitHub Actions проверять отдельно.
- Полное функциональное тестирование на Windows и реальный аудио smoke-test остаются до P06 / приёмки этапа.

### P06 — ручной smoke-test, архитектура и закрытие кодового пакета (код слит)

- Историческая база P06: `feature/p05-windows-launcher`. В ходе закрытия пакета PR #6 перенаправлен в `master` после слияния PR #5.
- `scripts/smoke_whisperx.py` использует **существующий** `WhisperXInference.run()`, модель `small`, русский `ru`, `transcribe`, VAD + alignment + требуемые speaker labels. Реальный тест требует частный WAV/MP3 и доступ к Pyannote.
- `--diagnose-only` без скачивания моделей и без обработки файла; критерии `passed` / `partial` / `blocked` / `failed` явно различены. В отчёте нет текста распознавания и токена.
- `tests/unit/test_smoke_whisperx.py` содержит **автономные тесты** валидации, отчёта, ошибочных случаев и mock-проверку прямого вызова унаследованного конвейера. Проверка в CI и на целевой Windows **пока не подтверждена**.
- ADR-001 принят как дизайн: один Python-сервер, общий сервис заданий, один inference-owner/worker, durable SQLite и atomic артефакты. Реализация JobService, TXT/JSON и независимости от вкладки запланирована на P07–P25.
- Ручной real-audio smoke на ноутбуке Windows/Ryzen/8 ГБ **NOT_RUN**, аппаратная/пакетная совместимость `small` + alignment + Pyannote **UNKNOWN**. Пользовательская приёмка по текущему процессу отложена до MVP.
- Итоговый файл передачи: [`docs/stage0-acceptance.md`](stage0-acceptance.md). Для теста: [`docs/smoke-test-windows.md`](smoke-test-windows.md). Для архитектуры: [`docs/adr/ADR-001-single-worker.md`](adr/ADR-001-single-worker.md).

### Слияния и проверка этапа 0 (2026-10-16)

| PR | Merge commit в `master` | GitHub |
| --- | --- | --- |
| #1 | `04496a95ba1a3f9872d96c46d96921104f1abc93` | merged |
| #2 | `d46cc07c38d26c7feb3e52755f22468a7b51cb66` | merged |
| #3 | `90b11ff5c0bd09fe828cda957124d07b057a391f` | merged |
| #4 | `025a87443f9a518b90f3191945f7f20d83b53a3e` | merged |
| #5 | `4c431d4d619d58907ec7bd079831e6ec9ccadd99` | merged |
| #6 | `c1fb54f6ed1a0fc23a9cbc1cdef3e5c85666a511` | merged |

После слияния `master` и исходная ветка P06 имеют **одинаковые
деревья файлов** (GitHub compare: 0 изменённых файлов).
Локальный частичный прогон Python 3.13/Linux: автономные тесты
диагностики P02 **15/15**, launcher P05 **26/26** (mock,
без Windows-моделей). **Пакетный прогон `pytest tests/unit` на
полном `master` не выполнен**; тесты P03/P04/P06 не подтверждены
инфраструктурой GitHub Actions. Запусков Actions: **0**.
`pip check` целевого `venv`, реальный Windows smoke, Pyannote и
сквозной MVP не проверены.

**Вывод:** разработка кода этапа 0 завершена и интегрирована.
Это не эксплуатационная приёмка продукта; аппаратные проверки
и устранение проблемы CI остаются входными условиями QA MVP.

## Этап 1 — MVP core, фактические слияния (2026-10-16)

План P07–P12 выполнен и объединён в master последовательными GitHub PR
#7, #8, #10, #11, #12, #13. PR #9 был промежуточным исправлением CI.
На старте нового пакета master = bc869242ef1349912da3c321df9f18cfb63b53d7.

| ID плана | GitHub PR | Состояние |
| --- | --- | --- |
| P07 | #7 | MERGED — MVP Small/ru preset |
| P08 | #8 | MERGED — валидируемые входные файлы, FFprobe |
| P09 | #10 | MERGED — headless TranscriptionService |
| P10 | #11 | MERGED — VAD timestamps, ru/transcribe |
| P11 | #12 | MERGED — alignment/Pyannote error diagnostics |
| P12 | #13 | MERGED — единый JSON/TXT и speaker normalization |

На указанном master GitHub Actions CI успешно завершился на Python
3.10, 3.11 и 3.12. Это только автономные тесты, без Windows/CPU smoke
и без реальной загрузки WhisperX/Pyannote. Docker publishing workflows
падают, но в локальный MVP не входят.

## Чат 03 — очередь и сохранность (P13–P19)

### P13 = GitHub PR #14

- Ветка: feature/pr14-task-schema-migrations.
- База: bc869242ef1349912da3c321df9f18cfb63b53d7.
- Статус на момент фиксации: OPEN; не считать MERGED до фактического слияния.
- Контракт: Task stage/attempt/private artifact paths/model/timestamps/error_code;
  SQLite migration v1, backup старой базы, миграционные unit-тесты.
- Подробности: docs/pr14-task-schema.md.
- Никакого нового worker, upload persistence или UI в P13 нет.
- CI первой редакции ветки: success Python 3.10/3.11/3.12;
  после изменения документации перепроверить head и CI.
- Не изменялись: backend/records.db (tracked), WhisperX pipeline, audio.
- Windows/CPU/real ASR: NOT_RUN.

### Следующая работа

После принятия P13 начать P14 по плану (предположительно GitHub PR #15):
атомарный приём входа в data/jobs/<uuid>/input с отметкой QUEUED,
не полагаясь на временный путь Gradio. Не запускать inference из upload.

### P14 (GitHub PR #15): durable audio intake

- Базовый master: 1b862556a5bd29cccfdd07297b4eef889d3da07e.
- Ветка: feature/pr15-durable-intake.
- Добавлены modules/jobs/storage.py и modules/jobs/service.py:
  InputSource → P08 FFprobe → data/jobs/<uuid>/input → SQLite QUEUED.
- Новая локальная БД data/jobs/jobs.sqlite3; старый backend/records.db не изменён.
- Предусмотрены атомарная публикация файла, повторная проверка,
  очистка при известном сбое до принятия, безопасные UUID и отсутствие
  секретов/путей в JobTicket.
- Подробности docs/pr15-durable-intake.md.
- Статус PR15 и CI всегда проверять по GitHub; не считать merge доказанным
  лишь по наличию этих строк в ветке.
- Реальный Windows/CPU/Gradio upload: NOT_RUN.
- Следующий функциональный PR: P15 плана / GitHub PR #16 — один worker FIFO.

### P15 (GitHub PR #16): один worker + FIFO

- Base SHA: cb7f3eef29a3b64f19245c7d4015223ddecd807d.
- Ветка: feature/pr16-single-fifo-worker.
- Добавлен modules/jobs/worker.py: один lock-holder, один процесс
  обработки в фоне (thread), FIFO SQLite, запуск TranscriptionService.
- Расширен JobService: list_jobs, отмена только QUEUED, retry FAILED.
- Результаты PR13 writer создаются до COMPLETED; P17 ещё должен
  обеспечить полную атомарность артефактов.
- Worker пока **не** включён одновременно с legacy Gradio App, чтобы не
  создать два независимых ASR-owner. См. docs/pr16-single-worker.md.
- Автоматические тесты tests/unit/test_single_job_worker.py.
- Проверять фактический MERGED / CI на GitHub: запись в ветке не доказательство.
- Windows/CPU/Pyannote и browser E2E NOT_RUN.
- Следующий PR: план P16 / GitHub PR #17 — стадии и статус polling.

### P16 (GitHub PR #17) — progress и status polling

- База: 48fba5a5940c9925a477e623ccbd319c16bf2b16.
- Ветка: feature/pr17-persisted-stage-status.
- HeadlessProgress распознаёт реальные VAD, ASR, ALIGN, DIARIZE события;
  worker добавляет PREPARING и EXPORT.
- SQLite сохраняет монотонный progress + stage для конкретной попытки.
  JobService.get_status(uuid) и list_jobs() дают свежие безопасные snapshot.
- `progress` — условные точки, не процент работы по времени и не ETA.
- Миграции и существующий WhisperX pipeline не изменены.
- Обновления UI/REST и реальный Windows smoke: НЕ выполнены.
- Детали: docs/pr17-stage-progress.md, tests/unit/test_persisted_stage_progress.py.
- GitHub CI и MERGED проверять фактически; запись в ветке не подтверждение merge.
- Далее: P17 дорожной карты / GitHub PR #18 — атомарные результаты, retention.

### P17 (GitHub PR #18): atomic transcript storage / retention

- База PR17: 3aa6589c606620f8769049ae3695f8ea0d6c69fc.
- Ветка: feature/pr18-atomic-results-retention.
- Вместо отдельного JSON/TXT writer для durable job используется атомарная
  публикация каталога result/attempt-N с manifest.json и SHA-256.
- `COMPLETED` присваивается только после проверки опубликованной пары;
  локальный `get_artifact(uuid, format)` проверяет status, БД и checksum.
- Retention: сохранять до явного удаления. Никакой фоновой чистки личных
  аудио/стенограмм по дате нет; orphan-pending оставляем для P18 recovery.
- Историческая `backend/records.db`, Gradio UI и inference неизменны.
- Подробности: docs/pr18-atomic-artifacts.md.
- GitHub CI / MERGED смотреть по фактическому статусу PR, не по тексту ветки.
- Windows/NTFS, real WhisperX/Pyannote, отключение питания — NOT_RUN.
- Далее P18 дорожной карты / GitHub PR19: restart recovery.

### P18 (GitHub PR #19): restart recovery

- База: 356791b4e4f5e1cdc5e7e2158b34980d708eca3b.
- Ветка: feature/pr19-restart-recovery.
- JobWorker.start() под .worker.lock сверяет IN_PROGRESS с проверенными
  result/attempt-N, восстанавливает COMPLETED либо ставит INTERRUPTED.
- Нет автоматического повторного ASR: retry INTERRUPTED — только явно
  и при существующем аудио, следующая попытка получает новый attempt.
- SQLite транзакция целиком rollback при сбое; QUEUED продолжаются
  строго FIFO, завершённые/отменённые/ошибочные задачи не переписываются.
- RecoveryReport считает orphan каталоги и незавершённые наборы файлов;
  не удаляет ничего автоматически и не раскрывает частных данных.
- Подробности: docs/pr19-restart-recovery.md.
- CI/MERGED подтверждать по GitHub, не по записи в этой ветке.
- Windows/NTFS, real WhisperX/Pyannote, аварийное отключение питания: NOT_RUN.
- Далее P19 плана / GitHub PR #20: cancellation/error regression и gate.
