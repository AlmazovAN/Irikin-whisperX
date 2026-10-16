# Irikin-whisperX — состояние разработки

Последняя проверка репозитория: 2026-10-16.
Базовая ветка: `master`.
Исходный HEAD до P01: `f8d58d1c0af3cfe80d30b8bada4a4a790788a9a9`.

## Этап 0 — фундамент Windows

| ID плана | Статус | Ветка | Результат |
| --- | --- | --- | --- |
| P01 | [OPEN, PR #1](https://github.com/AlmazovAN/Irikin-whisperX/pull/1); CI не подтверждён | `test/p01-fast-ci` | Быстрые автономные тесты, облегчённый CI, аудит существующих зависимостей |
| P02 | [OPEN, PR #2](https://github.com/AlmazovAN/Irikin-whisperX/pull/2); ожидает Windows-прогона | `feature/p02-hardware-preflight` | CLI-диагностика Windows/Python/FFmpeg/GPU/VRAM/PyTorch/HF и автономные тесты |
| P03 | [DRAFT, PR #3](https://github.com/AlmazovAN/Irikin-whisperX/pull/3); ожидает Windows-тестов | `feature/p03-windows-install` | CPU/NVIDIA wheel-профили, кандидатные pin-версии, защитный установщик, unit-тесты |
| P04 | [DRAFT, PR #4](https://github.com/AlmazovAN/Irikin-whisperX/pull/4); ожидает тестирования | `fix/p04-local-security` | Loopback Gradio/FastAPI, безопасные пути, очистка секретов YAML/Task, .env |
| P05 | PLANNED | — | Windows launcher |
| P06 | PLANNED | — | Реальный smoke-test и ADR |

### Не проверено на P01

- Реальная установка Python/PyTorch/WhisperX/Pyannote на Windows и аппаратная
  совместимость; это P02–P03.
- Реальный inference, alignment, diarization и расход VRAM; это P06.
- Готовность CI считается подтверждённой только после фактического результата
  GitHub Actions. Локальные автономные проверки и GitHub Actions фиксируются отдельно.

### Границы P01

Изменения касаются только тестовой инфраструктуры и документации.
Контракты `app.py`, WhisperX pipeline, схемы SQLite и настройки распознавания
не изменены. Следующий PR после слияния P01: `P02 — hardware preflight`.

### P02 — замечания перед приёмкой

- P02 создан поверх ещё открытой ветки P01, поэтому слияние в `master` допускается только после P01. GitHub PR P02 направлен в ветку P01.
- Linux-тесты диагностики 15/15 проходят в изолированной рабочей копии на Python 3.13. Это не подтверждение запуска на Windows 11 пользователя.
- Для пользовательской приёмки требуется запуск `python scripts\\diagnose_environment.py --json` на целевом ПК и оценка отчёта (секреты в отчёте не выводятся).
- Проверка Pyannote HF-gated лицензий и реальный ASR остаются для P06; P02 не загружает модели.

### P03 — installation candidate

- Базовая ветка P03: `feature/p02-hardware-preflight`; обе предыдущие PR пока OPEN.
- Версии torch/torchaudio/torchvision (2.7.1/2.7.1/0.22.1) и whisperx==3.4.2 согласованы с опубликованными требованиями; реальный Windows-прогон пока **не подтверждён**.
- Для ноутбука AMD Ryzen 5 5500U / Radeon / 8 ГБ RAM по умолчанию выбран **CPU**, CUDA только по запросу и при обнаружении NVIDIA.
- `Install.bat` не удаляет прежний `venv`; повтор того же профиля разрешён только с `--reuse-venv` при наличии marker-файла.
- Основные прямые зависимости закреплены, но полного lock с транзитивными хешами нет. Проверка `pip check` и реальное распознавание нужны для завершения P03/P06.
- Следующие действия: аудит diff/тестов, фактический `Install.bat --profile cpu --dry-run` на Windows; после осознанного согласования — установка и протокол.

### P04 — локальная безопасность

- База: `feature/p03-windows-install`; PR #1–#3 до сих пор OPEN.
- Добавлена защита loopback-only, убраны `eval()` и shell `start <path>`.
- При сохранении YAML и Task секреты удаляются; существующая БД **не изменялась**.
- Файл `backend/records.db` уже отслеживается Git и **не удалён**: `.gitignore` не отменяет ранее добавленный файл; миграцию выполнять отдельно после аудита данных.
- Автоматическая проверка на Windows/Gradio/FastAPI и реальная инференция ещё не подтверждены.
