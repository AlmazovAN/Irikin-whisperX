# P01 — быстрый CI и аудит воспроизводимости

Дата проверки: 2026-10-16. Основание: исходники ветки `master` на SHA
`f8d58d1c0af3cfe80d30b8bada4a4a790788a9a9`.

## Два режима тестирования

| Режим | Как запускать | Что проверяется | Ограничения |
| --- | --- | --- | --- |
| Быстрый CI (по умолчанию) | push/PR в `master` и `intel-gpu` | Python 3.10–3.12: компиляция исходников, проверка bash-синтаксиса, `tests/unit` | Не проверяет загрузку моделей, API/Gradio или реальную транскрибацию |
| Legacy integration (только вручную) | Actions → CI → Run workflow → `run_heavy=true` | Существующие `tests/` (кроме `tests/unit`) и `backend/tests`; полный стек | Могут понадобиться интернет, скачивание моделей и много диска/памяти; GPU на hosted runner не гарантирован |

Локальные быстрые проверки (если установлен Python и pytest):

```sh
python -m compileall -q app.py backend modules scripts tests
python -m pytest -q tests/unit
# По умолчанию "pytest -q" также запускает только tests/unit.
```

Конфигурация `pytest.ini` задаёт `testpaths = tests/unit`; полный старый набор
нужно запускать только с явно указанным путём, например
`python -m pytest -rs tests --ignore=tests/unit`.

Быстрый контур устанавливает **только pytest**, не устанавливает `requirements.txt`,
не импортирует WhisperX/torch/Gradio и не скачивает файлы. Он не подменяет реальный
smoke-test Whisper Small + Pyannote на Windows, запланированный в P06.

### Почему существующий полный набор не подходит для проверки каждого PR

- `tests/test_config.py` определяет `download_file` как `autouse` fixture с `requests.get`
  для удалённого `jfk.wav`. При подключении этого модуля сетевой вызов возможен
  без явного запроса теста; таймаута и проверки HTTP-статуса в fixture нет.
- `tests/test_transcription.py` запускает модели ASR и может обращаться к YouTube;
  `tests/test_diarization.py` и `tests/test_bgm_separation.py` выполняют GPU/model
  испытания; `tests/test_translation.py` использует NLLB и опциональный DeepL.
- Предыдущий CI ставил полный стек для трёх версий Python отдельно для WebUI и backend,
  а shell-job выполнял `timeout 60s ./start-webui.sh || true`, что не выявляло отказ
  команды как ошибку CI.
- Поэтому `python -m pytest tests` **не считается автономным офлайн-тестом**.
  Автоматически проверяется только явная директория `tests/unit`.

### Что осталось на P03: расхождения зависимостей

| Место | Подтверждённое состояние на базовом SHA | Действие |
| --- | --- | --- |
| `requirements.txt` | Задан `--extra-index-url https://download.pytorch.org/whl/cu126`, независимо от фактической видеокарты | Подбор CPU/CUDA-профилей после P02 |
| `requirements.txt` | `torch`, `torchaudio`, `whisperx`, `pyannote.audio`, `gradio` без точных версий | Протестировать и зафиксировать совместимые версии |
| `requirements.txt` | `openai_whisper` указан дважды плюс `openai-whisper` (эквивалентное нормализованное имя пакета) | Убрать дубли после проверки зависимостей |
| `requirements.txt` / `README.md` | README утверждает, что `torchaudio` закреплён на 2.7.1; в файле фактически просто `torchaudio` | Согласовать документацию с проверенным стеком |
| `requirements.txt` | `numpy<2.3` ограничен сверху, но не закреплён полностью | Учесть при фиксировании окружения |
| `backend/requirements-backend.txt` | Подключает общий `requirements.txt` и добавляет backend-пакеты | Исключить ненужный повтор тяжёлой установки |

Это **статический аудит**, не протокол успешной установки. P01 не изменяет
AI-зависимости или поведение рабочего inference-конвейера.

### Маркеры pytest

Зарегистрированы `fast`, `integration`, `gpu`, `network`, `model_download`
для новых и постепенно мигрируемых тестов. Они сами по себе **не предотвращают
импорты тяжёлых пакетов при сборе полной тестовой коллекции**. Именно поэтому
быстрый CI запускает отдельную директорию, а не использует только `-m fast`.
