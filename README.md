# Support Intent Classifier

Отправьте текст банковского обращения на английском, и API на FastAPI вернёт одну из 77 категорий BANKING77. Для классификации используется LinearSVC с TF-IDF по словам и символам. Готовую модель можно запустить сразу или обучить заново по инструкции ниже.

## Запуск в Docker

~~~sh
docker build -t support-intent-classifier .
docker run --rm -p 127.0.0.1:8000:8000 support-intent-classifier
~~~

Контейнер при старте читает [готовую модель](models/support-intents.joblib). После загрузки `GET /health` возвращает `{"status":"ok"}`. Документация API открывается на [localhost:8000/docs](http://localhost:8000/docs).

Откройте второй терминал и отправьте [пример запроса](example-request.json) к `POST /predict`.

~~~sh
curl --fail-with-body \
  -H "Content-Type: application/json" \
  --data-binary "@example-request.json" \
  http://127.0.0.1:8000/predict
~~~

~~~powershell
curl.exe --fail-with-body -H "Content-Type: application/json" --data-binary "@example-request.json" http://127.0.0.1:8000/predict
~~~

В примере клиент пишет, что карта ещё не пришла.

~~~json
{"text": "My card has not arrived in the mail yet"}
~~~

~~~json
{"intent": "card_arrival"}
~~~

В поле `text` можно передать до 1 000 символов. На пустой или некорректный запрос API отвечает HTTP 422. Классификатор всегда выбирает одну из 77 тем. Для сообщений на другие темы понадобится отдельное правило маршрутизации.

## Оценка

Из опубликованной обучающей выборки выделили 20% сообщений на стратифицированную валидацию. После нормализации убрали повторы внутри выборок и 7 тестовых сообщений, совпавших с обучающими.

По macro-F1 на валидации выбрали TF-IDF по словам и символам. Эту модель переобучили на всей очищенной обучающей выборке и проверили на 3 071 отложенном обращении.

| Выборка | Модель | Macro-F1 |
|---|---|---:|
| Validation | Самый частый класс | 0,0005 |
| Validation | TF-IDF по словам + LinearSVC | 0,870 |
| Validation | TF-IDF по словам и символам + LinearSVC | **0,899** |
| Test | TF-IDF по словам и символам + LinearSVC | **0,909** |

Одна из частых ошибок связана с переводами. Модель путала `pending_transfer` и `failed_transfer`. Accuracy, размеры выборок и остальные частые ошибки есть в [отчёте](metrics.json).

## Обучение и тесты

В Windows запустите PowerShell с Python 3.13.

~~~powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe download_data.py
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
~~~

На Linux и macOS используйте `python3.13 -m venv .venv` и `.venv/bin/python`. После обучения файлы появятся в `artifacts/`.

Для запуска API с новой моделью укажите `artifacts/model.joblib` в переменной `SUPPORT_MODEL_PATH` и выполните `python -m uvicorn api:app`. В CLI готовую модель можно проверить командой `python predict.py "My card has not arrived in the mail yet"`.

Тесты проверяют очистку текста, удаление дубликатов между выборками и ответы сохранённой модели на корректные и некорректные запросы. [GitHub Actions](.github/workflows/ci.yml) прогоняет эти тесты, а затем проверяет API в собранном контейнере.

Скрипт скачивает CSV из указанной версии набора, опубликованного под CC BY 4.0. [BANKING77 от PolyAI](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data)

Статья Iñigo Casanueva и соавторов, 2020. [Efficient Intent Detection with Dual Sentence Encoders](https://arxiv.org/abs/2003.04807)
