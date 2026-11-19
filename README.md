# Support Intent Classifier

Классификатор определяет одну из 77 тем банковского обращения на английском. Модель использует TF-IDF по словам и символам и LinearSVC. Готовые веса входят в репозиторий, запросы принимает FastAPI.

Отдельный [RAG-помощник по учебной базе знаний](rag/README.md) ищет инструкции и формирует ответ с источником. В нём сравниваются два промпта и варианты поиска. Инструкции вымышленные и не относятся к Сберу или другому реальному банку.

## Запуск в Docker

~~~sh
docker build -t support-intent-classifier .
docker run --rm -p 127.0.0.1:8000:8000 support-intent-classifier
~~~

Контейнер загружает [готовую модель](models/support-intents.joblib). После этого `GET /health` возвращает `{"status":"ok"}`. Схема запросов доступна в [Swagger UI](http://localhost:8000/docs).

Во втором терминале отправьте [пример запроса](example-request.json) к `POST /predict`.

~~~sh
curl --fail-with-body \
  -H "Content-Type: application/json" \
  --data-binary "@example-request.json" \
  http://127.0.0.1:8000/predict
~~~

~~~powershell
curl.exe --fail-with-body -H "Content-Type: application/json" --data-binary "@example-request.json" http://127.0.0.1:8000/predict
~~~

Пример обращения о задержке доставки карты:

~~~json
{"text": "My card has not arrived in the mail yet"}
~~~

~~~json
{"intent": "card_arrival"}
~~~

Поле `text` принимает до 1 000 символов. Пустые и некорректные запросы получают HTTP 422. Модель всегда выбирает одну из 77 тем, поэтому для обращений вне BANKING77 потребуется отдельное правило маршрутизации.

## Оценка

Из опубликованной обучающей выборки выделили 20% сообщений на стратифицированную валидацию. После нормализации убрали повторы внутри выборок и 7 тестовых сообщений, совпавших с обучающими.

По macro-F1 на валидации выбрали TF-IDF по словам и символам. Эту модель переобучили на всей очищенной обучающей выборке и проверили на 3 071 отложенном обращении.

| Выборка | Модель | Macro-F1 |
|---|---|---:|
| Validation | Самый частый класс | 0,0005 |
| Validation | TF-IDF по словам + LinearSVC | 0,870 |
| Validation | TF-IDF по словам и символам + LinearSVC | **0,899** |
| Test | TF-IDF по словам и символам + LinearSVC | **0,909** |

Модель часто путала `pending_transfer` и `failed_transfer`. Accuracy, размеры выборок и другие ошибки приведены в [отчёте](metrics.json).

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

[GitHub Actions](.github/workflows/ci.yml) запускает быстрые тесты очистки данных, разделения выборок и API. Веса LLM для них не нужны. Проверку Docker-контейнера можно включить вручную параметром `check_container`. Полный RAG-прогон вынесен в отдельный ручной workflow, чтобы обычные правки проходили быстрее.

Скрипт скачивает CSV из указанной версии набора, опубликованного под CC BY 4.0. [BANKING77 от PolyAI](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data)

Статья Iñigo Casanueva и соавторов, 2020. [Efficient Intent Detection with Dual Sentence Encoders](https://arxiv.org/abs/2003.04807)
