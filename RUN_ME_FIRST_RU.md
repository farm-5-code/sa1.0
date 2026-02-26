# Sport Analyzer — быстрый запуск (для себя)

## 1) Установка зависимостей
В папке проекта:

```bash
pip install -r requirements.txt
```

## 2) (Необязательно) ключи API
Если ключей нет — приложение всё равно работает, просто часть данных будет недоступна.

1) Скопируй пример:
```bash
cp sport_analyzer/.env.example .env
```
2) Открой `.env` и вставь ключи (если есть).

## 3) Самопроверка (рекомендуется)
Одна команда покажет, всё ли в порядке:

```bash
python -m sport_analyzer.main --self-check
```

## 4) Запуск в консоли (быстро)
Список ближайших матчей (нужен FOOTBALL_DATA_KEY, иначе будет пусто):
```bash
python -m sport_analyzer.main --matches
```

Анализ матча:
```bash
python -m sport_analyzer.main --home "Bayern" --away "Dortmund" --city "Munich" --date "2026-03-10T20:00:00"
```

## 5) Запуск с интерфейсом (красиво)
```bash
streamlit run sport_analyzer/dashboard/app.py
```

## 6) Настройки «для себя» (очень просто)
Открой файл:

`sport_analyzer/config/settings_user.py`

Там можно:
- поменять веса факторов (USER_WEIGHTS)
- поднять/опустить порог уверенности (USER_MIN_CONFIDENCE)
- изменить сколько матчей считать для формы (USER_FORM_MATCHES)

После изменения снова:
```bash
python -m sport_analyzer.main --self-check
```
