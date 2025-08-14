# Labor-Verzeichnis

Dieses Verzeichnis dient als "Spielwiese" und Archiv für Skripte, die entweder experimentell, veraltet oder nicht in die Hauptanwendung integriert sind.

Der Code hier ist **nicht** Teil des produktiven Workflows und kann veraltete Programmiermuster oder Abhängigkeiten enthalten.

## Inhalt

- `chatbot.py`: Ein experimenteller, Kommandozeilen-basierter Chatbot, der den `AlimaManager` verwendet.
- `crawl.py`: Ein veraltetes Skript zum Crawlen von Webseiten, dessen Funktionalität durch `doi_resolver.py` abgelöst wurde.
- `bilderkennung.py`: Eine Kommandozeilen-Version der Bild-zu-Text-Funktion. Die Logik sollte in die zentrale Pipeline integriert werden.
- `test_ollama.py` / `test_ollama_2.py`: Skripte zum Testen der Ollama-Anbindung.
- `widget_test.py`: Test-Skript für UI-Widgets.
- `sshtest.py`: Test-Skript für SSH-Verbindungen.
