# remove soon
## obsolete


from PyQt6.QtNetwork import QNetworkRequest, QNetworkAccessManager
from PyQt6.QtCore import QUrl, QByteArray, pyqtSignal, pyqtSlot, QObject
import json
import logging
from typing import Optional, Dict, Any
from ..utils.config import Config, ConfigSection, AIProvider


class AIProcessor(QObject):

    def __init__(self):
        self.config = Config.get_instance()
        self.logger = logging.getLogger(__name__)

        # Hole die Konfigurationen
        self.prompt_config = self.config.get_section(ConfigSection.PROMPTS)
        self.ai_config = self.config.get_section(ConfigSection.AI)

        if not self.prompt_config or not self.prompt_config.templates:
            raise ValueError("Keine Prompt-Templates in der Konfiguration gefunden")

        # Hole die Provider-Konfiguration
        self.logger.info(self.ai_config.provider)
        self.provider_config = self.config.get_ai_provider_config(
            self.ai_config.provider
        )
        # if not self.provider_config:
        #    raise ValueError(f"Keine Konfiguration für Provider {self.ai_config.provider} gefunden")

        # Initialisiere Network Manager
        self.network_manager = QNetworkAccessManager()
        self.generated_prompt = ""

    def set_input(
        self,
        abstract: str,
        keywords: str = "",
        template_name: str = "abstract_analysis",
    ):
        """
        Setzt die Eingabedaten für die AI-Verarbeitung.

        Args:
            abstract: Der zu analysierende Abstract
            keywords: Vorhandene Keywords (optional)
            template_name: Name des Prompt-Templates
        """
        self.logger.info(
            f"Setze Eingabedaten: Abstract={abstract}, Keywords={keywords}, Template={template_name}"
        )
        template = self.prompt_config.templates[template_name]
        # Bereite die Variablen vor
        variables = {
            "abstract": abstract,
            "keywords": keywords if keywords else "Keine Keywords vorhanden",
        }

        try:
            # Erstelle den Prompt
            prompt = template.template.format(**variables)
        except KeyError as e:
            raise ValueError(f"Fehlende Variable im Template: {e}")
        self.generated_prompt = prompt
        return prompt

    def prepare_request(
        self,
        prompt: str = "",
        temperature: float = 0.7,
        seed: int = None,
        model: str = "",
    ) -> tuple[QNetworkRequest, QByteArray]:
        # Verwende die Provider-URL aus der Konfiguration und füge API-Key hinzu
        base_url = self.provider_config["api_url"]
        if not base_url:
            raise ValueError(
                f"Keine API-URL für Provider {self.ai_config.provider} gefunden"
            )

        self.logger.info(model)
        if not model == "":
            self.logger.info("Modelname gefunden")

            base_url = base_url.replace("modelname", model)
        else:
            self.logger.info("Kein Modelname gefunden")
            base_url = base_url.replace("modelname", "gemini-1.5-flash")

        if not base_url:
            raise ValueError(
                f"Keine API-URL für Provider {self.ai_config.provider} gefunden"
            )
        self.logger.info(base_url)

        api_url = f"{base_url}?key={self.ai_config.api_key}"
        # else:

        # Erstelle Request
        request = QNetworkRequest(QUrl(api_url))
        request.setHeader(
            QNetworkRequest.KnownHeaders.ContentTypeHeader, "application/json"
        )

        data = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt  # ,
                            # "temperature": temperature,
                            # "seed": seed
                        }
                    ]
                }
            ]
        }

        self.logger.info(data)
        # Konvertiere die Daten in QByteArray
        request_data = QByteArray(json.dumps(data).encode())
        self.logger.info(request_data)
        return request, request_data

    def send_request(self, abstract: str, keywords: str = ""):
        """
        Sendet eine Anfrage an die AI-API.
        """
        try:
            request, data = self.prepare_request(abstract, keywords)
            reply = self.network_manager.post(request, data)
            reply.finished.connect(lambda: self.process_response(reply))
        except Exception as e:
            self.logger.error(f"Fehler bei der Anfrage: {str(e)}")
            raise

    def process_response(self, reply):
        """
        Verarbeitet die Antwort der AI-API.
        """
        try:
            response_data = reply
            # json_response = json.loads(response_data)
            # self.logger.info(json_response)
            # Provider-spezifische Verarbeitung
            # if self.ai_config.provider == AIProvider.GEMINI:
            # if "candidates" not in json_response:
            #    raise ValueError("Ungültiges Antwortformat: 'candidates' fehlt")
            content = reply["candidates"][0]["content"]["parts"][0]["text"]
            # else:
            #    if "choices" not in json_response:
            #        raise ValueError("Ungültiges Antwortformat: 'choices' fehlt")
            #    content = json_response["choices"][0]["message"]["content"]

            return content
        except Exception as e:
            self.logger.error(f"Fehler bei der Verarbeitung der Antwort: {str(e)}")
            raise
        # finally:
        #    reply.deleteLater()
