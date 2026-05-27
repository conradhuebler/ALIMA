# Headless agent command handler for ALIMA CLI — Claude Generated (P-ι)
"""``alima agent`` — drive the chat-agent toolset headless from the CLI.

Reuses the same ``AgentLoop`` + chat toolset as the GUI chat (via
``HeadlessAgentRunner``), so a power-user can go from a DOI to a tool-driven
answer without opening the GUI.

Permission model (Roadmap "explicit" default): confirmation-gated operations
(mutations, pipeline starts) prompt y/N on stdin via ``StdinProposalGateway``.
``--autonomous`` flips ``ChatConfig.autonomous_pipeline`` so they apply without
a prompt (destructive cache writes still respect ``no_cache_writes``).
"""
from __future__ import annotations

import json
import logging
import signal
import sys
import threading
from datetime import datetime
from typing import Any, Optional, Tuple

from src.core.alima_manager import AlimaManager
from src.core.headless_agent import HeadlessAgentRunner, resolve_provider_model
from src.core.headless_gateway import StdinProposalGateway
from src.core.pipeline_manager import PipelineManager
from src.core.unified_knowledge_manager import UnifiedKnowledgeManager
from src.utils.config_manager import ConfigManager
from src.utils.config_models import ChatConfig
from src.utils.doi_resolver import resolve_input_to_text
from src.utils.pipeline_utils import execute_input_extraction
from src.webapp.result_serialization import (
    build_export_payload,
    extract_results_from_analysis_state,
)


def _load_chat_config(config_manager: ConfigManager) -> ChatConfig:
    # chat_config lives on the top-level AlimaConfig (load_config), NOT on the
    # UnifiedProviderConfig — reading get_unified_config().chat_config always
    # yields None and silently drops the saved chat defaults + autonomous flag.
    try:
        alima_cfg = config_manager.load_config()
        chat_cfg = getattr(alima_cfg, "chat_config", None)
        if chat_cfg is not None:
            return chat_cfg
    except Exception:
        logging.getLogger(__name__).exception("ChatConfig lookup failed; using defaults")
    return ChatConfig()


def _resolve_provider_model(args, chat_config, pipeline_manager, llm_service) -> Tuple[str, str]:
    """Resolve provider/model for the CLI agent (see resolve_provider_model)."""
    return resolve_provider_model(
        getattr(args, "provider", None),
        getattr(args, "model", None),
        chat_config=chat_config,
        pipeline_manager=pipeline_manager,
        llm_service=llm_service,
    )


def _resolve_input(args, llm_service, logger) -> Tuple[str, str, str]:
    """Return (context_text, input_type, source) for the work under analysis.

    Empty context_text means "no work loaded" — the agent then works purely
    from the prompt / its tools.
    """
    if getattr(args, "doi", None):
        success, text, error = resolve_input_to_text(args.doi, logger)
        if not success:
            raise ValueError(f"DOI/URL resolution failed: {error}")
        return text or "", "doi", args.doi
    if getattr(args, "input_image", None):
        text, source_info, _method = execute_input_extraction(
            llm_service=llm_service,
            input_source=args.input_image,
            input_type="image",
            logger=logger,
        )
        return text or "", "image", args.input_image
    if getattr(args, "input_file", None):
        with open(args.input_file, "r", encoding="utf-8") as f:
            return f.read(), "text", args.input_file
    if getattr(args, "input", None):
        return args.input, "text", "<inline>"
    return "", "none", ""


def handle_agent(args, config_manager: ConfigManager, llm_service, prompt_service: Any,
                 logger: logging.Logger) -> int:
    """``alima agent`` — run one headless agent turn over an optional input."""
    quiet = bool(getattr(args, "quiet", False))

    def _emit_token(tok: str) -> None:
        if not quiet:
            sys.stdout.write(tok)
            sys.stdout.flush()

    def _emit_status(line: str) -> None:
        sys.stderr.write(line)
        sys.stderr.flush()

    def _emit_tool_call(tc) -> None:
        sys.stderr.write(f"\n🔧 {tc.name}({json.dumps(tc.arguments, ensure_ascii=False)[:120]})\n")
        sys.stderr.flush()

    def _emit_tool_result(name: str, result: str) -> None:
        sys.stderr.write(f"   ↳ {result[:160]}\n")
        sys.stderr.flush()

    # --- core services -------------------------------------------------
    alima_manager = AlimaManager(llm_service, prompt_service, config_manager, logger)
    cache_manager = UnifiedKnowledgeManager()
    pipeline_manager = PipelineManager(
        alima_manager=alima_manager,
        cache_manager=cache_manager,
        logger=logger,
        config_manager=config_manager,
    )

    chat_config = _load_chat_config(config_manager)
    autonomous = bool(getattr(args, "autonomous", False))
    gateway = None
    if autonomous:
        chat_config.autonomous_pipeline = True
    else:
        gateway = StdinProposalGateway()

    provider, model = _resolve_provider_model(args, chat_config, pipeline_manager, llm_service)
    if not provider or not model:
        print("❌ Kein LLM-Provider/Modell. Setze --provider/--model oder ChatConfig-Defaults.",
              file=sys.stderr)
        return 2

    # --- input ---------------------------------------------------------
    try:
        context_text, input_type, source = _resolve_input(args, llm_service, logger)
    except (ValueError, OSError) as e:
        print(f"❌ Eingabe fehlgeschlagen: {e}", file=sys.stderr)
        return 2

    user_message = getattr(args, "prompt", None) or (
        "Analysiere das vorliegende Werk und schlage GND-Schlagwörter sowie "
        "DK-Klassifikationen vor." if context_text else
        "Was kannst du tun? Liste verfügbare Daten und Werkzeuge."
    )
    # Trim context for the prompt header; tools fetch full data on demand.
    context_str = (context_text[:1500] + "…") if len(context_text) > 1500 else context_text

    runner = HeadlessAgentRunner(
        llm_service=llm_service,
        pipeline_manager=pipeline_manager,
        chat_config=chat_config,
        gateway=gateway,
        max_iterations=getattr(args, "max_iterations", None),
    )

    # --- cancel: SIGINT sets a stop event on this thread so the P-ζ
    #     pipeline tools (_resolve_should_stop) and AgentLoop both abort.
    stop_event = threading.Event()
    threading.current_thread()._stop_event = stop_event  # read by pipeline tools

    def _on_sigint(signum, frame):
        sys.stderr.write("\n⏹ Abbruch angefordert …\n")
        stop_event.set()
        try:
            llm_service.cancel_generation(reason="cli_sigint")
        except Exception:
            pass

    prev_handler = signal.signal(signal.SIGINT, _on_sigint)
    try:
        result = runner.run(
            user_message,
            provider=provider,
            model=model,
            context_str=context_str,
            temperature=getattr(args, "temperature", None),
            on_token=_emit_token,
            on_status=_emit_status,
            on_tool_call=_emit_tool_call,
            on_tool_result=_emit_tool_result,
            should_stop=stop_event.is_set,
        )
    finally:
        signal.signal(signal.SIGINT, prev_handler)

    if not quiet:
        sys.stdout.write("\n")

    # --- serialize -----------------------------------------------------
    analysis_state = getattr(pipeline_manager, "current_analysis_state", None)
    results = extract_results_from_analysis_state(analysis_state) if analysis_state else None
    payload = build_export_payload(
        session_id="cli-agent-" + datetime.now().strftime("%Y%m%d_%H%M%S"),
        created_at=datetime.now().isoformat(),
        status="completed",
        current_step=None,
        input_data={"type": input_type, "source": source},
        results=results,
    )
    payload["agent"] = {
        "final_content": result.content,
        "iterations": result.iterations,
        "tool_log": result.tool_log,
        "provider": provider,
        "model": model,
        "autonomous": autonomous,
    }

    output = getattr(args, "output", None)
    if output:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n💾 Wrote agent result to {output}", file=sys.stderr)
    else:
        print()
        json.dump(payload, sys.stdout, ensure_ascii=False, indent=2, default=str)
        print()

    return 0
