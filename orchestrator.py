# orchestrator.py · v1.0.19 · 2026-04-23 (Asia/Almaty)

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = "1.0.19"
TZ = ZoneInfo("Asia/Almaty")
CODEX_RETRY_MINUTES = 30
CLAUDE_RETRY_MINUTES = 60

ROOT = Path(__file__).resolve().parent
QUEUE_FILE = ROOT / "orchestrator_queue.json"
STATE_FILE = ROOT / "orchestrator_state.json"
AGENTS_FILE = ROOT / "orchestrator_agents.json"
TASK_TEMPLATE_FILE = ROOT / "orchestrator_task_template.json"
MAIN_BOT_HEALTH_TASK_FILE = ROOT / "task_003_main_bot_health_check.json"
LOG_DIR = ROOT / ".ai_logs"


def now_iso() -> str:
    return datetime.now(TZ).isoformat(timespec="seconds")


def parse_iso_dt(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=TZ)
    return parsed.astimezone(TZ)


def retry_after_iso(minutes: int) -> str:
    return (datetime.now(TZ) + timedelta(minutes=minutes)).isoformat(timespec="seconds")


def retry_is_due(value: object) -> bool:
    retry_at = parse_iso_dt(value)
    return retry_at is None or datetime.now(TZ) >= retry_at


def normalize_task_class(value: object) -> str:
    task_class = str(value or "complex").strip().lower()
    return task_class if task_class in {"simple", "complex"} else "complex"


def normalize_review_policy(value: object) -> str:
    review_policy = str(value or "require_claude").strip().lower()
    return review_policy if review_policy in {"auto", "require_claude"} else "require_claude"


def should_require_claude_review(task: dict) -> bool:
    task_class = normalize_task_class(task.get("task_class"))
    review_policy = normalize_review_policy(task.get("review_policy"))
    if task_class == "complex":
        return True
    return review_policy == "require_claude"


def normalize_target_system(value: object) -> str:
    target_system = str(value or "main_bot").strip().lower()
    return target_system if target_system == "main_bot" else "main_bot"


def normalize_review_round(value: object) -> int:
    try:
        review_round = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, review_round)


def normalize_max_review_rounds(value: object) -> int:
    try:
        max_rounds = int(value or 1)
    except (TypeError, ValueError):
        return 1
    return max(1, max_rounds)


def sanitize_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def no_harm_rules() -> list[str]:
    return [
        "Не расширять scope задачи и не подменять цель.",
        "Не трогать файлы вне allowed_files.",
        "Не делать широких рефакторингов и не менять архитектуру без необходимости.",
        "Не отключать защиту, SSL/TLS, ACL или проверки ради сокрытия симптомов.",
        "Не понижать severity логов, чтобы скрыть проблему.",
        "Не менять бизнес-логику без доказанного root cause.",
        "Если доказательств недостаточно — остановиться на локализации, а не выдумывать фикс.",
    ]


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {path}")
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Ожидался JSON-объект: {path}")
    return data


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else ROOT / path


def set_state(
    *,
    running: bool,
    current_stage: str | None,
    current_agent: str | None,
    last_error: str | None,
) -> dict:
    state = load_json(STATE_FILE)
    state["running"] = running
    state["current_stage"] = current_stage
    state["current_agent"] = current_agent
    state["last_error"] = last_error
    state["last_update"] = now_iso()
    save_json(STATE_FILE, state)
    return state


def validate_required_files() -> list[str]:
    errors: list[str] = []
    required = [
        QUEUE_FILE,
        STATE_FILE,
        AGENTS_FILE,
        TASK_TEMPLATE_FILE,
    ]
    for path in required:
        if not path.exists():
            errors.append(f"Отсутствует файл: {path.name}")
    return errors


def validate_agents_config(agents: dict) -> list[str]:
    errors: list[str] = []

    for agent_name in ("codex", "claude"):
        if agent_name not in agents:
            errors.append(f"Нет секции '{agent_name}' в orchestrator_agents.json")
            continue

        agent = agents[agent_name]
        command = str(agent.get("command", "")).strip()
        timeout_sec = agent.get("timeout_sec")
        args = agent.get("args", [])

        if not command:
            errors.append(f"У агента '{agent_name}' не указан command")
        elif shutil.which(command) is None and not Path(command).exists():
            errors.append(f"Команда агента '{agent_name}' не найдена: {command}")

        if not isinstance(timeout_sec, int) or timeout_sec <= 0:
            errors.append(f"У агента '{agent_name}' некорректный timeout_sec")

        if not isinstance(args, list):
            errors.append(f"У агента '{agent_name}' поле args должно быть списком")

    return errors


def validate_task(task: dict) -> list[str]:
    errors: list[str] = []

    required_fields = [
        "task_id",
        "title",
        "project_root",
        "target_system",
        "status",
        "stage",
        "goal",
        "constraints",
        "inputs",
        "allowed_files",
        "codex_output_file",
        "claude_output_file",
    ]
    for field in required_fields:
        if field not in task:
            errors.append(f"В задаче отсутствует поле: {field}")

    task_id = str(task.get("task_id", "")).strip()
    title = str(task.get("title", "")).strip()
    status = str(task.get("status", "")).strip()
    stage = str(task.get("stage", "")).strip()
    target_system = normalize_target_system(task.get("target_system"))
    task_class = normalize_task_class(task.get("task_class"))
    review_policy = normalize_review_policy(task.get("review_policy"))
    review_round = normalize_review_round(task.get("review_round"))
    max_review_rounds = normalize_max_review_rounds(task.get("max_review_rounds"))

    if not task_id:
        errors.append("Поле task_id пустое")
    if not title:
        errors.append("Поле title пустое")
    if status not in {"new", "queued", "running", "review", "done", "failed", "rate_limited"}:
        errors.append(f"Недопустимое значение status: {status}")
    if stage not in {"analysis", "execution", "review", "done"}:
        errors.append(f"Недопустимое значение stage: {stage}")
    if target_system != "main_bot":
        errors.append(f"Недопустимое значение target_system: {target_system}")
    if task_class == "complex" and review_policy != "require_claude":
        errors.append("Для task_class=complex review_policy должен быть require_claude")
    if review_round > max_review_rounds:
        errors.append("review_round не может быть больше max_review_rounds")

    constraints = task.get("constraints")
    allowed_files = task.get("allowed_files")
    inputs = task.get("inputs")

    if not isinstance(constraints, list):
        errors.append("Поле constraints должно быть списком")
    if not isinstance(allowed_files, list):
        errors.append("Поле allowed_files должно быть списком")
    if not isinstance(inputs, dict):
        errors.append("Поле inputs должно быть объектом")

    if isinstance(inputs, dict):
        for key in ("files", "logs", "notes"):
            if key not in inputs:
                errors.append(f"В inputs отсутствует поле: {key}")
            elif not isinstance(inputs[key], list):
                errors.append(f"inputs.{key} должно быть списком")

    task["target_system"] = target_system
    task["task_class"] = task_class
    task["review_policy"] = review_policy
    task["review_round"] = review_round
    task["max_review_rounds"] = max_review_rounds
    task["revise_requested"] = task.get("revise_requested") is True
    task["required_changes"] = sanitize_string_list(task.get("required_changes"))
    task["out_of_scope_requests"] = sanitize_string_list(task.get("out_of_scope_requests"))

    return errors


def get_status() -> dict:
    queue = load_json(QUEUE_FILE)
    state = load_json(STATE_FILE)
    agents = load_json(AGENTS_FILE)

    errors = []
    errors.extend(validate_required_files())
    errors.extend(validate_agents_config(agents))

    status = {
        "version": VERSION,
        "checked_at": now_iso(),
        "project_root": str(ROOT),
        "queue_tasks": len(queue.get("tasks", [])),
        "history_tasks": len(queue.get("history", [])),
        "active_task": queue.get("active_task"),
        "running": state.get("running"),
        "current_stage": state.get("current_stage"),
        "current_agent": state.get("current_agent"),
        "errors": errors,
        "ready": len(errors) == 0,
    }
    return status


def print_status() -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        status = get_status()
    except Exception as e:
        set_state(
            running=False,
            current_stage="status_check",
            current_agent=None,
            last_error=str(e),
        )
        print(f"[ERROR] {e}")
        return 1

    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0 if status["ready"] else 2


def enqueue_task(task_path_str: str) -> int:
    task_path = resolve_path(task_path_str)
    if not task_path.exists():
        print(f"[ERROR] Файл задачи не найден: {task_path}")
        return 1

    try:
        queue = load_json(QUEUE_FILE)
        task = load_json(task_path)
        errors = validate_task(task)
        if errors:
            print(json.dumps({"ok": False, "errors": errors}, ensure_ascii=False, indent=2))
            return 2

        task["status"] = "queued"
        task["updated_at"] = now_iso()
        if not str(task.get("created_at", "")).strip():
            task["created_at"] = now_iso()

        task_id = str(task["task_id"]).strip()

        existing_ids = {str(item.get("task_id", "")).strip() for item in queue.get("tasks", [])}
        existing_ids.update({str(item.get("task_id", "")).strip() for item in queue.get("history", [])})
        active_task = queue.get("active_task")
        if isinstance(active_task, dict):
            existing_ids.add(str(active_task.get("task_id", "")).strip())

        if task_id in existing_ids:
            print(json.dumps({
                "ok": False,
                "error": f"Задача с task_id '{task_id}' уже существует"
            }, ensure_ascii=False, indent=2))
            return 3

        queue.setdefault("tasks", []).append(task)
        save_json(QUEUE_FILE, queue)

        set_state(
            running=False,
            current_stage="queued",
            current_agent=None,
            last_error=None,
        )

        print(json.dumps({
            "ok": True,
            "message": "Задача поставлена в очередь",
            "task_id": task_id,
            "queue_tasks": len(queue["tasks"]),
        }, ensure_ascii=False, indent=2))
        return 0

    except Exception as e:
        set_state(
            running=False,
            current_stage="enqueue",
            current_agent=None,
            last_error=str(e),
        )
        print(f"[ERROR] {e}")
        return 1


def build_codex_prompt(task: dict) -> str:
    files = "\n".join(f"- {item}" for item in task.get("inputs", {}).get("files", [])) or "- нет"
    logs = "\n".join(f"- {item}" for item in task.get("inputs", {}).get("logs", [])) or "- нет"
    notes = "\n".join(f"- {item}" for item in task.get("inputs", {}).get("notes", [])) or "- нет"
    allowed_files = "\n".join(f"- {item}" for item in task.get("allowed_files", [])) or "- нет"
    constraints = "\n".join(f"- {item}" for item in task.get("constraints", [])) or "- нет"
    do_no_harm = "\n".join(f"- {item}" for item in no_harm_rules())

    if task.get("revise_requested") is True:
        required_changes = "\n".join(f"- {item}" for item in sanitize_string_list(task.get("required_changes"))) or "- нет"
        out_of_scope = "\n".join(f"- {item}" for item in sanitize_string_list(task.get("out_of_scope_requests"))) or "- нет"
        codex_result = task.get("codex_result") if isinstance(task.get("codex_result"), dict) else {}
        stdout_file = codex_result.get("stdout_file") or ""
        stderr_file = codex_result.get("stderr_file") or ""
        claude_result = task.get("claude_result") if isinstance(task.get("claude_result"), dict) else {}
        review_file = claude_result.get("review_file") or task.get("claude_output_file") or ""

        return f"""Ты работаешь локально в проекте: {ROOT}

ЭТО НЕ НОВАЯ ЗАДАЧА.
ЭТО ОДИН revise-pass ПО ЗАМЕЧАНИЯМ CLAUDE.

ЗАДАЧА:
{task.get("title", "")}

ЦЕЛЬ:
{task.get("goal", "")}

ОГРАНИЧЕНИЯ:
{constraints}

ВХОДНЫЕ ФАЙЛЫ:
{files}

ЛОГИ:
{logs}

ПРИМЕЧАНИЯ:
{notes}

РАЗРЕШЕНО ТРОГАТЬ ТОЛЬКО:
{allowed_files}

ЦЕЛЕВАЯ СИСТЕМА:
- target_system: {normalize_target_system(task.get("target_system"))}

КЛАСС ЗАДАЧИ:
- task_class: {normalize_task_class(task.get("task_class"))}
- review_policy: {normalize_review_policy(task.get("review_policy"))}
- review_round: {normalize_review_round(task.get("review_round"))}
- max_review_rounds: {normalize_max_review_rounds(task.get("max_review_rounds"))}

ПРАВИЛА НЕ НАВРЕДИ:
{do_no_harm}

ПРЕДЫДУЩИЙ РЕЗУЛЬТАТ CODEX:
- {task.get("codex_output_file", "")}
- {stdout_file}
- {stderr_file}

REVIEW CLAUDE:
- {review_file}

ОБЯЗАТЕЛЬНО ИСПРАВИТЬ:
{required_changes}

ВНЕ SCOPE, НЕ ИСПОЛНЯТЬ:
{out_of_scope}

ПРАВИЛА:
- Исправляй только пункты из ОБЯЗАТЕЛЬНО ИСПРАВИТЬ.
- Не расширяй scope и не начинай новый аудит с нуля.
- Если какой-то пункт неприменим, укажи это явно с доказательством.
- Это единственный revise-pass.

Верни только итоговый JSON без пояснений, markdown и вводных фраз.
Ключи JSON:
summary, findings, addressed_review_items, changed_files, risks, checks_run, next_action.

Важно:
- внутри JSON используй только ASCII-символы;
- весь русский текст кодируй escape-последовательностями \\uXXXX;
- не добавляй никакой текст до или после JSON.
"""

    return f"""Ты работаешь локально в проекте: {ROOT}

ЗАДАЧА:
{task.get("title", "")}

ЦЕЛЬ:
{task.get("goal", "")}

ОГРАНИЧЕНИЯ:
{constraints}

ВХОДНЫЕ ФАЙЛЫ:
{files}

ЛОГИ:
{logs}

ПРИМЕЧАНИЯ:
{notes}

РАЗРЕШЕНО ТРОГАТЬ ТОЛЬКО:
{allowed_files}

ЦЕЛЕВАЯ СИСТЕМА:
- target_system: {normalize_target_system(task.get("target_system"))}

КЛАСС ЗАДАЧИ:
- task_class: {normalize_task_class(task.get("task_class"))}
- review_policy: {normalize_review_policy(task.get("review_policy"))}

ПРАВИЛА НЕ НАВРЕДИ:
{do_no_harm}

ВЫПОЛНИ ЗАДАЧУ ПОЛНОСТЬЮ В РАМКАХ РАЗРЕШЁННЫХ ФАЙЛОВ.
Если найдёшь подтверждённый баг в разрешённых файлах — внеси минимальный патч.
Если багов нет — не меняй код и честно зафиксируй это в JSON.

Верни только итоговый JSON без пояснений, markdown и вводных фраз.
Ключи JSON:
summary, findings, changed_files, risks, checks_run, next_action.

Важно:
- внутри JSON используй только ASCII-символы;
- весь русский текст кодируй escape-последовательностями \\uXXXX;
- не добавляй никакой текст до или после JSON.
"""


def complete_without_claude() -> int:
    queue = load_json(QUEUE_FILE)
    active_task = queue.get("active_task")
    if not isinstance(active_task, dict):
        raise RuntimeError("active_task потерян перед автозавершением")

    active_task["updated_at"] = now_iso()
    queue["active_task"] = active_task
    save_json(QUEUE_FILE, queue)
    return complete_active_task("done", "Codex completed; Claude review not required")


def build_main_bot_health_task() -> dict:
    if not MAIN_BOT_HEALTH_TASK_FILE.exists():
        raise FileNotFoundError(f"Файл шаблона main_bot health-check не найден: {MAIN_BOT_HEALTH_TASK_FILE.name}")

    task = load_json(MAIN_BOT_HEALTH_TASK_FILE)
    task_suffix = datetime.now(TZ).strftime("%Y%m%d-%H%M%S")
    task_id = f"TASK-003-MAIN-BOT-HEALTH-{task_suffix}"
    created_at = now_iso()

    task["task_id"] = task_id
    task["status"] = "new"
    task["stage"] = "analysis"
    task["created_at"] = created_at
    task["updated_at"] = created_at
    task["codex_output_file"] = f".ai_logs/{task_id}_codex_result.json"
    task["claude_output_file"] = f".ai_reviews/{task_id}_claude_review.json"
    task["last_codex_attempt_at"] = None
    task["last_claude_attempt_at"] = None
    task["next_codex_retry_after"] = None
    task["next_claude_retry_after"] = None
    task["codex_rate_limited"] = False
    task["claude_rate_limited"] = False
    task["last_codex_error"] = None
    task["last_claude_error"] = None
    task["review_attempts"] = 0
    task["codex_result"] = None
    task["claude_result"] = None
    task["final_status"] = None
    task["review_round"] = 0
    task["max_review_rounds"] = 1
    task["revise_requested"] = False
    task["required_changes"] = []
    task["out_of_scope_requests"] = []
    return task


def enqueue_task_object(task: dict) -> dict:
    queue = load_json(QUEUE_FILE)
    errors = validate_task(task)
    if errors:
        raise ValueError("; ".join(errors))

    task["status"] = "queued"
    task["updated_at"] = now_iso()
    if not str(task.get("created_at", "")).strip():
        task["created_at"] = now_iso()

    task_id = str(task["task_id"]).strip()
    existing_ids = {str(item.get("task_id", "")).strip() for item in queue.get("tasks", [])}
    existing_ids.update({str(item.get("task_id", "")).strip() for item in queue.get("history", [])})
    active_task = queue.get("active_task")
    if isinstance(active_task, dict):
        existing_ids.add(str(active_task.get("task_id", "")).strip())
    if task_id in existing_ids:
        raise ValueError(f"Задача с task_id '{task_id}' уже существует")

    queue.setdefault("tasks", []).append(task)
    save_json(QUEUE_FILE, queue)

    set_state(
        running=False,
        current_stage="queued",
        current_agent=None,
        last_error=None,
    )
    return task


def mark_task_for_review(active_task: dict, queue: dict) -> int:
    active_task["status"] = "review"
    active_task["stage"] = "review"
    active_task["revise_requested"] = False
    queue["active_task"] = active_task
    save_json(QUEUE_FILE, queue)

    set_state(
        running=True,
        current_stage="review",
        current_agent=None,
        last_error=None,
    )

    print(json.dumps({
        "ok": True,
        "message": "Codex отработал. Задача переведена на этап review.",
        "task_id": active_task["task_id"],
        "task_class": active_task["task_class"],
        "review_policy": active_task["review_policy"],
        "review_round": active_task["review_round"],
        "max_review_rounds": active_task["max_review_rounds"],
        "codex_output_file": active_task["codex_output_file"],
        "stdout_file": active_task["codex_result"]["stdout_file"],
        "stderr_file": active_task["codex_result"]["stderr_file"],
    }, ensure_ascii=False, indent=2))
    return 0


def run_active_codex_pass(*, active_task: dict, queue: dict, agents: dict, requeue_on_rate_limit: bool) -> int:
    active_task["last_codex_attempt_at"] = now_iso()
    active_task["updated_at"] = now_iso()
    queue["active_task"] = active_task
    save_json(QUEUE_FILE, queue)

    set_state(
        running=True,
        current_stage=active_task.get("stage") or "analysis",
        current_agent="codex",
        last_error=None,
    )

    result = run_codex_task(active_task, agents)

    queue = load_json(QUEUE_FILE)
    current_active = queue.get("active_task") or active_task
    current_active["updated_at"] = now_iso()
    current_active["codex_result"] = result

    if result["ok"]:
        current_active["codex_rate_limited"] = False
        current_active["last_codex_error"] = None
        current_active["next_codex_retry_after"] = None
        if should_require_claude_review(current_active):
            return mark_task_for_review(current_active, queue)

        current_active["status"] = "done"
        current_active["stage"] = "done"
        queue["active_task"] = current_active
        save_json(QUEUE_FILE, queue)

        set_state(
            running=False,
            current_stage="auto_complete",
            current_agent=None,
            last_error=None,
        )
        return complete_without_claude()

    if result.get("rate_limited"):
        current_active["codex_rate_limited"] = True
        current_active["last_codex_error"] = result.get("error")
        current_active["next_codex_retry_after"] = retry_after_iso(CODEX_RETRY_MINUTES)
        current_active["updated_at"] = now_iso()

        if requeue_on_rate_limit:
            current_active["status"] = "queued"
            current_active["stage"] = "analysis"
            queue["active_task"] = None
            tasks = queue.get("tasks", [])
            queue["tasks"] = [current_active, *tasks]
        else:
            current_active["status"] = "running"
            current_active["stage"] = "execution"
            queue["active_task"] = current_active

        save_json(QUEUE_FILE, queue)

        set_state(
            running=False,
            current_stage="codex_rate_limited",
            current_agent=None,
            last_error=result.get("error"),
        )

        print(json.dumps({
            "ok": False,
            "message": "Codex упёрся в лимит.",
            "task_id": current_active["task_id"],
            "status": current_active["status"],
            "stage": current_active["stage"],
            "rate_limit_hint": result.get("rate_limit_hint"),
            "next_codex_retry_after": current_active["next_codex_retry_after"],
            "codex_output_file": current_active["codex_output_file"],
            "stdout_file": result["stdout_file"],
            "stderr_file": result["stderr_file"],
        }, ensure_ascii=False, indent=2))
        return 0

    current_active["status"] = "failed"
    queue["active_task"] = current_active
    save_json(QUEUE_FILE, queue)

    set_state(
        running=False,
        current_stage="failed",
        current_agent="codex",
        last_error=result.get("error") or f"Codex завершился с кодом {result.get('returncode')}",
    )

    print(json.dumps({
        "ok": False,
        "message": "Codex завершился с ошибкой",
        "task_id": current_active["task_id"],
        "codex_output_file": current_active["codex_output_file"],
        "stdout_file": result["stdout_file"],
        "stderr_file": result["stderr_file"],
        "returncode": result.get("returncode"),
    }, ensure_ascii=False, indent=2))
    return 4


def build_claude_prompt(task: dict) -> str:
    codex_result = task.get("codex_result") if isinstance(task.get("codex_result"), dict) else {}
    stdout_file = codex_result.get("stdout_file") or ""
    stderr_file = codex_result.get("stderr_file") or ""
    codex_output_file = task.get("codex_output_file") or ""
    do_no_harm = "\n".join(f"- {item}" for item in no_harm_rules())
    constraints = chr(10).join(f"- {item}" for item in task.get("constraints", [])) or "- нет"
    is_final_review = normalize_review_round(task.get("review_round")) >= normalize_max_review_rounds(task.get("max_review_rounds"))

    if is_final_review:
        return f"""Ты работаешь локально в проекте: {ROOT}

ЭТО ФИНАЛЬНЫЙ REVIEW ПОСЛЕ ОДНОГО REVISE-PASS.

ЗАДАЧА:
{task.get("title", "")}

ЦЕЛЬ:
{task.get("goal", "")}

ОГРАНИЧЕНИЯ:
{constraints}

ФАЙЛЫ РЕЗУЛЬТАТА CODEX:
- {codex_output_file}
- {stdout_file}
- {stderr_file}

ЦЕЛЕВАЯ СИСТЕМА:
- target_system: {normalize_target_system(task.get("target_system"))}

КЛАСС ЗАДАЧИ:
- task_class: {normalize_task_class(task.get("task_class"))}
- review_policy: {normalize_review_policy(task.get("review_policy"))}
- review_round: {normalize_review_round(task.get("review_round"))}
- max_review_rounds: {normalize_max_review_rounds(task.get("max_review_rounds"))}

ПРАВИЛА НЕ НАВРЕДИ:
{do_no_harm}

ПРАВИЛА REVIEW:
- Проверяй только, закрыты ли required_changes предыдущего review.
- Не расширяй scope.
- Не добавляй новые большие требования.
- Не запускай второй revise-цикл.
- Твой итог должен быть окончательным.

Верни только JSON без markdown и пояснений.
Ключи JSON:
accepted, verdict, summary, risks, unresolved_items, out_of_scope_requests, next_action.

Правила:
- accepted=true только если результат можно принять без доработок;
- verdict используй один из: accepted, rejected;
- если данных недостаточно, accepted=false и verdict=rejected;
- не меняй файлы проекта.
"""

    return f"""Ты работаешь локально в проекте: {ROOT}

ТЕБЕ НУЖНО СДЕЛАТЬ REVIEW РЕЗУЛЬТАТА CODEX.

ЗАДАЧА:
{task.get("title", "")}

ЦЕЛЬ:
{task.get("goal", "")}

ОГРАНИЧЕНИЯ:
{constraints}

ФАЙЛЫ РЕЗУЛЬТАТА CODEX:
- {codex_output_file}
- {stdout_file}
- {stderr_file}

ЦЕЛЕВАЯ СИСТЕМА:
- target_system: {normalize_target_system(task.get("target_system"))}

КЛАСС ЗАДАЧИ:
- task_class: {normalize_task_class(task.get("task_class"))}
- review_policy: {normalize_review_policy(task.get("review_policy"))}
- review_round: {normalize_review_round(task.get("review_round"))}
- max_review_rounds: {normalize_max_review_rounds(task.get("max_review_rounds"))}

ПРАВИЛА НЕ НАВРЕДИ:
{do_no_harm}

Верни только JSON без markdown и пояснений.
Ключи JSON:
accepted, verdict, summary, risks, required_changes, out_of_scope_requests, next_action.

Правила:
- accepted=true только если результат можно принять без доработок;
- verdict используй один из: accepted, revise, rejected;
- если данных недостаточно, accepted=false и verdict=revise;
- required_changes должен быть коротким и конечным списком;
- всё вне scope перечисляй только в out_of_scope_requests, но не включай в required_changes;
- не меняй файлы проекта.
"""


def decode_output_bytes(data: bytes | None) -> str:
    if not data:
        return ""

    candidates: list[tuple[int, str]] = []

    for enc in ("utf-8", "utf-8-sig", "cp1251", "cp866"):
        try:
            text = data.decode(enc)
            score = text.count("\ufffd")
            candidates.append((score, text))
        except UnicodeDecodeError:
            continue

    if candidates:
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    return data.decode("utf-8", errors="replace")


def detect_rate_limit(stderr_text: str) -> bool:
    lines = [line.strip().lower() for line in (stderr_text or "").splitlines() if line.strip()]
    strict_markers = [
        "you've hit your usage limit",
        "you've hit your limit",
        "you have hit your usage limit",
        "you have hit your limit",
        "hit your usage limit",
        "usage limit",
        '"api_error_status":429',
        '"api_error_status": 429',
        "api_error_status: 429",
    ]

    for line in lines:
        is_error_line = line.startswith("error:") or "api_error_status" in line or "429" in line
        if not is_error_line:
            continue
        if any(marker in line for marker in strict_markers):
            return True

    return False


def extract_rate_limit_hint(stderr_text: str) -> str | None:
    lines = [line.strip() for line in (stderr_text or "").splitlines() if line.strip()]

    for line in reversed(lines):
        lowered = line.lower()

        if line.startswith("{"):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict) and payload.get("api_error_status") == 429:
                result_text = str(payload.get("result") or "")
                match = re.search(r"resets?\s+(.+?)(?:[.]\s*$|$)", result_text, flags=re.IGNORECASE)
                if match:
                    return match.group(1).strip()
                if result_text:
                    return result_text.strip()

        if not (
            lowered.startswith("error:")
            or "api_error_status" in lowered
            or "429" in lowered
        ):
            continue

        if "you've hit your usage limit" in lowered or "you've hit your limit" in lowered:
            match = re.search(r"try again at\s+(.+?)(?:[.]\s*$|$)", line, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()

            match = re.search(r"resets?\s+(.+?)(?:[.]\s*$|$)", line, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()

        if '"api_error_status":429' in lowered or '"api_error_status": 429' in lowered:
            match = re.search(r"resets?\s+(.+?)(?:[.]\s*$|$)", line, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()

    return None


def stdout_has_json_object(stdout_text: str) -> bool:
    text = (stdout_text or "").strip()
    if not text.startswith("{"):
        return False
    try:
        return isinstance(json.loads(text), dict)
    except json.JSONDecodeError:
        return False


def run_codex_task(task: dict, agents: dict) -> dict:
    agent = agents["codex"]
    prompt = build_codex_prompt(task)

    stdout_path = LOG_DIR / f'{task["task_id"]}_codex_stdout.txt'
    stderr_path = LOG_DIR / f'{task["task_id"]}_codex_stderr.txt'
    result_path = resolve_path(task["codex_output_file"])

    command = [agent["command"], *agent.get("args", []), "-"]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["NO_COLOR"] = "1"
    env["FORCE_COLOR"] = "0"

    try:
        completed = subprocess.run(
            command,
            cwd=str(ROOT),
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=int(agent["timeout_sec"]),
            env=env,
        )
    except subprocess.TimeoutExpired as e:
        stdout_text = decode_output_bytes(e.stdout if isinstance(e.stdout, bytes) else None)
        stderr_text = decode_output_bytes(e.stderr if isinstance(e.stderr, bytes) else None)
        save_text(stdout_path, stdout_text)
        save_text(stderr_path, stderr_text)

        result = {
            "ok": False,
            "agent": "codex",
            "task_id": task["task_id"],
            "returncode": None,
            "rate_limited": False,
            "rate_limit_hint": None,
            "prompt_via_stdin": True,
            "error": f"Таймаут выполнения Codex: {agent['timeout_sec']} сек",
            "stdout_file": str(stdout_path.relative_to(ROOT)),
            "stderr_file": str(stderr_path.relative_to(ROOT)),
            "finished_at": now_iso(),
        }
        save_json(result_path, result)
        return result

    stdout_text = decode_output_bytes(completed.stdout)
    stderr_text = decode_output_bytes(completed.stderr)

    save_text(stdout_path, stdout_text)
    save_text(stderr_path, stderr_text)

    combined_text = f"{stderr_text}\n{stdout_text}"
    stdout_json_ok = completed.returncode == 0 and stdout_has_json_object(stdout_text)
    rate_limited = False if stdout_json_ok else detect_rate_limit(combined_text)
    rate_limit_hint = None if stdout_json_ok else extract_rate_limit_hint(combined_text)

    error_text = None
    if rate_limited:
        error_text = "Достигнут лимит Codex"
        if rate_limit_hint:
            error_text += f"; повторить после: {rate_limit_hint}"
    elif completed.returncode != 0:
        error_text = f"Codex завершился с кодом {completed.returncode}"

    result = {
        "ok": completed.returncode == 0 and not rate_limited,
        "agent": "codex",
        "task_id": task["task_id"],
        "returncode": completed.returncode,
        "rate_limited": rate_limited,
        "rate_limit_hint": rate_limit_hint,
        "prompt_via_stdin": True,
        "error": error_text,
        "command": command,
        "stdout_file": str(stdout_path.relative_to(ROOT)),
        "stderr_file": str(stderr_path.relative_to(ROOT)),
        "finished_at": now_iso(),
    }
    save_json(result_path, result)
    return result


def parse_claude_review(stdout_text: str) -> dict:
    text = (stdout_text or "").strip()
    parsed: object
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return {
                "accepted": False,
                "verdict": "revise",
                "summary": "Claude вернул не-JSON ответ",
                "raw": text,
            }
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {
                "accepted": False,
                "verdict": "revise",
                "summary": "Claude JSON не удалось разобрать",
                "raw": text,
            }

    if not isinstance(parsed, dict):
        return {
            "accepted": False,
            "verdict": "revise",
            "summary": "Claude вернул JSON не-объект",
            "raw": text,
        }

    if isinstance(parsed.get("result"), str):
        result_text = str(parsed["result"]).strip()
        try:
            inner = json.loads(result_text)
        except json.JSONDecodeError:
            inner = None
        if isinstance(inner, dict):
            parsed = inner

    verdict = str(parsed.get("verdict", "")).strip().lower()
    accepted = parsed.get("accepted") is True or verdict in {"accepted", "approve", "approved"}
    if verdict not in {"accepted", "revise", "rejected"}:
        verdict = "accepted" if accepted else "revise"

    parsed["accepted"] = accepted
    parsed["verdict"] = verdict
    parsed["required_changes"] = sanitize_string_list(parsed.get("required_changes"))
    parsed["out_of_scope_requests"] = sanitize_string_list(parsed.get("out_of_scope_requests"))
    parsed["unresolved_items"] = sanitize_string_list(parsed.get("unresolved_items"))
    return parsed


def run_claude_task(task: dict, agents: dict) -> dict:
    agent = agents["claude"]
    prompt = build_claude_prompt(task)

    task_id = str(task["task_id"])
    stdout_path = LOG_DIR / f"{task_id}_claude_stdout.txt"
    stderr_path = LOG_DIR / f"{task_id}_claude_stderr.txt"
    review_output_file = str(task.get("claude_output_file") or "").strip()
    review_path = resolve_path(review_output_file) if review_output_file else LOG_DIR.parent / ".ai_reviews" / f"{task_id}_claude_review.json"

    command = [agent["command"], *agent.get("args", [])]

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["NO_COLOR"] = "1"
    env["FORCE_COLOR"] = "0"

    try:
        completed = subprocess.run(
            command,
            cwd=str(ROOT),
            input=prompt.encode("utf-8"),
            capture_output=True,
            timeout=int(agent["timeout_sec"]),
            env=env,
        )
    except subprocess.TimeoutExpired as e:
        stdout_text = decode_output_bytes(e.stdout if isinstance(e.stdout, bytes) else None)
        stderr_text = decode_output_bytes(e.stderr if isinstance(e.stderr, bytes) else None)
        save_text(stdout_path, stdout_text)
        save_text(stderr_path, stderr_text)

        result = {
            "ok": False,
            "agent": "claude",
            "task_id": task_id,
            "returncode": None,
            "rate_limited": False,
            "rate_limit_hint": None,
            "accepted": False,
            "verdict": "failed",
            "error": f"Таймаут выполнения Claude: {agent['timeout_sec']} сек",
            "stdout_file": str(stdout_path.relative_to(ROOT)),
            "stderr_file": str(stderr_path.relative_to(ROOT)),
            "review_file": str(review_path.relative_to(ROOT)),
            "finished_at": now_iso(),
        }
        save_json(review_path, result)
        return result

    stdout_text = decode_output_bytes(completed.stdout)
    stderr_text = decode_output_bytes(completed.stderr)
    save_text(stdout_path, stdout_text)
    save_text(stderr_path, stderr_text)

    combined_text = f"{stderr_text}\n{stdout_text}"
    review = parse_claude_review(stdout_text) if completed.returncode == 0 else {}
    stdout_json_ok = bool(review)
    rate_limited = False if stdout_json_ok else detect_rate_limit(combined_text)
    rate_limit_hint = None if stdout_json_ok else extract_rate_limit_hint(combined_text)
    verdict = str(review.get("verdict", "failed"))
    accepted = review.get("accepted") is True

    error_text = None
    if rate_limited:
        error_text = "Достигнут лимит Claude"
        if rate_limit_hint:
            error_text += f"; повторить после: {rate_limit_hint}"
    elif completed.returncode != 0:
        error_text = f"Claude завершился с кодом {completed.returncode}"

    result = {
        "ok": completed.returncode == 0 and not rate_limited,
        "agent": "claude",
        "task_id": task_id,
        "returncode": completed.returncode,
        "rate_limited": rate_limited,
        "rate_limit_hint": rate_limit_hint,
        "accepted": accepted,
        "verdict": verdict,
        "error": error_text,
        "command": command,
        "stdout_file": str(stdout_path.relative_to(ROOT)),
        "stderr_file": str(stderr_path.relative_to(ROOT)),
        "review_file": str(review_path.relative_to(ROOT)),
        "review": review,
        "finished_at": now_iso(),
    }
    save_json(review_path, result)
    return result


def start_next_task() -> int:
    try:
        queue = load_json(QUEUE_FILE)
        agents = load_json(AGENTS_FILE)

        errors = []
        errors.extend(validate_required_files())
        errors.extend(validate_agents_config(agents))
        if errors:
            print(json.dumps({"ok": False, "errors": errors}, ensure_ascii=False, indent=2))
            return 2

        if queue.get("active_task") is not None:
            print(json.dumps({
                "ok": False,
                "error": "Есть активная задача. Сначала заверши или сбрось её."
            }, ensure_ascii=False, indent=2))
            return 3

        tasks = queue.get("tasks", [])
        if not tasks:
            print(json.dumps({
                "ok": True,
                "message": "Очередь пуста"
            }, ensure_ascii=False, indent=2))
            return 0

        task = tasks.pop(0)
        task["target_system"] = normalize_target_system(task.get("target_system"))
        task["status"] = "running"
        task["stage"] = "analysis"
        task["task_class"] = normalize_task_class(task.get("task_class"))
        task["review_policy"] = normalize_review_policy(task.get("review_policy"))
        task["review_round"] = normalize_review_round(task.get("review_round"))
        task["max_review_rounds"] = normalize_max_review_rounds(task.get("max_review_rounds"))
        task["revise_requested"] = task.get("revise_requested") is True
        task["required_changes"] = sanitize_string_list(task.get("required_changes"))
        task["out_of_scope_requests"] = sanitize_string_list(task.get("out_of_scope_requests"))
        task["updated_at"] = now_iso()

        queue["active_task"] = task
        queue["tasks"] = tasks
        save_json(QUEUE_FILE, queue)
        return run_active_codex_pass(
            active_task=task,
            queue=queue,
            agents=agents,
            requeue_on_rate_limit=True,
        )


    except Exception as e:
        set_state(
            running=False,
            current_stage="start",
            current_agent="codex",
            last_error=str(e),
        )
        print(f"[ERROR] {e}")
        return 1


def review_once() -> int:
    try:
        queue = load_json(QUEUE_FILE)
        agents = load_json(AGENTS_FILE)

        errors = []
        errors.extend(validate_required_files())
        errors.extend(validate_agents_config(agents))
        if errors:
            print(json.dumps({"ok": False, "errors": errors}, ensure_ascii=False, indent=2))
            return 2

        active_task = queue.get("active_task")
        if not isinstance(active_task, dict):
            print(json.dumps({
                "ok": True,
                "message": "review-once: активной задачи для review нет",
            }, ensure_ascii=False, indent=2))
            return 0

        status = str(active_task.get("status", "")).strip()
        stage = str(active_task.get("stage", "")).strip()
        if status != "review" or stage != "review":
            print(json.dumps({
                "ok": True,
                "message": "review-once: активная задача не в review",
                "task_id": active_task.get("task_id"),
                "status": status,
                "stage": stage,
            }, ensure_ascii=False, indent=2))
            return 0

        if not should_require_claude_review(active_task):
            set_state(
                running=False,
                current_stage="auto_complete",
                current_agent=None,
                last_error=None,
            )
            return complete_without_claude()

        next_retry = active_task.get("next_claude_retry_after")
        if active_task.get("claude_rate_limited") is True and not retry_is_due(next_retry):
            set_state(
                running=False,
                current_stage="claude_cooldown",
                current_agent=None,
                last_error=active_task.get("last_claude_error"),
            )
            print(json.dumps({
                "ok": True,
                "message": "review-once: cooldown Claude ещё не прошёл",
                "task_id": active_task.get("task_id"),
                "next_claude_retry_after": next_retry,
            }, ensure_ascii=False, indent=2))
            return 0

        active_task["last_claude_attempt_at"] = now_iso()
        active_task["review_attempts"] = int(active_task.get("review_attempts") or 0) + 1
        active_task["updated_at"] = now_iso()
        queue["active_task"] = active_task
        save_json(QUEUE_FILE, queue)

        set_state(
            running=True,
            current_stage="review",
            current_agent="claude",
            last_error=None,
        )

        result = run_claude_task(active_task, agents)

        queue = load_json(QUEUE_FILE)
        active_task = queue.get("active_task") or active_task
        active_task["claude_result"] = result
        active_task["updated_at"] = now_iso()

        if result.get("rate_limited"):
            active_task["status"] = "review"
            active_task["stage"] = "review"
            active_task["claude_rate_limited"] = True
            active_task["last_claude_error"] = result.get("error")
            active_task["next_claude_retry_after"] = retry_after_iso(CLAUDE_RETRY_MINUTES)
            queue["active_task"] = active_task
            save_json(QUEUE_FILE, queue)

            set_state(
                running=False,
                current_stage="claude_rate_limited",
                current_agent=None,
                last_error=result.get("error"),
            )

            print(json.dumps({
                "ok": False,
                "message": "Claude упёрся в лимит. Задача остаётся в review.",
                "task_id": active_task.get("task_id"),
                "next_claude_retry_after": active_task["next_claude_retry_after"],
                "stdout_file": result.get("stdout_file"),
                "stderr_file": result.get("stderr_file"),
            }, ensure_ascii=False, indent=2))
            return 0

        active_task["claude_rate_limited"] = False
        active_task["last_claude_error"] = result.get("error")
        active_task["next_claude_retry_after"] = None
        queue["active_task"] = active_task
        save_json(QUEUE_FILE, queue)

        if result.get("ok") and result.get("accepted") is True:
            return complete_active_task("done", "Claude review accepted")
        review_payload = result.get("review") if isinstance(result.get("review"), dict) else {}
        verdict = str(result.get("verdict") or review_payload.get("verdict") or "").strip().lower()

        if verdict == "revise":
            current_round = normalize_review_round(active_task.get("review_round"))
            max_rounds = normalize_max_review_rounds(active_task.get("max_review_rounds"))
            if current_round >= max_rounds:
                active_task["out_of_scope_requests"] = sanitize_string_list(review_payload.get("out_of_scope_requests"))
                queue["active_task"] = active_task
                save_json(QUEUE_FILE, queue)
                return complete_active_task("failed", "Claude requested additional revise beyond bounded limit")

            active_task["review_round"] = current_round + 1
            active_task["revise_requested"] = True
            active_task["required_changes"] = sanitize_string_list(review_payload.get("required_changes"))
            active_task["out_of_scope_requests"] = sanitize_string_list(review_payload.get("out_of_scope_requests"))
            active_task["status"] = "running"
            active_task["stage"] = "execution"
            active_task["updated_at"] = now_iso()
            queue["active_task"] = active_task
            save_json(QUEUE_FILE, queue)

            set_state(
                running=False,
                current_stage="revise_requested",
                current_agent=None,
                last_error=None,
            )

            print(json.dumps({
                "ok": True,
                "message": "Claude запросил один bounded revise-pass. Задача возвращена Codex.",
                "task_id": active_task.get("task_id"),
                "review_round": active_task["review_round"],
                "max_review_rounds": max_rounds,
                "required_changes": active_task["required_changes"],
                "out_of_scope_requests": active_task["out_of_scope_requests"],
            }, ensure_ascii=False, indent=2))
            return 0

        return complete_active_task("failed", "Claude review rejected or failed")

    except Exception as e:
        set_state(
            running=False,
            current_stage="review_once",
            current_agent="claude",
            last_error=str(e),
        )
        print(f"[ERROR] {e}")
        return 1


def worker_once() -> int:
    try:
        queue = load_json(QUEUE_FILE)
        agents = load_json(AGENTS_FILE)

        errors = []
        errors.extend(validate_required_files())
        errors.extend(validate_agents_config(agents))
        if errors:
            print(json.dumps({"ok": False, "errors": errors}, ensure_ascii=False, indent=2))
            return 2

        active_task = queue.get("active_task")

        if isinstance(active_task, dict):
            status = str(active_task.get("status", "")).strip()
            stage = str(active_task.get("stage", "")).strip()
            task_id = str(active_task.get("task_id", "")).strip()

            if status == "review" and stage == "review":
                return review_once()

            if status == "running" and stage == "execution":
                next_retry = active_task.get("next_codex_retry_after")
                if active_task.get("codex_rate_limited") is True and not retry_is_due(next_retry):
                    set_state(
                        running=False,
                        current_stage="codex_cooldown",
                        current_agent=None,
                        last_error=active_task.get("last_codex_error"),
                    )
                    print(json.dumps({
                        "ok": True,
                        "message": "worker-once: cooldown Codex ещё не прошёл для revise-pass",
                        "task_id": task_id,
                        "next_codex_retry_after": next_retry,
                    }, ensure_ascii=False, indent=2))
                    return 0
                return run_active_codex_pass(
                    active_task=active_task,
                    queue=queue,
                    agents=agents,
                    requeue_on_rate_limit=False,
                )

            state = load_json(STATE_FILE)
            state["current_stage"] = stage or status or state.get("current_stage") or "idle"
            state["last_error"] = None
            state["last_update"] = now_iso()
            save_json(STATE_FILE, state)

            print(json.dumps({
                "ok": True,
                "message": "worker-once: новая задача не запущена, активная задача уже существует",
                "task_id": task_id,
                "status": status,
                "stage": stage,
            }, ensure_ascii=False, indent=2))
            return 0

        if active_task is not None:
            set_state(
                running=False,
                current_stage="blocked",
                current_agent=None,
                last_error="Некорректный active_task в orchestrator_queue.json",
            )

            print(json.dumps({
                "ok": False,
                "message": "worker-once: новая задача не запущена, active_task повреждён",
                "active_task_type": type(active_task).__name__,
            }, ensure_ascii=False, indent=2))
            return 3

        tasks = queue.get("tasks", [])
        if not tasks:
            set_state(
                running=False,
                current_stage="idle",
                current_agent=None,
                last_error=None,
            )

            print(json.dumps({
                "ok": True,
                "message": "worker-once: очередь пуста"
            }, ensure_ascii=False, indent=2))
            return 0

        first_task = tasks[0]
        if isinstance(first_task, dict):
            next_retry = first_task.get("next_codex_retry_after")
            if first_task.get("codex_rate_limited") is True and not retry_is_due(next_retry):
                set_state(
                    running=False,
                    current_stage="codex_cooldown",
                    current_agent=None,
                    last_error=first_task.get("last_codex_error"),
                )
                print(json.dumps({
                    "ok": True,
                    "message": "worker-once: cooldown Codex ещё не прошёл",
                    "task_id": first_task.get("task_id"),
                    "next_codex_retry_after": next_retry,
                }, ensure_ascii=False, indent=2))
                return 0

        return start_next_task()

    except Exception as e:
        set_state(
            running=False,
            current_stage="worker_once",
            current_agent=None,
            last_error=str(e),
        )
        print(f"[ERROR] {e}")
        return 1


def complete_active_task(final_status: str, note: str | None = None, force: bool = False) -> int:
    if final_status not in {"done", "failed"}:
        print(json.dumps({
            "ok": False,
            "error": "final_status должен быть done или failed",
        }, ensure_ascii=False, indent=2))
        return 2

    try:
        queue = load_json(QUEUE_FILE)
        active_task = queue.get("active_task")

        if active_task is None:
            set_state(
                running=False,
                current_stage="idle",
                current_agent=None,
                last_error=None,
            )
            print(json.dumps({
                "ok": True,
                "message": "Активной задачи нет",
            }, ensure_ascii=False, indent=2))
            return 0

        if not isinstance(active_task, dict):
            set_state(
                running=False,
                current_stage="blocked",
                current_agent=None,
                last_error="Некорректный active_task в orchestrator_queue.json",
            )
            print(json.dumps({
                "ok": False,
                "error": "active_task повреждён, автоматическое завершение запрещено",
                "active_task_type": type(active_task).__name__,
            }, ensure_ascii=False, indent=2))
            return 3

        status = str(active_task.get("status", "")).strip()
        if status == "running" and not force:
            print(json.dumps({
                "ok": False,
                "error": "Задача сейчас running. Для принудительного закрытия нужен --force.",
                "task_id": active_task.get("task_id"),
            }, ensure_ascii=False, indent=2))
            return 4

        finished_at = now_iso()
        active_task["status"] = final_status
        active_task["stage"] = "done"
        active_task["final_status"] = final_status
        active_task["completed_at"] = finished_at
        active_task["updated_at"] = finished_at
        if note:
            active_task["completion_note"] = note

        history = queue.get("history", [])
        if not isinstance(history, list):
            history = []
        history.append(active_task)

        queue["active_task"] = None
        queue["history"] = history
        save_json(QUEUE_FILE, queue)

        set_state(
            running=False,
            current_stage="idle",
            current_agent=None,
            last_error=None,
        )

        print(json.dumps({
            "ok": True,
            "message": "Активная задача перенесена в history",
            "task_id": active_task.get("task_id"),
            "final_status": final_status,
            "history_tasks": len(history),
        }, ensure_ascii=False, indent=2))
        return 0

    except Exception as e:
        set_state(
            running=False,
            current_stage="complete",
            current_agent=None,
            last_error=str(e),
        )
        print(f"[ERROR] {e}")
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Локальный оркестратор Codex + Claude")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("status", help="Показать статус оркестратора")

    enqueue_parser = subparsers.add_parser("enqueue", help="Поставить задачу в очередь")
    enqueue_parser.add_argument("task_file", help="Путь к JSON-файлу задачи")

    subparsers.add_parser("start", help="Взять первую задачу из очереди и запустить Codex")
    subparsers.add_parser("worker-once", help="Один безопасный цикл для Планировщика Windows")
    subparsers.add_parser("review-once", help="Один безопасный запуск Claude review для active_task")

    complete_parser = subparsers.add_parser("complete", help="Перенести активную задачу в history")
    complete_parser.add_argument("--final-status", choices=("done", "failed"), default="done")
    complete_parser.add_argument("--note", default=None)
    complete_parser.add_argument("--force", action="store_true")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command in (None, "status"):
        return print_status()

    if args.command == "enqueue":
        return enqueue_task(args.task_file)

    if args.command == "start":
        return start_next_task()

    if args.command == "worker-once":
        return worker_once()

    if args.command == "review-once":
        return review_once()

    if args.command == "complete":
        return complete_active_task(args.final_status, args.note, args.force)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
