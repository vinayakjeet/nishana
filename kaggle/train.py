"""Train one Nishana variant on a Kaggle GPU session.

    # session 1 (fresh run)
    python kaggle/train.py --tier synthetic-only --size 500 --rank 16

    # session 2 (Kaggle restarted the kernel; same command plus:)
    python kaggle/train.py --tier synthetic-only --size 500 --rank 16 --resume

    # after the final session, produce predictions for the eval sweep
    python kaggle/train.py --tier synthetic-only --size 500 --rank 16 --predict

Runs inside a Kaggle notebook cell (`!python kaggle/train.py ...`), which is why
unsloth is imported late: this repo's tests and CI never need it installed.

Checkpoints are adapter-only by design. Kaggle sessions die without warning and
the next one may land on a different GPU, so the only artifact guaranteed to
survive is the small one. Optimizer state does not transfer across sessions;
the loss discontinuity at each boundary is visible in the logged history and is
reported rather than smoothed over.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nishana.data.curation import TrainingRow, variant_id  # noqa: E402
from nishana.data.hashing import content_hash  # noqa: E402
from nishana.data.loader import load_jsonl  # noqa: E402
from nishana.train.checkpoints import (  # noqa: E402
    AdapterManifest,
    check_resume,
    load_manifest,
    save_manifest,
)
from nishana.train.config import DEFAULT_BASE_MODEL, TrainConfig  # noqa: E402
from nishana.train.schedule import plan_sessions, total_steps  # noqa: E402
from nishana.types import Prediction  # noqa: E402

CHECKPOINT_ROOT = Path("checkpoints")
PREDICTIONS_DIR = Path("results/ablation")
TRAINING_DIR = Path("var/training")
EVAL_DATASET = Path("datasets/tickets-eval.jsonl")

INSTRUCTION = """Label the Hinglish support ticket. Reply with exactly one JSON line:
{{"intent": "<order_status|refund|account|other>",
  "entities": [{{"type": "...", "value": "..."}}]}}
Copy entity values exactly as written in the ticket.

Ticket:
{ticket}"""

ALPACA_TEMPLATE = """Below is an instruction that describes a task. \
Write a response that completes the request.

### Instruction:
{instruction}

### Response:
{output}"""


def render_row(row: TrainingRow) -> str:
    entities = sorted(row.entities, key=lambda e: (e.get("type", ""), e.get("value", "")))
    return json.dumps({"intent": row.intent, "entities": entities}, ensure_ascii=False)


def training_texts(rows: list[TrainingRow]) -> list[str]:
    out = []
    for row in rows:
        instruction = INSTRUCTION.format(ticket=row.text)
        out.append(ALPACA_TEMPLATE.format(instruction=instruction, output=render_row(row)))
    return out


def load_training_rows(tier: str, size: int) -> tuple[list[TrainingRow], str]:
    path = TRAINING_DIR / f"{tier}-{size}.jsonl"
    raw = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [TrainingRow.model_validate(item) for item in raw[:size]]
    return rows, content_hash([row.model_dump() for row in rows])


def make_config(tier: str, size: int, rank: int) -> TrainConfig:
    return TrainConfig(
        variant_id=variant_id(tier, size, DEFAULT_BASE_MODEL, rank),
        curation_tier=tier,
        dataset_size=size,
        lora_r=rank,
        lora_alpha=32 if rank >= 16 else rank * 2,
    )


def checkpoint_dir(config: TrainConfig) -> Path:
    return CHECKPOINT_ROOT / config.variant_id


def train_session(
    config: TrainConfig,
    rows: list[TrainingRow],
    dataset_hash: str,
    minutes_per_session: float,
    seconds_per_step: float,
    resume: bool,
) -> Path:
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    from datasets import Dataset

    out = checkpoint_dir(config)
    manifest: AdapterManifest | None = None
    if resume:
        manifest = load_manifest(out)
        check_resume(manifest, config, dataset_hash)

    model, tokenizer = FastLanguageModel.from_pretrained(
        str(out if manifest else config.base_model),
        max_seq_length=config.max_seq_len,
        load_in_4bit=True,
    )
    if not manifest:
        model = FastLanguageModel.get_peft_model(
            model,
            r=config.lora_r,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )

    steps_total = total_steps(len(rows), config.epochs, config.effective_batch)
    plan = plan_sessions(steps_total, minutes_per_session, seconds_per_step)
    session_number = (manifest.session + 1) if manifest else 1
    step_start = (manifest.step_end if manifest else 0) + 1
    if step_start > steps_total:
        raise SystemExit("training already complete; drop --resume")

    budget = plan.sessions[session_number - 1][1] - plan.sessions[session_number - 1][0]
    print(
        f"session {session_number}/{plan.n_sessions}: steps {step_start}"
        f"..{step_start - 1 + budget} of {steps_total} "
        f"(planned from {seconds_per_step:.2f}s/step, {minutes_per_session:.0f} min cap)"
    )

    dataset = Dataset.from_dict({"text": training_texts(rows)})
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        args=SFTConfig(
            max_seq_length=config.max_seq_len,
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.grad_accum,
            learning_rate=config.learning_rate,
            warmup_ratio=config.warmup_ratio,
            lr_scheduler_type=config.scheduler,
            seed=config.seed,
            logging_steps=10,
            output_dir=str(out / "trainer"),
            # max_steps bounds this session exactly: Kaggle ends runs mid-step,
            # so the plan budgets conservatively and stops itself before the
            # wall does.
            max_steps=budget,
            report_to="none",
        ),
    )
    history_before = list(trainer.state.log_history)
    trainer.train()
    _report_loss_jump(history_before, trainer.state.log_history, session_number)

    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))

    new_manifest = AdapterManifest(
        variant_id=config.variant_id,
        base_model=config.base_model,
        base_revision=config.base_revision,
        dataset_hash=dataset_hash,
        curation_tier=config.curation_tier,
        dataset_size=config.dataset_size,
        lora_r=config.lora_r,
        step_start=step_start,
        step_end=step_start - 1 + budget,
        epoch_start=round((step_start - 1) * config.effective_batch / len(rows), 3),
        epoch_end=round((step_start - 1 + budget) * config.effective_batch / len(rows), 3),
        session=session_number,
    )
    save_manifest(out, new_manifest)
    print(f"adapter saved to {out}")
    if new_manifest.step_end < steps_total:
        print(f"incomplete: {steps_total - new_manifest.step_end} steps remain; "
              f"start a new Kaggle session and rerun with --resume")
    return out


def _report_loss_jump(before: list[dict], after: list[dict], session: int) -> None:
    """Print the loss discontinuity at a session boundary instead of hiding it.

    Resuming without optimizer state costs a little momentum. The number belongs
    in the log where anyone computing the run's story can see it.
    """
    if before:
        last = next((h["loss"] for h in reversed(before) if "loss" in h), None)
        first = next((h["loss"] for h in after if "loss" in h), None)
        if last is not None and first is not None:
            print(f"session boundary {session}: loss {last:.4f} -> {first:.4f} "
                  f"(jump {first - last:+.4f}, expected when optimizer state resets)")


def predict(args: argparse.Namespace) -> int:
    from unsloth import FastLanguageModel

    config = make_config(args.tier, args.size, args.rank)
    out = checkpoint_dir(config)
    manifest = load_manifest(out)

    model, tokenizer = FastLanguageModel.from_pretrained(
        str(out), max_seq_length=config.max_seq_len, load_in_4bit=True
    )
    FastLanguageModel.for_inference(model)

    tickets = load_jsonl(str(EVAL_DATASET))
    rows: list[Prediction] = []
    model_name = f"{config.variant_id}@{manifest.adapter_sha256[:8]}"
    for ticket in tickets:
        instruction = INSTRUCTION.format(ticket=ticket.input["prompt"])
        prompt = ALPACA_TEMPLATE.format(instruction=instruction, output="")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        generated = model.generate(
            **inputs, max_new_tokens=160, temperature=None, do_sample=False
        )
        text = tokenizer.decode(
            generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        rows.append(
            Prediction(
                item_id=ticket.id,
                prediction=text.strip(),
                model=model_name,
                generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
            )
        )

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    target = PREDICTIONS_DIR / f"{config.tier}-{config.dataset_size}-r{config.lora_r}.jsonl"
    target.write_text("\n".join(r.model_dump_json() for r in rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} predictions to {target}")
    print("next: add this file to results/runs.yaml, then scripts/train_eval.py scores it.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tier", required=True,
        choices=("synthetic-only", "mixed", "verified-heavy"),
    )
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument(
        "--minutes-per-session", type=float, default=330.0,
        help="Kaggle T4 sessions run about 5.5 hours before the clock cuts",
    )
    parser.add_argument(
        "--seconds-per-step", type=float, default=2.0,
        help="measured value from a prior session's log, never guessed",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--predict", action="store_true")
    args = parser.parse_args()

    if args.predict:
        return predict(args)

    config = make_config(args.tier, args.size, args.rank)
    rows, dataset_hash = load_training_rows(args.tier, args.size)
    train_session(
        config, rows, dataset_hash,
        args.minutes_per_session, args.seconds_per_step, args.resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
