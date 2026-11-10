"""Run one cast_compile conversation (docs/cast_compile_plan.md).

conditions: A = python + evaluate (prompt_A.txt). (B, C and D follow the plan; they need their own prompt files.)
The evaluate tool keeps one bf16 Qwen + refusal judge loaded on the GPU in this driver process.
"""
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from wsp import loop, sandbox                       # noqa: E402

CASES = Path("/scratch/scholar/pande1/wsp/cases")
RUNS = HERE.parents[1] / "runs" / "cast_compile"
CONDITIONS = {"A": dict(prompt="prompt_A.txt", max_runs=25, nudge=loop.Nudge(below=10, max=1),
                        tools=("python", "evaluate"))}


def validate(answer):
    j = loop.last_json(answer)
    if j is None:
        return None, "no JSON block"
    if "edit_file" not in j or "mechanism" not in j:
        return j, "JSON needs edit_file and mechanism"
    return j, None


class EvaluateTool:
    def __init__(self, work, log):
        import torch
        sys.path.insert(0, str(HERE))
        import evaluate as ev
        self.ev, self.work, self.log = ev, Path(work), Path(log)
        self.lm = ev.LM("qwen1.5-1.8b-chat", dtype=torch.bfloat16)
        self.lm.model.requires_grad_(False)
        self.judge = ev.RefusalJudge()

    def __call__(self, body, k):
        p = self.work / "edit.npz"
        if not p.exists():
            return "No /work/edit.npz yet. Write one with save_edit(edit) first."
        try:
            edit = self.ev.load_edit(p)
            r = self.ev.dev_report(self.lm, self.judge, edit, n=32, seed=k)
        except Exception as e:                     # report problems to the model instead of crashing the driver
            r = {"error": f"{type(e).__name__}: {e}"}
        with open(self.log, "a") as fh:
            fh.write(json.dumps({"run": k, "tool": "evaluate", "result": r}) + "\n")
        return f"EVALUATE call {k} (/work/edit.npz, 32 harmful + 32 harmless held-out prompts, not in /case/dev):\n" + json.dumps(r, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=list(CONDITIONS), default="A")
    ap.add_argument("--case", default="cast_compile_qwen1.5-1.8b-chat")
    ap.add_argument("--effort", default="xhigh")
    ap.add_argument("--max-runs", type=int)
    ap.add_argument("--tag", default="")
    ap.add_argument("--model", default=loop.codex.MODEL)
    a = ap.parse_args()
    R = CONDITIONS[a.condition]
    max_runs = a.max_runs or R["max_runs"]
    case = CASES / a.case
    assert (case / "cast" / "cast.json").exists(), case
    tag = "_".join(x for x in (a.tag, "" if a.model == loop.codex.MODEL else a.model) if x)
    out = RUNS / (a.condition + (f"_{tag}" if tag else "")) / a.case
    work = out / "work"
    prompt = (HERE / R["prompt"]).read_text()
    if max_runs != R["max_runs"]:
        prompt = prompt.replace(f"up to {R['max_runs']} times", f"up to {max_runs} times")
    sb = sandbox.Sandbox(work, case_dir=case, venv="numpy", prelude=(HERE / "prelude.py").read_text(),
                         timeout=600, mem=48 << 30, max_out=15_000, log=out / "tool_calls.jsonl")
    tools = {"python": lambda body, k: sb.run(body, {"run": k, "tool": "python"})}
    if "evaluate" in R["tools"]:
        tools["evaluate"] = EvaluateTool(work, out / "tool_calls.jsonl")
    conv = loop.Conversation(out, prompt, effort=a.effort, max_runs=max_runs, tools=tools, validate=validate,
                             nudge=R["nudge"], model=a.model)
    s = conv.run()
    print(json.dumps({k: s[k] for k in ("turn", "runs", "done", "nudges", "format_retries", "cum_input")}))


if __name__ == "__main__":
    main()
