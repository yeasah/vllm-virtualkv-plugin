"""The plugin, evicting nothing, must be indistinguishable from not having it.

This is the test that keeps a mechanism bug from being read later as a quality
result. Everything else about a paged run changes the output on purpose, so
there is nothing to compare it against; a budget of zero changes nothing on
purpose, so it can be compared against everything. If this fails, no accuracy
number taken with the plugin means anything, whatever it says.

Needs a GPU and a model, so it is opt-in:

    VIRTUALKV_TEST_MODEL=unsloth/Llama-3.2-1B-Instruct pytest tests/

Each arm runs in its own process. An engine does not fully release its KV pool
when the object goes away, and sharing an allocator between the arms would mean
the second one runs on state the first one shaped.
"""

import json
import os
import subprocess
import sys
import tempfile

import pytest

MODEL = os.environ.get("VIRTUALKV_TEST_MODEL")
pytestmark = pytest.mark.skipif(
    not MODEL, reason="set VIRTUALKV_TEST_MODEL to run the control arm")

CHILD = r'''
import json, os, sys
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
sys.path.insert(0, os.path.join(sys.argv[3], "tools"))
from harness import haystack
from vllm import LLM, SamplingParams

llm = LLM(model=sys.argv[1], max_model_len=1200, gpu_memory_utilization=0.55,
          enforce_eager=True, enable_prefix_caching=False, max_num_seqs=1,
          trust_remote_code=True)
tok = llm.get_tokenizer()
out = llm.generate([{"prompt_token_ids": haystack(tok, 1024)[:1017]}],
                   SamplingParams(temperature=0.0, max_tokens=16,
                                  logprobs=5, ignore_eos=True))[0]
mgrs = (llm.llm_engine.engine_core.engine_core.scheduler
        .kv_cache_manager.coordinator.single_type_managers)
res = {"ids": [int(t) for t in out.outputs[0].token_ids],
       "manager": type(mgrs[0]).__name__,
       "freed": int(getattr(mgrs[0], "blocks_freed", 0)),
       "logprobs": [
           {str(t): round(lp.logprob, 6) for t, lp in step.items()}
           for step in (out.outputs[0].logprobs or [])]}
json.dump(res, open(sys.argv[2], "w"))
'''


def run(env_extra, tmp, tag):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(tmp, "child.py")
    with open(script, "w") as f:
        f.write(CHILD)
    out = os.path.join(tmp, f"{tag}.json")
    env = {**os.environ, **env_extra}
    proc = subprocess.run([sys.executable, script, MODEL, out, root],
                          capture_output=True, text=True, env=env)
    assert proc.returncode == 0, proc.stderr[-3000:]
    with open(out) as f:
        return json.load(f)


def test_zero_budget_is_indistinguishable_from_no_plugin():
    with tempfile.TemporaryDirectory() as tmp:
        off = run({"VLLM_VIRTUALKV": "0"}, tmp, "off")
        control = run({"VLLM_VIRTUALKV": "1", "VLLM_VIRTUALKV_BUDGET": "0",
                       "VLLM_VIRTUALKV_POLICY": "full"}, tmp, "control")

    assert off["manager"] == "FullAttentionManager", (
        "the plugin engaged in the arm that disabled it")
    assert control["manager"] == "PagedAttentionManager", (
        "the plugin did not engage, so this compares nothing to nothing")
    assert control["freed"] == 0, (
        f"the control arm freed {control['freed']} blocks; it must evict none")
    assert control["ids"] == off["ids"], "the control arm changed the tokens"
    assert control["logprobs"] == off["logprobs"], (
        "the control arm changed the distributions without changing the tokens "
        "-- identical output here is necessary but not sufficient")
