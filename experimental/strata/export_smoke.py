#!/usr/bin/env python3
"""Train synthetic heads, export C weights, compile and verify their predictions.

This checks the Python-to-C boundary, not any real model. --emit-only DIR is useful
when Python and the C compiler run in different environments (e.g. Windows/WSL).
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

import heads


def prepare(directory):
    directory = Path(directory)
    meta = {"schema": heads.TRACE_SCHEMA, "family": "qwen38",
            "checkpoint_revision": "a" * 40, "tokenizer_sha256": "b" * 64,
            "hidden_tap": "synthetic_fixture_readout_🦜", "quantization": 'fixture-f32_"quoted"',
            "hidden_size": 2, "vocab_size": 4}
    rows = [{"hidden": [1., 0.], "future_tokens": [0, 1, 2]},
            {"hidden": [0., 1.], "future_tokens": [0, 3, 1]}]
    trace = directory / "trace.jsonl"
    trace.write_text("\n".join(json.dumps(row) for row in [meta, *rows]) + "\n", encoding="utf-8")
    pack = heads.train(trace, [2, 3], 2, 100, 2, 0.05, 42)
    (directory / "weights.h").write_text(heads.export_c(pack), encoding="utf-8")
    (directory / "smoke.c").write_text('''#include <stdio.h>
#include <string.h>
#include "weights.h"
int main(void) {
    float hidden[2][2]={{1,0},{0,1}}, workspace[4], logits[8];
    int tokens[2], expected[2][2]={{1,2},{3,1}};
    size_t w,l;
    if(strata_heads_validate(&strata_trained_bank)!=STRATA_HEADS_OK) return 1;
    if(strata_heads_sizes(&strata_trained_bank,&w,&l)!=STRATA_HEADS_OK || w!=4 || l!=8) return 2;
    for(int parallel=0;parallel<2;parallel++) for(int row=0;row<2;row++) {
        if(strata_heads_predict(&strata_trained_bank,hidden[row],2,workspace,4,
             logits,8,tokens,2,parallel)!=STRATA_HEADS_OK) return 3;
        if(memcmp(tokens,expected[row],sizeof tokens)) return 4;
    }
    puts("trained Python -> exported C heads: prediction parity passed");
    return 0;
}
''', encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-only", type=Path)
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    args = parser.parse_args()
    if args.emit_only:
        args.emit_only.mkdir(parents=True, exist_ok=False)
        prepare(args.emit_only)
        print(f"Prepared synthetic training/export fixture in {args.emit_only}; compilation still required")
        return
    with tempfile.TemporaryDirectory(prefix="tmp_strata_export_", dir=heads.ROOT / "c/tests") as directory:
        prepare(directory)
        directory = Path(directory)
        binary = directory / ("smoke.exe" if os.name == "nt" else "smoke")
        subprocess.run([*shlex.split(args.cc), "-std=c99", "-O2", "-Wall", "-Wextra",
                        "-Werror", "-fopenmp", "-I", str(heads.ROOT / "c"),
                        str(directory / "smoke.c"), "-lm", "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True, env=os.environ | {"OMP_NUM_THREADS": "4"})


if __name__ == "__main__":
    main()
