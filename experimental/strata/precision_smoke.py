#!/usr/bin/env python3
"""Compile cross-language INT4/8/16 bytes and prediction checks on synthetic heads."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

import heads
import precision
from export_smoke import prepare as prepare_f32


def prepare(directory):
    directory = Path(directory)
    prepare_f32(directory)
    pack = heads.train(directory / "trace.jsonl", [2, 3], 2, 100, 2, 0.05, 42)
    (directory / "pack.json").write_text(json.dumps(pack, allow_nan=False), encoding="utf-8")
    text = ['#include <math.h>', '#include <stdio.h>', '#include <stdlib.h>', '#include <string.h>']
    for bits in (4, 8, 16):
        (directory / f"weights{bits}.h").write_text(precision.export_c(pack, bits, 3, f"bank{bits}"), encoding="utf-8")
        text.append(f'#include "weights{bits}.h"')
    text.append(heads.export_c(pack, "original"))
    text.append('''int main(void) {
      if(strata_heads_validate(&original_bank)!=STRATA_HEADS_OK ||
         strcmp(bank4_family,original_family) || strcmp(bank4_checkpoint_revision,original_checkpoint_revision) ||
         strcmp(bank4_tokenizer_sha256,original_tokenizer_sha256) || strcmp(bank4_hidden_tap,original_hidden_tap) ||
         strcmp(bank4_quantization,original_quantization)) return 8;
      const strata_quant_heads_bank *banks[]={&bank4_bank,&bank8_bank,&bank16_bank};
      unsigned bits[]={4,8,16}; float hidden[2][2]={{1,0},{0,1}};
      int expected[2][2]={{1,2},{3,1}};
      for(int p=0;p<3;p++) {
        size_t wc,lc;
        if(strata_quant_heads_sizes(banks[p],&wc,&lc)!=COLI_STRATA_OK) return 1;
        float *work=malloc(wc*sizeof(float)), *logits=malloc(lc*sizeof(float));
        if(!work || !logits) return 2;
        for(int h=0;h<2;h++) for(int part=0;part<2;part++) {
          const float *src=part?original_heads[h].up:original_heads[h].down;
          size_t n=part?original_heads[h].up_count:original_heads[h].down_count;
          const uint8_t *wanted=part?banks[p]->heads[h].up:banks[p]->heads[h].down;
          size_t wanted_bytes=part?banks[p]->heads[h].up_bytes:banks[p]->heads[h].down_bytes;
          size_t need,got; if(strata_quant_size(n,3,bits[p],&need)!=COLI_STRATA_OK) return 3;
          uint8_t *bytes=malloc(need); if(!bytes) return 4;
          if(strata_quant_encode(src,n,3,bits[p],bytes,need,&got)!=COLI_STRATA_OK || got!=wanted_bytes || memcmp(bytes,wanted,got)) return 5;
          free(bytes);
        }
        for(int parallel=0;parallel<2;parallel++) for(int row=0;row<2;row++) {
          int tokens[2];
          if(strata_quant_heads_predict(banks[p],hidden[row],2,work,wc,logits,lc,tokens,2,parallel)!=COLI_STRATA_OK) return 6;
          if(memcmp(tokens,expected[row],sizeof tokens)) return 7;
        }
        free(work); free(logits);
      }
      puts("INT4/INT8/INT16: C/Python bytes and parallel prediction parity passed");
      return 0;
    }''')
    (directory / "precision_smoke.c").write_text("\n".join(text), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emit-only", type=Path)
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    args = parser.parse_args()
    if args.emit_only:
        args.emit_only.mkdir(parents=True, exist_ok=False)
        prepare(args.emit_only)
        print(f"Prepared precision fixture in {args.emit_only}; compilation still required")
        return
    with tempfile.TemporaryDirectory(prefix="tmp_strata_precision_", dir=heads.ROOT / "c/tests") as directory:
        prepare(directory)
        directory = Path(directory)
        binary = directory / ("smoke.exe" if os.name == "nt" else "smoke")
        subprocess.run([*shlex.split(args.cc), "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
                        "-fopenmp", "-I", str(heads.ROOT / "c"), str(directory / "precision_smoke.c"),
                        "-lm", "-o", str(binary)], check=True)
        subprocess.run([str(binary)], check=True, env=os.environ | {"OMP_NUM_THREADS": "4"})


if __name__ == "__main__":
    main()
