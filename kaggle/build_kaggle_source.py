#!/usr/bin/env python3
"""
Generate a single self-contained C++ file for Kaggle by concatenating
the Madhava L2 library + the BIGANN benchmark into one translation unit.

Output: winnex_madhava_cpp/kaggle/winnex_madhava_bench_standalone.cpp
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def read(rel):
    with open(os.path.join(ROOT, rel)) as f:
        return f.read()

hdr = read('include/winnex_madhava/winnex_madhava.hpp')
impl = read('src/winnex_madhava.cpp')
bench = read('examples/bench_bigann_l2.cpp')

# Drop the include of the header inside impl/bench (the header is inlined above),
# and drop the project-internal system headers that would need a project root.
impl = impl.replace('#include "winnex_madhava/winnex_madhava.hpp"', '')
bench = bench.replace('#include "winnex_madhava/winnex_madhava.hpp"', '')
# Keep a single header include of the std headers that both need.
import re
# strip duplicate <sys/mman.h> etc. is fine — include guards handle it.

out = []
out.append("// ==== winnex_madhava.hpp (inlined) ====")
out.append(hdr)
out.append("// ==== winnex_madhava.cpp (implementation) ====")
out.append(impl)
out.append("// ==== bench_bigann_l2.cpp (main) ====")
out.append(bench)

src = "\n".join(out)
os.makedirs(os.path.join(ROOT, 'kaggle'), exist_ok=True)
dst = os.path.join(ROOT, 'kaggle', 'winnex_madhava_bench_standalone.cpp')
with open(dst, 'w') as f:
    f.write(src)
print(f"Gerado: {dst} ({len(src)} chars, {src.count(chr(10))} linhas)")
