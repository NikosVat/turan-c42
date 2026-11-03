# third_party (read-only clones, not committed)

Recreate with:

```bash
git clone https://github.com/Bakeas/turan-c42-bound third_party/bakeas && git -C third_party/bakeas checkout dc9c491ccb0a9ee8de753739eb082fa84c25c091   # tag v1.1.0, MIT
git clone https://github.com/rpinho/c42-certificate third_party/pinho && git -C third_party/pinho checkout 009120a811abd6840a9417e9f1c02cd490c5f608   # PR #305 bundle, MIT
git clone https://github.com/DennisKoshta/c42-turan-power-sum-certificate third_party/koshta   # read at 8e1797e0d35454ad18cfebbbeb5858bf202b0748
git clone https://github.com/teorth/optimizationproblems third_party/optimizationproblems   # read at 87b4ebbd426ff8db597a2004cc45da17903bdcf2
```

All credit for the code and mathematics in these repositories goes to their authors (Bakeas; Pinho; Koshta;
Tao et al. for the table). Nothing from them is copied into src/; src/evaluator.py uses the same
product-integration idea as Bakeas's `crosscheck/limit_direct.py`.
