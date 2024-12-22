- follow the tuts
  - Why is the article crazy about data being aligned and how can it improve performance
    as author uses aligned directive for mask bytes also

- after tuts, do compile with -O3 and see difference between original and optimise code
  it runs very fast, check the reason

- It seems that numpy code is giving me around 33 GFLOPS

- look for fast very fast matrix transpose and then multiply matrix by normally and see it's
  effect. Is it worth of taking extra cycles for transpose for matrix mul -> if it runs faster
  than any written code

- good inference code: andrewkchan.dev/posts/yalm.html

Future:
- test time compute for assembly creator llm
