- follow the tuts
  - I don't understand why even though it loads data to xmm0 register
    then still the result is correct as the register is overwrite by new values
    like it is using one register
    let's look at assembly compiled code and understand kernel_12x4 assembly fully

  - Why is the article crazy about data being aligned and how can it improve performance
    as author uses aligned directive for mask bytes also

- after tuts, do compile with -O3 and see difference between original and optimise code
  it runs very fast, check the reason

- It seems that numpy code is giving me around 33 GFLOPS

- look for fast very fast matrix transpose and then multiply matrix by normally and see it's
  effect. Is it worth of taking extra cycles for transpose for matrix mul -> if it runs faster
  than any written code

- `notes/cache_matmul.md` delve into the reason why it is fast, let's not use vague terms and really
  understand why and how it works.

Future:

- test time compute for assembly creator llm

- good inference code: andrewkchan.dev/posts/yalm.html
