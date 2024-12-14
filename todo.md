- follow the tuts
  - first completing the masking part
  - Then write column major and row major in one file and then put
    all kernel size in arrays and for each size -> check the average flops
    (first reason if comparing directly in one file make sense for both order)
    Take different matrix sizes as per std input

    Then lastly take all graph and find either to go with row or column based
    and then also accounting the fact about which kernel size gives highest flops

    Remember to clean col_kernel_matmul.c to have important function in different files
  - Why is the article crazy about data being aligned and how can it improve performance
    as author uses aligned directive for mask bytes also
