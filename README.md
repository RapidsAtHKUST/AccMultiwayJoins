# GPU-based Multi-way Join Operators

Accelerating the evaluation of multi-table join queries in databases and graphs using multi-way join (AMHJ) and worst-case optimal join (ALFTJ).

## Compile

Compile the dependency `cub` before compiling our code.

```shell
mdkir dependencies/cudpp/build
cd dependencies/cudpp/build
cmake ..
make -j16
cd ../../..
```

Then, compile our code by the following commands.

```shell
mkdir MHJ-GPU/build
cd MHJ-GPU/build
cmake ..
make -j16
cd ../..
mkdir LFTJ-GPU/build
cd LFTJ-GPU/build
cmake ..
make -j16
cd ../..
```

The versions of the software we use is listed as follows.

```
cmake: 3.20.2
Make: 4.2.1
GCC: 8.5.0
cuda: 10.2
```

## Execute

Use the following command to run AMHJ.

```shell
./MHJ-GPU/build/exec-AMHJ <query-type> <dataset-path> <ooc> <ws> <dro> <fib>
```

The description and options of each parameter are listed in the following table.

| Parameters   | Description                            | Valid Value     |
|--------------|----------------------------------------|-----------------|
| query-type   | Type of query                          | NORMAL/TRI/FOUR |
| dataset-path | Path of the query                      | N/A             |
| ooc          | Disable/Enable out of core support     | 0/1             |
| ws           | Disable/Enable work sharing            | 0/1             |
| dro          | Disable/Enable direct result output    | 0/1             |
| fib          | Disable/Enable frequent item buffering | 0/1             |


Use the following command to run ALFTJ.

```shell
./LFTJ-GPU/build/exec-LFTJ <dataset-path> <algo-type> <ooc> <ws>
```

The description and options of each parameter are listed in the following table.

| Parameters   | Description                        | Valid Value |
|--------------|------------------------------------|-------------|
| dataset-path | Path of the query                  | N/A         |
| algo-type    | Type of algorithm (BFS/DFS)        | 0/1         |
| ooc          | Disable/Enable out of core support | 0/1         |
| ws           | Disable/Enable work sharing        | 0/1         |

