# GPU-based Multi-way Join Operators

Accelerating the evaluation of multi-table join queries in databases and graphs using multi-way and worst-case optimal joins.

Folder | Description
--- | ---
[general-join-gpu](LFTJ-GPU)| BSF and DFS-based LFTJ implementations on GPUs
[hash-chain-join-gpu](MHJ-GPU)| Hash-based chain join on GPUs
[hash-chain-join-cpu](MHJ-CPU)| Hash-based chain join on CPUs
[dependencies](dependencies) | third-party libraries used in the project

## Init dependencies

```zsh
//to initialize the ModernGPU submodules
git submodule init
git submodule update
```

