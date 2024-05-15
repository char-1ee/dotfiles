### Single node 8 cards
```shell
# All-reduce
torchrun --nproc_per_node 8 test_dist.py -b 2M -e 256M -f 2

# H2D
torchrun --nproc_per_node 8 test_move.py -b 2M -e 256M -f 2
```

### 10x8 cards
```shell
# All-reduce
torchrun --nproc_per_node 8 test_dist.py -b 2M -e 256M -f 2

# H2D
torchrun --nproc_per_node 8 test_move.py -b 2M -e 256M -f 2
```