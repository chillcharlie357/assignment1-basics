#!/bin/bash

rsync -avz -e 'ssh -p 46497' --exclude-from='.rsync-exclude' \
    /Users/heleyang/Code/ai_infra/cs336/assignment1-basics \
    root@northwest1.gpugeek.com:~/cs336/