#!/bin/bash

rsync -avz --exclude-from='.rsync-exclude' \
    /Users/heleyang/Code/ai_infra/cs336/assignment1-basics \
    heleyang@10.110.59.61:~/Code/cs336
