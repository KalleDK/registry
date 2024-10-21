#!/usr/bin/env bash

EDITOR="sed -i '2,/^\$/s/^pick\b/s/'" git rebase -i origin/main
