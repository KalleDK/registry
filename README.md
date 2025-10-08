# registry

Registry for packages

[https://kalledk.github.io/registry/](https://kalledk.github.io/registry/)

## To update fetcher pkgs

```bash
gh repo clone KalleDK/registry

git checkout repo

task update

# If any changes

git add .

git commit -m msg

./squash.sh

git push -f

git remote rm origin

git -c gc.reflogExpire=0 -c gc.reflogExpireUnreachable=0 -c gc.rerereresolved=0 -c gc.rerereunresolved=0 -c gc.pruneExpire=now gc

```

## To change in the tools
```bash
gh repo clone KalleDK/registry

# change tool

git add .

git commit -m "msg"

git push

git fetch && git checkout repo && git reset --hard origin/repo && ./squash.sh && git push -f && git checkout main

git remote rm origin

git -c gc.reflogExpire=0 -c gc.reflogExpireUnreachable=0 -c gc.rerereresolved=0 -c gc.rerereunresolved=0 -c gc.pruneExpire=now gc


```
