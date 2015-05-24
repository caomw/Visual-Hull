: delete unnecessary files before pull or push

rmdir /s/q Debug

rmdir /s/q ipch

rmdir /s/q main\Debug

del Visual-Hull.sdf

git add --all

git commit -m "Up"

git push