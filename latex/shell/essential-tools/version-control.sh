export GIT_PAGER=cat
rm -rf /tmp/git-test
set -x
PS4='\n$ '



cd /tmp
mkdir git-test
cd git-test
git init
echo "test file" > test_file.txt
git status


git add test_file.txt
git status


git log

git commit -m"initial add"

git status


git log

echo "another line" >> test_file.txt # add another line of text
git diff test_file.txt

git add test_file.txt
git commit -m"added a second line"
git log

git branch
git checkout -b feature/swap-db-backend
git branch
