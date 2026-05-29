export GIT_PAGER=cat
rm -rf /tmp/uv-test
mkdir /tmp/uv-test
# to get ascii tree listings
shopt -s expand_aliases
alias tree='tree --charset=ascii'
cd /tmp/uv-test
set -x
PS4='\n$ '

if command -v deactivate >/dev/null 2>&1; then
    deactivate
fi
if [ -n "$CONDA_DEFAULT_ENV" ]; then
    conda deactivate
fi


uv init --package uv-example
tree uv-example

cd uv-example
uv venv
tree .venv


{ set +x; } 2>/dev/null
source .venv/bin/activate
set -x
echo "$ source .venv/bin/activate"
which python


uv add numpy

cat pyproject.toml

uv pip compile pyproject.toml -o requirements.txt