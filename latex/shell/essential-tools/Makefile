all: uv.out version-control.out

uv.out:
	bash uv.sh > uv.out 2>&1

version-control.out:
	bash version-control.sh > version-control.out 2>&1

clean:
	-rm *.out
