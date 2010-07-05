all: celt silk
	(cd src; make)

celt:
	(cd celt; make)

silk:
	(cd silk; make)

