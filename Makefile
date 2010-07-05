all: celt silk
	(cd src; make)

celt:
	(cd celt; make)

silk:
	(cd celt; silk)

