shopt -s nullglob
for file in models/*.pnml
do
	java -jar tools/pnml2nupn-1.5.3.jar "$file"
done
shopt -u nullglob
