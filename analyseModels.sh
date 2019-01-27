#!/bin/bash
shopt -s nullglob
output="results/mcc.out"
#echo "Filename,Places,Transitions,Arcs,Ordinary,Simple_free_choice,Extended_free_choice,State_machine,Marked_graph,Connected,Strongly_connected,Source_place,Sink_place,Source_transitions,Sink_transitions,Loop_free,Conservative,Subconservative,Nested_units,Safe,Deadlock,Quasi_live,Live,Markings,Firings,Max_tokens_place,Max_tokens_marking,Dead_transitions,Concurrent_Units,Exclusives" > $output
for file in models/*.nupn
do
	echo "$file"
	echo "$file" | sed 's/models\///g' | sed 's/.nupn//g' | tr '\n' ' ' >> $output
	caesar.bdd -mcc "$file" | awk -F '[{]' '{for (i=2; i<NF; i++) printf $i " "; print $NF}' | sed 's/\\//g' | sed 's/Explain[^}}]*//g' \
	| awk -F ' %' '{print $1}' | sed 's/\$geq\$ />=/g' | sed 's/\$leq\$ /<=/g' | sed 's/}//g' | sed 's/True/1/g' | sed 's/False/0/g' \
	| sed 's/Unknown/\-1/g' | sed 's/PlacesTransitionsArcs //g' | sed 's/MarkingsFiringsBoundConcurrency //g' \
	| sed 's/ /,/g' | tr '\n' ',' >> $output
	DEAD="$(caesar.bdd -dead-transitions "$file")"
	AMOUNTDEAD="$(echo $DEAD | tr -cd '1' | wc -c)"
	TOTALD="$(echo $DEAD | wc -c)"
	echo "scale = 5; $AMOUNTDEAD / $TOTALD" | bc | tr '\n' ',' >> $output
	CONCURRENTUNITS="$(caesar.bdd -concurrent-units "$file")"
	ONES="$(echo $CONCURRENTUNITS | tr -cd '1' | wc -c)"
	TOTALC="$(echo $CONCURRENTUNITS | wc -c)"
	echo "scale = 5; $ONES / $TOTALC" | bc | tr '\n' ',' >> $output
	EXCLUSIVEPLACES="$(caesar.bdd -exclusive-places "$file")"
	EXCLUSIVES="$(echo $EXCLUSIVEPLACES | tr -cd '1=<>' | wc -c)"
	UNKNOWN="$(echo $EXCLUSIVEPLACES | tr -cd '~[].' | wc -c)"
	TOTALE="$(echo $EXCLUSIVEPLACES | wc -c)"
	echo "scale = 5; ($EXCLUSIVES + 0.5 * $UNKNOWN) / $TOTALE" | bc >> $output
	mv "$file" models/analysed/
done
shopt -u nullglob
#| sed 's/\$in\$ //g'

