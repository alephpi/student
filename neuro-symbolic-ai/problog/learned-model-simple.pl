0.5::start(0); 0.5::start(1).
0.316239316239316::emit(Time,0,'cloudy-dry'); 0.683760683760684::emit(Time,0,'sunshine').
0.595744680851064::emit(Time,1,'cloudy-rainy'); 0.290780141843972::emit(Time,1,'fog'); 0.113475177304965::emit(Time,1,'foggy-rainy').
emit(_,2,'sunshine+rain').
0.528735632183908::trans(Time,0,0); 0.28735632183908::trans(Time,0,1); 0.183908045977011::trans(Time,0,2).
0.170454545454545::trans(Time,1,0); 0.602272727272727::trans(Time,1,1); 0.227272727272727::trans(Time,1,2).
0.24::trans(Time,2,0); 0.52::trans(Time,2,1); 0.24::trans(Time,2,2).
state/2.
trans/3.
emit/3.
observe/2.
observe_sequence/1.
observe_sequence_aux/1.
state_sequence/1.
state_sequence_aux/1.
observe(Time,Symbol) :- state(Time,State), emit(Time,State,Symbol).
print([]) :- nl.
print(X) :- write(X), nl.
print(X,Y) :- write(X), write(' - '), write(Y), nl.
observe_sequence_aux(Time,State,[Symbol | Tail]) :- trans(Time,State,State1), Time1 is Time+1, emit(Time1,State1,Symbol), observe_sequence_aux(Time1,State1,Tail).
observe_sequence_aux(_,_,[]).
observe_sequence([First | Rest]) :- start(Start_state), emit(0,Start_state,First), observe_sequence_aux(0,Start_state,Rest).
state_sequence_aux(_,_,[]).
state_sequence_aux(Time,State,[State1 | Rest]) :- trans(Time,State,State1), Time1 is Time+1, state_sequence_aux(Time1,State1,Rest).
state_sequence([Start | Rest]) :- start(Start), state_sequence_aux(0,Start,Rest).
state(0,State) :- start(State).
state(Time,State) :- Time>0, Previous is Time-1, state(Previous,Previous_state), trans(Previous,Previous_state,State).
generate_sequence(L,N) :- length(L,N), observe_sequence(L).
