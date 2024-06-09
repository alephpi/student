0.060059318482181::start(0); 0.0::start(1); 0.939940681517819::start(2).
0.0::emit(Time,0,'cloudy-dry'); 0.260588802367778::emit(Time,0,'sunshine'); 0.739411197632222::emit(Time,0,'cloudy-rainy'); 0.0::emit(Time,0,'fog'); 0.0::emit(Time,0,'foggy-rainy').
0.999999999906301::emit(Time,0,'sunshine+rain').
0.0::emit(Time,1,'cloudy-dry'); 0.238685686794105::emit(Time,1,'sunshine'); 0.761314313205895::emit(Time,1,'cloudy-rainy'); 0.0::emit(Time,1,'fog'); 0.0::emit(Time,1,'foggy-rainy').
1.0::emit(Time,1,'sunshine+rain').
0.152649414855631::emit(Time,2,'cloudy-dry'); 0.317024376639386::emit(Time,2,'sunshine'); 0.295163596430093::emit(Time,2,'cloudy-rainy'); 0.169152054299483::emit(Time,2,'fog'); 0.066010557775408::emit(Time,2,'foggy-rainy').
1.0::emit(Time,2,'sunshine+rain').
0.502155100063301::trans(Time,0,0); 0.46829558672445::trans(Time,0,1); 0.029549313212249::trans(Time,0,2).
0.137107551865758::trans(Time,1,0); 0.330839974633064::trans(Time,1,1); 0.532052473501178::trans(Time,1,2).
0.005976859341441::trans(Time,2,0); 0.041127916803413::trans(Time,2,1); 0.952895223855145::trans(Time,2,2).
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
