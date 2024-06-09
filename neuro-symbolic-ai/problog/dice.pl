% :- use_module(library(lists)).
% die(ID, Val) :- select_uniform(ID, [1,2,3,4,5,6], Val, _).
t(1/6)::die(Id,1); t(1/6)::die(Id,2); t(1/6)::die(Id,3); t(1/6)::die(Id,4); t(1/6)::die(Id,5); t(1/6)::die(Id,6).
roll(Outcome) :-
    die(1,Val1),
    die(2,Val2),
    Outcome is Val1+Val2.
% query(roll(X)).