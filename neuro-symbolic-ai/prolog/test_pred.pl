a :- b.
a :- c, d.
:- a, b, c.

test1 :-
    clause(a, X), write(X), nl.
