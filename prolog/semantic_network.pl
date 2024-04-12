/*---------------------------------------------------------------*/
/* Telecom Paristech - J-L. Dessalles 2009                       */
/*            http://teaching.dessalles.fr                       */
/*---------------------------------------------------------------*/

%--------------------------------
%       semantic networks
%--------------------------------

isa(bird, animal).
isa(albert, albatross).
isa(albatross, bird).
isa(kiwi, bird).
isa(willy, kiwi).
isa(crow, bird).
isa(ostrich, bird).

:- dynamic(locomotion/2).    % for tests

locomotion(bird, fly).
locomotion(kiwi, walk).
locomotion(X, Loc) :-
    isa(X, SuperX),
    locomotion(SuperX, Loc).

food(albatross,fish).
food(bird,grain).

/* drawback : n particular inheritance rules */
/* solution: one general predicate : "known" */

% if Fact exists for child concept then just return
known(Fact) :- 
    Fact,
    !.
% if Fact doesn't exist for child concept then return its parent fact.
known(Fact) :-
    Fact =.. [Rel, Arg1, Arg2],
    isa(Arg1, SuperArg1),
    SuperFact =.. [Rel, SuperArg1, Arg2],
    known(SuperFact).

habitat(Animal, Location):-
    known(locomotion(Animal, M)),
    not(M = fly),
    Location = continent.

habitat(_, unknown).