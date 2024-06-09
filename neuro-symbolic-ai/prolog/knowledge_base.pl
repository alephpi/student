:- dynamic(sunshine/0). % necessary with SWI-Prolog.
:- dynamic(raining/0). % necessary with SWI-Prolog.
:- dynamic(fog/0). % necessary with SWI-Prolog.

nice :-
    sunshine, not(raining).

funny :-
    sunshine, raining.
disgusting :-
    raining,fog.
raining.
fog.

empty(Predicate) :-
    retract(Predicate),
    % force backtracking to retract the other Predicate till retract return false.
    fail.

% base case always return true.
empty(_).

:- dynamic(found/1).
findany(Var, Pred, Results) :-
    Pred,
    assert(found(Var)),
    fail.

% if Pred fails, findany start to collect found
findany(_, _, Results) :-
    collect_found(Results).

collect_found([X|Results]) :-
    retract(found(X)),
    collect_found(Results),
    !. %no need to backtracking.

collect_found([]).