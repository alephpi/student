% genealogy
parent(marge, lisa).
parent(marge, bart).
parent(marge, maggie).
parent(homer, lisa).
parent(homer, bart).
parent(homer, maggie).
parent(abraham, homer).
parent(abraham, herb).
parent(mona, homer).
parent(jackie, marge).
parent(clancy, marge).
parent(jackie, patty).
parent(clancy, patty).
parent(jackie, selma).
parent(clancy, selma).
parent(selma, ling).

female(mona).
female(jackie).
female(marge).
female(ann).
female(patty).
female(selma).
female(ling).
female(lisa).
female(maggie).
male(abraham).
male(herb).
male(homer).
male(bart).
male(clancy).



child(X,Y) :-
    parent(Y,X).

mother(X,Y) :-
    parent(X,Y),
    female(X).

grandparent(X,Y) :-
    parent(X,Z), % note that the a variable's scope is the clause
    parent(Z,Y). % variable Z keeps its value within the clause

sister(X,Y) :-
    parent(Z,X), % if X gets instantiated, Z gets instantiated as well
    parent(Z,Y),
    female(X),
    X \== Y. % can also be noted: not(X = Y).
    
ancestor(X,Y) :-
    parent(X,Y).
ancestor(X,Y) :-
    parent(X,Z),
    ancestor(Z,Y). % recursive call


aunt(X,Y) :-
    sister(X,Z),
    parent(Z,Y).

descendant(X,Y) :-
    ancestor(Y,X).

% list processing
extract(X, [X|L], L).
extract(X, [Y|L], [Y|L1]) :-
    extract(X, L, L1).

% reversible

permute([],[]).
permute([First|Rest], PermutedList) :-
    permute(Rest, PermutedRest),
    extract(First, PermutedList, PermutedRest).

last_elt([X], X).
last_elt([_|L], Y) :-
    last_elt(L, Y).

attach([X],Y,[X|Y]).
attach([X|H],Y,[X|Z]) :-
    attach(H,Y,Z).

assemble(L1,L2,L3,Result) :-
    attach(L1, L2, L1L2),
    attach(L1L2, L3, Result).
% reversible but will overflow after 
% X = [1, 2, 3, 4, 5, 6],
% Y = Z, Z = [] ;

sub_list(IncludedList, IncludingList) :-
    attach(IncludedList, _, IncludingList);
    sub_list([_|IncludedList], IncludingList).
% return true when it is a sublist but infinity loop when not a sublist.

% operators
% test :-
%     [+(1,2) == 3,
%     +(1,2) = 3,
%     +(1,2) \== 3,
%     +(1,2) \= 3].

% syntax equiv == and semantic unification =

test_equiv(X, Y, Z, V):- p(X, b(Z,a), X) = p(Y, Y, b(V,a)).
test_equiv:- p(X, b(Z,a), X) = p(Y, Y, b(V,a)).
test_unif :- p(X, b(Z,a), X) == p(Y, Y, b(V,a)).

% arithmetic needs evalutation by is.
eval1 :- X is +(1,2), X == 3.
eval1bis :- +(1,2) == 3.
eval1ter :- +(1,2) == +(1,2).
eval2 :- Y is +(1,2), Y = 3.
eval2bis :- +(1,2) = 3.
eval2ter :- +(1,2) = +(1,2).
eval3 :- Z is +(1,2), Z \== 3.

% cut

% member(X, [X]).
member(X, [Y|H]):-
    Y = X;
    member(X, H).

add(X, L, L):-
    member(X, L),
    !.
add(X, L, [X|L]).
% remove(Ele, [Ele|H], Rest) :-
%     remove(Ele, H, Rest);

remove(_, [], []).

% if match no backtracking for other rules, therefore we don't need to explicitly write out H\=X for the other rule.
% make me think of short-circuit if-condition.
remove(X, [X|H], Result) :-
    !,
    remove(X, H, Result).

remove(X, [H|T], [H|Result]):-
    remove(X, T, Result).

% the following equiv to not(female(X)) since if female(X) is true, it stops backtracking and fail.
q(X):-
    female(X),!,fail.

% if female is false, it goes here and q(X) is always true.
q(X).

% order of clauses matters
reasonable(Restaurant) :-
    not(expensive(Restaurant)).
fancy(belle_d_argent).
fancy(maximus).
expensive(belle_d_argent).

% return X=maximus
test1(X) :- fancy(X), reasonable(X).
% return false
%[trace]  ?- test2(X).
%    Call: (12) test2(_107064) ? creep
%    Call: (13) reasonable(_107064) ? creep
% ^  Call: (14) not(expensive(_107064)) ? creep
%    Call: (15) expensive(_107064) ? creep
%    Exit: (15) expensive(belle_d_argent) ? creep
% ^  Fail: (14) not(user:expensive(_107064)) ? creep
%    Fail: (13) reasonable(_107064) ? creep
%    Fail: (12) test2(_107064) ? creep
% false.
% the expensive(belle_d_argent) clause will be first executed then fails and no fancy(X) will be further executed. 
% while in test1(X), the fancy(X) clause will be first executed so X=maximus will be tried.
test2(X) :- reasonable(X), fancy(X).

% duplicate
duplicate([X], [X,X]).
duplicate([X|L], [X,X|LL]):-
    duplicate(L, LL).