% monkey box banana
/*---------------------------------------------------------------*/
/* Telecom Paris- Nils Hozenberger - J-L. Dessalles        2024  */
/* Logic and knowledge representation                            */
/*            http://ia206.simplicitytheory.science              */
/*---------------------------------------------------------------*/

% adapted from I. Bratko - "Prolog - Programming for Artificial Intelligence"
%              Addison Wesley 1990
% An ape is expected to form a plan to grasp a hanging banana using a box.
% Possible actions are 'walk', 'climb (on the box)', 'push (the box)', 
% 'grasp (the banana)'
% description of actions - The current state is stored using a functor 'state'
% with 4 arguments: 
%	- horizontal position of the ape 
%	- vertical position of the ape
%	- position of the box
%	- status of the banana 
% 'action' has three arguments: 
% 	- Initial state
%	- Final state
%	- act

% action(state(X,floor,T,Z), walk(X,Y), state(Y,floor,T,Z)).
% if put walk at first, it will walk endlessly, since walk will never take the monkey to the success state
action(state(middle,on_box,X,not_holding), grasp, state(middle,on_box,X,holding)).
action(state(X,floor,X,Y), climb, state(X,on_box,X,Y)).
action(state(X,floor,X,Z), push(X,Y), state(Y,floor,Y,Z)).
action(state(X,floor,T,Z), walk(X,Y), state(Y,floor,T,Z)).

% success(state(_,_, _, holding)).
% success(State1) :- 
% 	action(State1, A, State2),
% 	write('Action : '), write(A), nl, write(' --> '), write(State2), nl,
% 	success(State2).
% go :-
% 	success(state(door, floor, window, not_holding)).

success(state(_,_, _, holding), []).
success(State1, [action(State1, A, State2)|Plan]) :- 
	action(State1, A, State2),
	% write('Action : '), write(A), nl, write(' --> '), write(State2), nl,
	success(State2, Plan),
	!.
	
go(Plan) :-
	success(state(door, floor, window, not_holding), Plan).

% Accumulation
mirror([ ], [ ]).
mirror([X|L1], L2) :-
    mirror(L1,L3),
    append(L3, [X], L2).

% better solution with accumulator:
mirror2(Left, Right) :-
    invert(Left, [ ], Right).
invert([X|L1], L2, L3) :-    % the list is 'poured'
    invert(L1, [X|L2], L3).    % into the second argument
invert([ ], L, L).        % at the deepest level, the result L is merely copied

palindrome(X) :-
    mirror2(X, X).

% list of even length
palindrome2(Begin, End) :-
    Begin = End,
    !.

% list of odd length
palindrome2([_|Begin], End) :-
    Begin = End,
    !.

palindrome2([X|L1],L2) :-
    palindrome2(L1, [X|L2]).

palindrome2(X) :-
    palindrome2(X, []).