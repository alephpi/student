% The first argument of 'op' gives precedence (converse of priority).
:-op(140, fy, -).        % 'fy' means right-associative: '- -X' means '-(-X)'
:-op(160,xfy, [and, or, equiv, imp, impinv, nand, nor, nonimp, equiv, nonimpinv]).    
    % 'xfy' means that: X and Y and Z = X and (Y and Z)
    % Note that '-' has priority over 'and', due to its lower precedence.

% V here represents an evaluation
% so is_true(V, proposition) reads as a query "is proposition true under evaluation V?"
is_true(V, X and Y) :- is_true(V,X), is_true(V,Y).
is_true(V, X or _) :- is_true(V,X).
is_true(V, _ or Y) :- is_true(V,Y).
is_true(V, -X) :-
    not(is_true(V, X)). % link with Prolog's negation
is_true(v0,a).        % this means that v0 sends a to True and everything else (here, b and c) to false

% test is_true(v0, a and -b).
% will return true because finally prolog tries to evaluate is_true(b)
% but as close world assumption, it cannot prove it 
% so it will evaluate is_true(b) as false

% we just make use of the tautologies: X imp Y === -X or Y, X equiv Y === (X and Y) or (-X and -Y)
is_true(V, X imp Y) :-
    is_true(V, -X or Y).
is_true(V, X equiv Y) :-
    is_true(V, (X and Y) or (-X and -Y)).

% we can write also, i.e. we expand the evaluation morphism in semantic space
% is_true(V, X equiv Y) :- is_true(V,X), is_true(V,Y).
% is_true(V, X equiv Y) :- not(is_true(V,X)), not(is_true(V,Y)).

% is_true(V, X imp Y) :- is_true(V,X), is_true(V,Y).
% is_true(V, X imp _) :- not(is_true(V,X)).

% try to solve the query is_true(V,X), i.e. list all evaluations that makes X true.
% so if we are going to prove a tautology in this way, we should find that V returns all possible evaluations
is_true(V, X) :-
	member(X, V).	% only true elements are explicitly mentioned in V

valuation(V) :-
	% we keep all elements that V sends to true.
	% all other elements are supposed to be false.
	sub_set(V, [a,b,c]).	
	
sub_set([], []).
sub_set([X|XL], [X|YL]) :-
    sub_set(XL, YL).
sub_set(XL, [_|YL]) :-
    sub_set(XL, YL).

% we test
% de morgan law(a ⊃ (b ⊃ c)) ⊃ ((a ⊃ b) ⊃ (a ⊃ c))
% and Frege's axiom (((a ⊃ b) ∧ (b ⊃ c)) ⊃ ¬(¬c ∧ a))
test_morgan(V) :- valuation(V), is_true(V, (a imp (b imp c)) imp ((a imp b) imp (a imp c))).
test_frege(V) :- valuation(V), is_true(V, ((a imp b) and (b imp c)) imp -(-c and a)).

/* table for unary, alpha and beta formulas */

components(- -X, X, _, unary).
components(X and Y, X, Y, alpha).
components(-(X or Y), -X, -Y, alpha).
components(X or Y, X, Y, beta).
components(-(X and Y), -X, -Y, beta).
components(X imp Y, -X, Y, beta).
components(-(X imp Y), X, -Y, alpha).
components(X impinv Y, X, -Y, beta).
components(-(X impinv Y), -X, Y, alpha).
components(X nand Y, -X, -Y, beta).
components(-(X nand Y), X, Y, alpha).
components(X nor Y, -X, -Y, alpha).
components(-(X nor Y), X, Y, beta).
components(X nonimp Y, X, -Y, alpha).
components(-(X nonimp Y), -X, Y, beta).
components(X nonimpinv Y, -X, Y, alpha).
components(-(X nonimpinv Y), X, -Y, beta).


% Predicate cnf puts more elementary processing together
cnf(Conjunction, NewConjunction) :-
	oneStep(Conjunction, C1),
	cnf(C1, NewConjunction).
cnf(C, C).


% Predicate oneStep performs one elementary processing
oneStep([Clause | Rest_Conjonction], [ [F1, F2 | Rest_Clause] | Rest_Conjonction]) :-
	% looking for a beta formula in the clause
	remove(F1 or F2, Clause, Rest_Clause).

oneStep([Clause | Rest_Conjonction], [ [F1 | Rest_Clause], [F2 | Rest_Clause] | Rest_Conjonction]) :-
	% looking for an alpha formula in the clause
	remove(F1 and F2, Clause, Rest_Clause).

oneStep([Clause | Rest_Conjonction], [ [F | Rest_Clause] | Rest_Conjonction]) :-
	% looking for a beta formula in the clause
	remove(- -F, Clause, Rest_Clause).

oneStep([ F | Rest], [ F | New_Rest ]) :-
	% nothing left to do on F
	oneStep(Rest, New_Rest).

% we decompose the cnf by two steps, first rewriting all binary op to and, or, not
% then we do oneStep
% rewrite(Old_Clause, New_Clause) :-
%     components(Old_Clause, X, Y, alpha),
%     rewrite(X, W),
%     rewrite(Y, Z),
%     New_Clause = W and Z.

% rewrite(Old_Clause, New_Clause) :-
%     components(Old_Clause, X, Y, beta),
%     rewrite(X, W),
%     rewrite(Y, Z),
%     New_Clause = W or Z.

% rewrite(Old_Clause, New_Clause) :-
%     components(Old_Clause, X, _, unary),
%     rewrite(X, New_Clause).

% rewrite(Old_Clause, New_Clause) :-
%     New_Clause = Old_Clause.

% test(X):- rewrite(-((a imp ( b imp c)) imp ((a imp b) imp (a imp c))), X).
% test_cnf(Result):- rewrite(-((a imp ( b imp c)) imp ((a imp b) imp (a imp c))), X), cnf([[X]], Result).

% Or, we can do it in oneStep, that we rewrite according to the formula
oneStep([Clause | Rest_conjunction], [ [F1, F2 | Rest_Clause] | Rest_conjunction]) :-
    % looking for a beta formula in the clause
    remove(BetaFormula, Clause, Rest_Clause),    
    components(BetaFormula, F1, F2, beta).

oneStep([Clause | Rest_conjunction], [ [F|Rest_Clause] | Rest_conjunction]) :-
    % looking for a unary formula in the clause
    remove(UnaryFormula, Clause, Rest_Clause),    
    components(UnaryFormula, F, _, unary).

oneStep([Clause | Rest_conjunction], [ [F1|Rest_Clause], [F2|Rest_Clause] | Rest_conjunction]) :-
    % looking for an alpha formula in the clause
    remove(AlphaFormula, Clause, Rest_Clause),    
    components(AlphaFormula, F1, F2, alpha).

oneStep([ F | Reste], [ F | New_Rest ]) :-
    % nothing left to do on F
    oneStep(Reste, New_Rest).

% i.e.
/*------------------------------------------------*/
/* Auxiliary predicates                           */
/*------------------------------------------------*/

/* remove does as select, but removes all occurrences of X */
remove(X, L, NL) :-
    member(X,L),	% so that remove fails if X absent from L
    remove1(X, L, NL).
remove1(X, L, L) :-
    not(member(X,L)).
remove1(X, L, NL) :-
	select(X, L, L1),   % available in SWI-Prolog
	remove1(X, L1, NL).