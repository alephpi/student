
:- consult(propositional).

prove(F) :-
    cnf([[ -F ]], CNF),
    write('CNF of -['), write(F), write('] = '),
    write(CNF), nl,
    resolve(CNF).


resolve(CNF) :-
    write('Examining '), write(CNF), nl,
    get0(_),    % waits for user action
    resolve_empty_clause(CNF);resolve_between_two_clauses(CNF).

resolve_empty_clause(CNF) :-
    member([ ], CNF),
    write('This is a true formula'), nl.

resolve_between_two_clauses(CNF) :-
    % spy(select),
    select(C1, CNF, RCNF1),            % forgetting this parent clause
    select(C2, RCNF1, RCNF2),    % keeping this parent clause
    % write('C1 ='), write(C1), nl,
    % write('C2 ='), write(C2), nl,
    remove(X, C1, C1_Rest), remove(-X, C2, C2_Rest),
    write('Annihilating atom:'), write(X), nl,
    write('C1_Rest, C2_Rest'), write(C1_Rest), write(C2_Rest), nl,
    % append(C1_Rest, C2_Rest, C_New),
    merge(C1_Rest, C2_Rest, C_New),
    write('New clause joined:'), write(C_New), nl,
    resolve([C_New|RCNF2]).
    % user 'remove' from cnf program

% resolve(CNF) :-
%     member([ ], CNF),
%     write('This is a true formula'), nl.

% resolve(CNF) :-
%     write('Examining '), write(CNF), nl,
%     get0(_),    % waits for user action
%     select(C1, CNF, _),
%     select(C2, CNF, RCNF),
%     remove(P, C1, RC1),
%     remove(-P, C2, RC2),
%     append(RC1, RC2, RC),
%     resolve([RC|RCNF]).    % one of the parent clauses is removed

% this solution may loop. For instance: prove( a and -a ).
% Loops are unavoidable in some cases. A possibility is to use memory to avoid cycles.

go :-
    prove( ((a imp (b imp c)) imp ((a imp b) imp (a imp c))) ).

go1 :-
    prove(a and -a).

go2 :-
    prove( ((a and -a) or (b and -b) or (c and d) or (-d) or (c))).

% https://stackoverflow.com/questions/66449005/combining-two-lists-in-prolog
% merging list without repetitions to get better output print.
merge(A, B, C):-
    append(A1, Common, A),
    append(Common, B1, B),
    !,  % The cut here is to keep the longest common sub-list
    append([A1, Common, B1], C).