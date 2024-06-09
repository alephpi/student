.4::coin(heads,Id); .6::coin(tails,Id).

flip(PastFlips,[Flip|PastFlips]) :-
    length(PastFlips,N),
    coin(Flip,N).

next(Marbles,Stake,Flips,Result,Sequence) :- % Bob loses and has no marbles left
    flip(Flips,[tails|Flips]),
    Stake1 is 2*Stake,
    Marbles < Stake1,
    Result is Marbles,
    Sequence = [tails|Flips].
next(Marbles,Stake,Flips,Result,Sequence) :- % Bob wins
    flip(Flips,[heads|Flips]),
    Result is Marbles+2*Stake,
    Sequence = [heads|Flips].
next(Marbles,Stake,Flips,Result,Sequence) :- % Bob loses but keeps playing
    flip(Flips,[tails|Flips]),
    Stake1 is 2*Stake,
    Marbles >= Stake1,
    Marbles1 is Marbles - Stake1,
    next(Marbles1,Stake1,[tails|Flips],Result,Sequence).
game(Win) :-
    Initial is 15,
    Stake is 1,
    Marbles is Initial-Stake,
    next(Marbles,Stake,[],Result,L),
    Win is Result-Initial.

query(game(Win)).

% % next(+Marbles,+Stake,+Flips,-Result,-Sequence) :- 
% %     flip(Flips, [NewFlip|Flips]),
% %     (
% %         NewFlip = heads,
% %         Marbles is Marbles + Stake,

%         );
%     (
%         NewFlip = tails,
%         Marbles is Marbles - Stake,
%         )
% game(Win) :-