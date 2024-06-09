.2::raining.
.5::happy(P).
.05::school_test.
.1::fun_class.
.3::happy(P) :- fun_class.
.7::\+happy(P) :- raining.
.9::\+happy(P) :- school_test.

% evidence(alice_is_happy).
% evidence(fun_class).
% query(alice_is_happy).

friend(alice, charlie). % Alice is friends with Charlie
friend(charlie, bob).
friend(dorothy, emily).
friend(X,Y) :- Y @< X, friend(Y,X). % symmetry of friendship
friend(X,Y) :- X\=Y, friend(X,Z), friend(Z,Y). % transitivity of friendship
.8::happy(P):- friend(P,Q), happy(Q).

evidence(happy(bob)).
evidence(\+ happy(charlie)).
evidence(school_test).

query(happy(alice)).